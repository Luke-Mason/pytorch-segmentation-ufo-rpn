import datetime
import json
import os
import time
import datetime

import numpy as np
import torch
from base import BaseTrainer, DataPrefetcher
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from utils import transforms as local_transforms
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
import logging

class DSTLTrainer(BaseTrainer):
    def __init__(self, start_time, model, loss, resume, config, train_loader,
                            k_fold = None,
                 val_loader=None, train_logger=None, prefetch=False, root='.'):
        super(DSTLTrainer, self).__init__(start_time, model, loss, resume,
                                          config, train_loader, k_fold,
                                      val_loader, train_logger, root)
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(
            self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']: self.log_step = int(
            self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        self.min_clip_percentile = 2
        self.max_clip_percentile = 98

        # if self.device == torch.device('cpu'): prefetch = False
        # if prefetch:
        #     self.train_loader = DataPrefetcher(train_loader, device=self.device)
        #     self.val_loader = DataPrefetcher(val_loader, device=self.device)

        torch.backends.cudnn.benchmark = True

    def dra(self, array: np.ndarray) -> np.ndarray:
        output = np.zeros(array.shape, np.uint8)
        mask = array[0, :, :] != 0
        for i in range(1, array.shape[0] - 1):
            mask &= array[i, :, :] != 0

        for i in range(array.shape[0]):
            masked_array = array[i][mask]
            min_pixel = np.percentile(masked_array, self.min_clip_percentile)
            max_pixel = np.percentile(masked_array, self.max_clip_percentile)
            array[i] = array[i].clip(min_pixel, max_pixel)

            array[i] -= array[i].min()
            output[i] = array[i] / (array[i].max() / 255)

        return output

    def dra2(self, array: np.ndarray):
        # Calculate the values at the specified percentiles
        min_clip_value = np.percentile(array, self.min_clip_percentile)
        max_clip_value = np.percentile(array, self.max_clip_percentile)

        # Clip the values outside the specified range
        adjusted_array = np.clip(array, min_clip_value, max_clip_value)

        # Normalize the values to [0, 1] range
        adjusted_array = (adjusted_array - min_clip_value) / (
                max_clip_value - min_clip_value)

        return adjusted_array

    def _train_epoch(self, epoch):
        self.logger.info('\n')

        self.model.train()
        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.freeze_bn()
            else:
                self.model.freeze_bn()
        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (data, target) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            # data, target = data.to(self.device), target.to(self.device)

            # LOSS & OPTIMIZE
            output = self.model(data)
            target = target.to(self.device)

            if self.config['arch']['type'][:3] == 'PSP':
                assert output[0].size()[1:] == target.size()[1:]
                assert output[0].size()[1] == self.num_classes
                loss = self.loss(output[0], target)
                loss += self.loss(output[1], target) * 0.4
                output = output[0]
            else:
                # print(output.size()[1:])
                # print(target.size()[1:])
                assert output.size()[1:] == target.size()[1:]
                assert output.size()[1] == self.num_classes
                loss = self.loss(output, target)

            if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step(epoch=epoch - 1)

            self.total_loss.update(loss.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar(f'{self.k_fold}/{self.wrt_mode}/loss', loss.item(),
                                       self.wrt_step)
                self.writer.add_scalar('loss', loss.item(), self.wrt_step)

            # FOR EVAL
            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            # PRINT INFO
            pixAcc, mIoU, _ = self._get_seg_metrics().values()
            description = (f'TRAIN EPOCH {epoch} | Batch: {batch_idx + 1} | '
                           f'Loss: {self.total_loss.average:.3f}, '
                           f'PixelAcc: {pixAcc:.2f}, '
                           f'Mean IoU: {mIoU:.2f} |')
            tbar.set_description(description)

        # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics()
        for k, v in list(seg_metrics.items())[:-1]:
            self.writer.add_scalar(f'{self.k_fold}/{self.wrt_mode}/{k}', v, self.wrt_step)
            self.writer.add_scalar(f'{k}', v, self.wrt_step)

        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'{self.k_fold}/{self.wrt_mode}/Learning_rate_{i}',
                                   opt_group['lr'], self.wrt_step)
            self.writer.add_scalar(f'Learning_rate_{i}', opt_group['lr'], self.wrt_step)

        # RETURN LOSS & METRICS
        log = {'loss': self.total_loss.average,
               **seg_metrics}
        self.logger.info(f"Finished training epoch {epoch}")

        # if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning(
                'Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target) in enumerate(tbar):

                # LOSS
                output = self.model(data)
                target = target.to(self.device)

                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append(
                        [self.dra(data[0].data.cpu()), target_np[0],
                         output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                description = (f'EVAL EPOCH {epoch} | Batch: {batch_idx + 1} | '
                               f'Loss: {self.total_loss.average:.3f}, '
                               f'PixelAcc: {pixAcc:.2f}, '
                               f'Mean IoU: {mIoU:.2f} |')
                tbar.set_description(description)

            # WRTING & VISUALIZING THE MASKS
            val_img = []
            palette = self.train_loader.dataset.palette
            for dta, tgt, out in val_visual:
                dta = dta * 2048
                # print("TOTAL: ", tgt.sum(), out.sum())
                # print("Max: ", tgt.max(), out.max())
                # print("Min: ", tgt.min(), out.min())
                print("viz shapes: ", dta.shape, tgt.shape, out.shape)
                dta = dta.transpose(1,2,0)
                tgt = tgt.transpose(1,2,0)
                dta = self.restore_transform(dta.astype(np.uint8))

                tgt, out = colorize_mask(tgt, palette), colorize_mask(out, palette)
                dta, tgt, out = dta.convert('RGB'), tgt.convert('RGB'), out.convert('RGB')
                [dta, tgt, out] = [self.viz_transform(x) for x in [dta, tgt, out]]
                val_img.extend([dta, tgt, out])
            val_img = torch.stack(val_img, 0)
            val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
            self.writer.add_image(f'{self.k_fold}/{self.wrt_mode}/inputs_targets_predictions',
                                  val_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.k_fold}/{self.wrt_mode}/loss',
                                   self.total_loss.average, self.wrt_step)
            self.writer.add_scalar('loss',
                                   self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-1]:
                self.writer.add_scalar(f'{self.k_fold}/{self.wrt_mode}/{k}', v, self.wrt_step)
                self.writer.add_scalar(f'{k}', v, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }

            # WRITE TO FILE
            self.logger.info(description)
            seg_metrics_json = json.dumps(str(seg_metrics), indent=4,
                                          sort_keys=True)
            self.logger.info(seg_metrics_json)

        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()

        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }
