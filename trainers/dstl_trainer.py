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
from utils.metrics import (eval_metrics, recall, precision, f1_score,
                           pixel_accuracy, AverageMeter,
                           mean_average_precision, intersection_over_union)
import logging

# def precision(output, target):
#     true_positive = ((output == 0) * (target == 0)).sum().float()

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

        torch.backends.cudnn.benchmark = True
        self.threshold = config["threshold"]

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

        loss_history = np.array([])
        total_metric_totals = dict()
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (data, target) in enumerate(tbar):
            self.data_time.update(time.time() - tic)

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
                assert output.size()[1:] == target.size()[1:]
                assert output.size()[1] == self.num_classes
                loss = self.loss(output, target)

            if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                loss_history = np.append(loss_history, loss.item())

            # FOR EVAL
            metrics_totals = eval_metrics(output, target, self.threshold)
            if 'all' not in total_metric_totals:
                total_metric_totals['all'] = metrics_totals
            else:
                for k, v in metrics_totals.items():
                    total_metric_totals['all'][k] += v

            for class_idx in range(self.num_classes):
                class_metrics_totals = eval_metrics(
                    output[:, class_idx, :, :][:, np.newaxis, :, :],
                    target[:, class_idx, :, :][:, np.newaxis, :, :],
                                                    self.threshold)
                if str(class_idx) not in total_metric_totals:
                    total_metric_totals[str(class_idx)] = class_metrics_totals
                else:
                    for k, v in class_metrics_totals.items():
                        total_metric_totals[str(class_idx)][k] += v

            # PRINT INFO
            seg_metrics = self._get_seg_metrics(metrics_totals)
            description = f'TRAIN EPOCH {epoch} | Batch: {batch_idx + 1} | '
            for k, v in seg_metrics.items():
                description += f'{self.convert_to_title_case(k)}: {v:.3f} | '
            tbar.set_description(description)

        self.logger.info(f"Finished training epoch {epoch}")

        # Add loss
        total_metric_totals['all']['loss'] = loss_history

        return total_metric_totals

    def convert_to_title_case(self, input_string):
        words = input_string.split('_')
        capitalized_words = [word.capitalize() for word in words]
        return ' '.join(capitalized_words)

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning(
                'Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        loss_history = np.array([])
        total_metric_totals = dict()

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

                if batch_idx % self.log_step == 0:
                    loss_history = np.append(loss_history, loss.item())

                # METRICS
                metrics_totals = eval_metrics(output, target, self.threshold)
                if 'all' not in total_metric_totals:
                    total_metric_totals['all'] = metrics_totals
                else:
                    for k, v in metrics_totals.items():
                        total_metric_totals['all'][k] += v

                for class_idx in range(self.num_classes):
                    class_metrics_totals = eval_metrics(
                        output[:, class_idx, :, :][:, np.newaxis, :, :],
                        target[:, class_idx, :, :][:, np.newaxis, :, :],
                                                    self.threshold)
                    if str(class_idx) not in total_metric_totals:
                        total_metric_totals[str(class_idx)] = class_metrics_totals
                    else:
                        for k, v in class_metrics_totals.items():
                            total_metric_totals[str(class_idx)][k] += v

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append(
                        [self.dra(data[0].data.cpu()), target_np[0],
                         output_np[0]])

                # PRINT INFO
                seg_metrics = self._get_seg_metrics(metrics_totals)
                description = f'EVAL EPOCH {epoch} | Batch: {batch_idx + 1} | '
                for k, v in seg_metrics.items():
                    description += f'{self.convert_to_title_case(k)}: {v:.3f} | '

                description += f"| B {self.batch_time.average:.2f} D {self.data_time.average:.2f} |"
                tbar.set_description(description)


            # WRTING & VISUALIZING THE MASKS
            val_img = []
            palette = self.train_loader.dataset.palette
            for dta, tgt, out in val_visual:
                # TODO scale the last 8 bands
                dta = dta * 2048

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
            self.writer.add_image(f'inputs_targets_predictions', val_img, epoch)

        # Add loss
        total_metric_totals['all']['loss'] = loss_history

        return total_metric_totals

    def _get_seg_metrics(self, seg_totals):
        pixAcc = pixel_accuracy(seg_totals['correct_pixels'], seg_totals['total_labeled_pixels'])
        p = precision(seg_totals['intersection'], seg_totals[
            'predicted_positives'])
        r = recall(seg_totals['intersection'], seg_totals[
            'total_positives'])
        f1 = f1_score(seg_totals['intersection'], seg_totals[
            'predicted_positives'], seg_totals['total_positives'])
        mAP = mean_average_precision(seg_totals['average_precision'])
        mIoU = intersection_over_union(seg_totals['intersection'], seg_totals[
            'union'])

        return {
            "Mean_IoU": np.round(mIoU, 3),
            "mAP": np.round(mAP, 3),
            "F1": np.round(f1, 3),
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Precision": np.round(p, 3),
            "Recall": np.round(r, 3),
        }
