import argparse
import json
import os
from sklearn.model_selection import KFold

import torch
import pandas as pd
import datetime

from dataloaders import DSTL
import models
from trainers import DSTLTrainer
from utils import Logger, losses, Array3dMergeConfig, BandGroup
import copy
from base import initiate_stats
from pathlib import Path
import numpy as np
import logging

torch.cuda.empty_cache()


def get_loader_instance(name, _wkt_data, config, train_indxs=None,
                        val_indxs=None):
    training_band_groups = []
    for group in config["train_loader"]['preprocessing'][
        'training_band_groups']:
        cfg = Array3dMergeConfig(group['merge_3d']["strategy"],
                                 group['merge_3d']["kernel"],
                                 group['merge_3d']["stride"]) if (
                    "merge_3d" in group) else None
        training_band_groups.append(BandGroup(group['bands'], cfg))

    # Preprocessing config
    preproccessing_config = copy.deepcopy(
        config["train_loader"]['preprocessing'])
    del preproccessing_config['training_band_groups']

    # Loader args
    loader_args = copy.deepcopy(config[name]['args'])
    batch_size_ = loader_args['batch_size']
    del loader_args['batch_size']

    return DSTL(_wkt_data, training_band_groups,
                batch_size_,
                **loader_args,
                **preproccessing_config,
                train_indxs=train_indxs,
                val_indxs=val_indxs,
                val=config["trainer"]["val"],
                )


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def stratified_split(sorted_array, group_size):
    """
    Stratified split of a sorted array into groups of specified size.

    Parameters:
    sorted_array (list): The sorted array to be split.
    group_size (int): The size of each group.

    Returns:
    list: A list of lists representing the stratified split.
    """
    num_elements = len(sorted_array)
    num_groups = num_elements // group_size
    remainder = num_elements % group_size

    groups = []
    start_idx = 0

    for i in range(num_groups):
        group_end = start_idx + group_size
        groups.append(sorted_array[start_idx:group_end])
        start_idx = group_end

    # If there's a remainder, add it as the last group
    if remainder > 0:
        groups.append(sorted_array[start_idx:])

    return np.array(groups).transpose((1, 0))


def write_metric(writer, stats, metric, func, class_name, metric_name):
    for m in stats[metric]:
        for index in range(len(m['train'])):
            m_t = m['train'][:, index]
            metric_t = func(m_t)

            m_v = m['val'][:, index]
            metric_v = func(m_v)

            writer.add_scalar(f'{class_name}/{metric_name}', {
                'train': metric_t,
                'val': metric_v
            }, index + 1)

def write_metric_2_param(writer, stats, metric_1, metric_2, func, class_name,
                      metric_name):
    for m1, m2 in zip(stats[metric_1], stats[metric_2]):
        for index in range(len(m1['train'])):
            m1_t = m1['train'][:, index]
            m2_t = m2['train'][:, index]
            metric_t = func(m1_t, m2_t)

            m1_v = m1['val'][:, index]
            m2_v = m2['val'][:, index]
            metric_v = func(m1_v, m2_v)

            writer.add_scalar(f'{class_name}/{metric_name}', {
                'train': metric_t,
                'val': metric_v
            }, index + 1)

def write_metric_3_param(writer, stats, metric_1, metric_2, metric_3, func,
                         class_name,
                        metric_name):
    for m1, m2, m3 in zip(stats[metric_1], stats[metric_2], stats[metric_3]):
        for index in range(len(m1['train'])):
            m1_t = m1['train'][:, index]
            m2_t = m2['train'][:, index]
            m3_t = m3['train'][:, index]
            metric_t = func(m1_t, m2_t, m3_t)

            m1_v = m1['val'][:, index]
            m2_v = m2['val'][:, index]
            m3_v = m3['val'][:, index]
            metric_v = func(m1_v, m2_v, m3_v)

            writer.add_scalar(f'{class_name}/{metric_name}', {
                'train': metric_t,
                'val': metric_v
            }, index + 1)

def write_stats_to_tensorboard(writer, class_stats):

    # LOSS
    write_metric(writer, stats, 'loss', np.mean, 'all', 'Loss')

    for class_name, stats in class_stats.item():

        # mAP
        write_metric(writer, stats, 'average_precision', np.mean, class_name, 'mAP')

        # PIXEL ACCURACY
        write_metric_2_param(writer, stats, 'total_correct',
                             'total_label',
                                pixel_accuracy, class_name, 'Pixel_Accuracy')

        # PRECISION
        write_metric_2_param(writer, stats, 'intersection', 'predicted_positives',
                                precision, class_name, 'Precision')

        # RECALL
        write_metric_2_param(writer, stats, 'intersection', 'total_positives',
                                recall, class_name, 'Recall')

        # F1 SCORE
        write_metric_3_param(writer, stats, 'intersection', 'predicted_positives',
                                'total_positives', f1_score, class_name,
                                'F1_Score')

        # MEAN IoU
        write_metric_2_param(writer, stats, 'intersection', 'union',
                                intersection_over_union, class_name, 'Mean_IoU')


def _append_stats(all_stats, stats):
    for key in all_stats.keys():
        all_stats[key]['train'] = np.append(all_stats[key]['train'], stats[key]['train'])
        all_stats[key]['val'] = np.append(all_stats[key]['val'], stats[key]['val'])

    return all_stats

def main(config, resume):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # This was made an environment variable and not in config because when
    # testing and running multiple config files on one machine is frustration
    # to update the config file each time you want to run it on a different
    # machine, like the gpu cluster that has a different file system or the
    # data exists elsewhere from the development environment.
    dstl_data_path = os.environ.get('DSTL_DATA_PATH')
    if dstl_data_path is None:
        raise EnvironmentError(
            'DSTL_DATA_PATH environment variable is not set, '
            'it must be a path to your DSTL data directory.')

    # Load the CSV into a DataFrame
    df = pd.read_csv(
        os.path.join(dstl_data_path, 'train_wkt_v4.csv/train_wkt_v4.csv'))

    # Get the data metadata list.
    _wkt_data = {}
    for index, row in df.iterrows():
        im_id = row['ImageId']
        class_type = row['ClassType']
        poly = row['MultipolygonWKT']

        # Add the polygon to the dictionary
        _wkt_data.setdefault(im_id, {})[int(class_type)] = poly

    _wkt_data = list(_wkt_data.items())
    training_classes_ = config['train_loader']['preprocessing']['training_classes']

    # Stratified K-Fold
    mask_stats = json.loads(Path(
        'dataloaders/labels/dstl-stats.json').read_text())
    image_ids = df['ImageId'].unique()
    im_area = [(idx, np.mean([mask_stats[im_id][str(cls)]['area'] for cls
                                in training_classes_]))
               for idx, im_id in enumerate(image_ids)]

    sorted_by_area = sorted(im_area, key=lambda x: str(x[1]), reverse=True)
    sorted_by_area = [t[0] for t in sorted_by_area]
    arr = stratified_split(sorted_by_area, 5)
    stratisfied_indices = arr.flatten()

    # LOSS
    loss = getattr(losses, config['loss'])(threshold=config['threshold'])
    start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')

    if config["trainer"]["val"]:
        # Split the data into K folds
        shuffle_ = config["trainer"]["k_shuffle"]
        random_state_ = config["trainer"]["k_random_state"] if shuffle_ else None
        kfold = KFold(n_splits=config["trainer"]["k_split"],
                      shuffle=shuffle_, random_state=random_state_)

        area_by_id = dict(im_area)

        writer = None

        # Initialise the stats
        fold_stats = np.array([])

        # Iterate over the K folds
        for fold, (train_indxs_of_indxs, val_indxs_of_indxs) in enumerate(kfold.split(stratisfied_indices)):
            logger.info(f'Starting Fold {fold + 1}:')
            train_indxs = stratisfied_indices[train_indxs_of_indxs]
            val_indxs = stratisfied_indices[val_indxs_of_indxs]

            # Logging
            logger.info(f"Train: {' '.join([str(k) for k in train_indxs])}")
            logger.info(f"Valid: {' '.join([str(k) for k in val_indxs])}")
            logger.info(
                f'Train area mean: {np.mean([area_by_id[im_id] for im_id in train_indxs]):.6f}')
            logger.info(
                f'Valid area mean: {np.mean([area_by_id[im_id] for im_id in val_indxs]):.6f}')
            train_area_by_class, valid_area_by_class = [
                {cls: np.mean(
                    [mask_stats[image_ids[im_id]][str(cls)]['area'] for im_id 
                     in im_ids])
                    for cls in training_classes_}
                for im_ids in [train_indxs, val_indxs]]

            logger.info(f"Train area by class: "
                        f"{' '.join(f'{cls}: {train_area_by_class[cls]:.6f}' for cls in training_classes_)}")
            logger.info(f"Valid area by class: "
                        f"{' '.join(f'cls-{cls}: {valid_area_by_class[cls]:.6f}' for cls in training_classes_)}")

            # DATA LOADERS
            train_loader = get_loader_instance(
                'train_loader', _wkt_data, config, train_indxs, val_indxs)

            # MODEL
            model = get_instance(models, 'arch', config, len(training_classes_))

            if train_loader.get_val_loader() is None:
                raise ValueError("Val Loader is None")

            # TRAINING
            trainer = DSTLTrainer(
                start_time=start_time,
                k_fold=fold,
                model=model,
                loss=loss,
                resume=resume,
                config=config,
                train_loader=train_loader,
                val_loader=train_loader.get_val_loader(),
                train_logger=logger,
                root=dstl_data_path,
            )

            wrter, stats = trainer.train()
            # im lazy and dont want to refactor the code
            writer = wrter

            if fold_stats is None:
                # Classes, Metric, Type
                fold_stats = stats
            else:
                for class_name, class_stats in fold_stats.items():
                    for metric_name, metric_stats in class_stats.items():
                        for type, stat in metric_stats.items():
                            fold_stats[class_name][metric_name][type] = (
                                np.append((fold_stats[class_name][
                                               metric_name][type], stat)))

            logger.info(f'Finished Fold {fold + 1}:')
            if config["trainer"]["k_stop"] is not None and config["trainer"][ \
                    "k_stop"] == fold + 1:
                break

        # Write the stats to tensorboard
        write_stats_to_tensorboard(writer, fold_stats)

        print('train_cIoU', train_ti)

    else:
        # DATA LOADERS
        train_loader = get_loader_instance('train_loader', _wkt_data, config)

        # MODELMODEL
        model = get_instance(models, 'arch', config, len(training_classes_))

        # TRAINING
        trainer = DSTLTrainer(
            start_time=start_time,
            model=model,
            loss=loss,
            resume=resume,
            config=config,
            train_loader=train_loader,
            train_logger=logger,
            root=dstl_data_path,
        )

        trainer.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    config = json.load(open(args.config))
    # args.resume = 'saved/ERFNet_voc_512_batch8_split_1_2/06-21_21-18/best_model.pth'
    # if args.resume:
    # config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
    # main(config, args.resume)
    # main(config, args.resume)
    # main(config, args.resume)
    # main(config, args.resume)
