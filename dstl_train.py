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

torch.cuda.empty_cache()


def get_loader_instance(name, _wkt_data, config, train_indxs=None,
                        val_indxs=None):
    training_band_groups = []
    for group in config["train_loader"]['preprocessing']['training_band_groups']:
        cfg = Array3dMergeConfig(group['merge_3d']["strategy"],
                                 group['merge_3d']["kernel"],
                                 group['merge_3d']["stride"]) if ("merge_3d" in group) else None
        training_band_groups.append(BandGroup(group['bands'], cfg))

    # Preprocessing config
    preproccessing_config = copy.deepcopy(config["train_loader"]['preprocessing'])
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

def split_array(array, num_parts):
    parts = []
    for start, end in zip(range(0, num_parts), range(num_parts, num_parts*2)):
        if end < len(array) - 1:
             end_index = end
        else:
            end_index = end - len(array)
        parts.append([array[start], array[end_index]])
    return parts

def main(config, resume):
    train_logger = Logger()

    # This was made an environment variable and not in config because when
    # testing and running multiple config files on one machine is frustration
    # to update the config file each time you want to run it on a different
    # machine, like the gpu cluster that has a different file system or the
    # data exists elsewhere from the development environment.
    dstl_data_path = os.environ.get('DSTL_DATA_PATH')
    if dstl_data_path is None:
        raise EnvironmentError('DSTL_DATA_PATH environment variable is not set, '
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

    # Stratified K-Fold
    mask_stats = json.loads(Path(
        'dataloaders/labels/dstl-stats.json').read_text())
    im_area = [(im_id, np.mean([mask_stats[im_id][str(cls)]['area']
                                for cls in hps.classes]))
               for im_id in all_im_ids]

    sorted_by_area = sorted(im_area, key=lambda x: (x[1], x[0]), reverse=True)
    arr = split_array(sorted_by_area, 5)
    stratisfied_indices_by_class_area =np.array(arr).flatten()

    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index=config['ignore_index'])
    start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')

    if config["trainer"]["val"]:
        # Split the data into K folds
        shuffle_ = config["trainer"]["k_shuffle"]
        random_state_ = config["trainer"]["k_random_state"] if shuffle_ else None
        kfold = KFold(n_splits=config["trainer"]["k_split"],
                      shuffle=shuffle_, random_state=random_state_)

        area_by_id = dict(im_area)

        # Iterate over the K folds
        for fold, (train_indxs, val_indxs) in enumerate(kfold.split(_wkt_data)):
            train_logger.add_entry(f'Starting Fold {fold + 1}:')

            # DATA LOADERS
            train_loader = get_loader_instance(
                'train_loader', _wkt_data, config, train_indxs, val_indxs)

            # MODEL
            model = get_instance(models, 'arch', config,
                                 train_loader.dataset.num_classes)

            if train_loader.get_val_loader() is None:
                raise ValueError("Val Loader is None")

            logger.info('Train: {}'.format(' '.join(sorted(train_ids))))
            logger.info('Valid: {}'.format(' '.join(sorted(valid_ids))))
            logger.info('Train area mean: {:.6f}'.format(
                np.mean([area_by_id[im_id] for im_id in valid_ids])))
            logger.info('Train area by class: {}'.format(
                ' '.join('{}: {:.6f}'.format(cls, train_area_by_class[cls])
                         for cls in hps.classes)))
            logger.info('Valid area mean: {:.6f}'.format(
                np.mean([area_by_id[im_id] for im_id in train_ids])))
            logger.info('Valid area by class: {}'.format(
                ' '.join('cls-{}: {:.6f}'.format(cls, valid_area_by_class[cls])
                         for cls in hps.classes)))

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
                train_logger=train_logger,
                root=dstl_data_path,
            )

            trainer.train()

            train_logger.add_entry(f'Finished Fold {fold + 1}:')
            if config["trainer"]["k_stop"] is not None and  config["trainer"][\
                    "k_stop"] == fold + 1:
                break

    else:
        # DATA LOADERS
        train_loader = get_loader_instance('train_loader', _wkt_data, config)

        # MODELMODEL
        model = get_instance(models, 'arch', config,
                             train_loader.dataset.num_classes)

        # TRAINING
        trainer = DSTLTrainer(
            start_time=start_time,
            model=model,
            loss=loss,
            resume=resume,
            config=config,
            train_loader=train_loader,
            train_logger=train_logger,
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
