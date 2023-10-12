import argparse
import json
import os

import torch

from dataloaders import DSTL
import models
from trainers import DSTLTrainer
from utils import Logger, losses, Array3dMergeConfig, BandGroup
import copy

torch.cuda.empty_cache()


def get_loader_instance(name, config, *args):
    # TODO implement all params in config
    training_band_groups = []
    for group in config["train_loader"]['preprocessing']['training_band_groups']:
        cfg = Array3dMergeConfig(group['merge_3d']["strategy"],
                                 group['merge_3d']["kernel"],
                                 group['merge_3d']["stride"]) if ("merge_3d" in group) else None
        training_band_groups.append(BandGroup(group['bands'], cfg))

    preproccessing_config = copy.deepcopy(config["train_loader"]['preprocessing'])
    del preproccessing_config['training_band_groups']

    # GET THE CORRESPONDING CLASS / FCT
    return (DSTL(training_band_groups, *args, **preproccessing_config,
                 **config[name]['args']))

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

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



    # MODELMODEL
    model = get_instance(models, 'arch', config,
                                train_loader.dataset.num_classes)
    # print(f'\n{model}\n')

    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index=config['ignore_index'])

    if config["trainer"]["val"]:
        # Split the data into K folds
        kfold = KFold(n_splits=K, shuffle=True,
                      random_state=42)  # K is the number of folds

        # Iterate over the K folds
        for fold, (train_indices, val_indices) in enumerate(kfold.split(data)):
            train_logger.add_entry(f'Starting Fold {fold + 1}:')

            # DATA LOADERS
            train_loader = get_loader_instance('train_loader', config, train_indices, val_indices)

            # TRAINING
            trainer = DSTLTrainer(
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
        train_loader = get_loader_instance('train_loader', config)

        # TRAINING
        trainer = DSTLTrainer(
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
