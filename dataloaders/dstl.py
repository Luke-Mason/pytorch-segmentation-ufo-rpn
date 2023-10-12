# Written by Luke Mason
# Date: 2020-06-03
# Purpose: To create a dataloader for the DSTL dataset.


import csv
import datetime
import json
import logging
import os
from itertools import islice
from pathlib import Path
from typing import Dict, Tuple, List
import sys
import cv2
import math
import numpy as np
import pandas as pd
import rasterio
import shapely.affinity
import shapely.geometry
import shapely.wkt
import torch
from base import BaseDataSet, BaseDataLoader
from shapely.geometry import MultiPolygon
from utils import (array_3d_merge, Array3dMergeConfig, BandGroup,
                   mask_for_polygons,
                   palette, generate_unique_config_hash)



class DSTLDataset(BaseDataSet):

    def __init__(self,
                 data,
                 training_band_groups: Tuple[BandGroup, BandGroup, BandGroup],
                 training_classes: List[int],
                 img_ref_scale: str,
                 patch_size: int,
                 overlap_percentage: float,
                 align_images: bool,
                 interpolation_method: int,
                 **kwargs):
        """Constructor, initialiser.

        Args:
            training_classes (List[int]): The class labels to train on.
            training_band_groups (Tuple[BandGroup, BandGroup, BandGroup]): The colour spectrum
            bands to train on.
            align_images (bool): Align the images.
            interpolation_method (int): The interpolation method to use.
        """
        if img_ref_scale not in ['RGB', 'P', 'M', 'A']:
            raise ValueError(f"Unknown image reference scale: {img_ref_scale}")

        if len(training_band_groups) != 3:
            raise ValueError("Band groups must be 3")

        for group in training_band_groups:
            if group.merge_3d is None and len(group.bands) != 1:
                raise ValueError("Number of bands in a group must be 1 if "
                                 "there is no merge strategy for the group")

        self.num_classes = 10 if len(training_classes) == 0 else len(training_classes) + 1

        # Attributes
        self.img_ref_scale = img_ref_scale
        self.training_classes = training_classes
        self.training_band_groups = training_band_groups
        self.patch_size = patch_size
        self.overlap_percentage = overlap_percentage
        self.align_images = align_images
        self.interpolation_method = interpolation_method

        # Colours TODO
        self.palette = palette.get_voc_palette(self.num_classes)

        # Setup directories and paths
        self.root = kwargs['root']
        self.image_dir = os.path.join(self.root, 'sixteen_band/sixteen_band')
        self.cache_dir = os.path.join(self.root, 'cached')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        stage_1_dir = os.path.join(self.cache_dir, 'stage_1')
        if not os.path.exists(stage_1_dir):
            os.makedirs(stage_1_dir)

        stage_2_dir = os.path.join(self.cache_dir, 'stage_2')
        if not os.path.exists(stage_2_dir):
            os.makedirs(stage_2_dir)

        masks_dir = os.path.join(self.cache_dir, 'masks')
        if not os.path.exists(masks_dir):
            os.makedirs(masks_dir)

        # Memory Cache
        self.images = dict()  # type Dict[str, np.ndarray]
        self.labels = dict()  # type Dict[str, np.ndarray]
        self._wkt_data = data  # type Dict[str, Dict[int, str]]
        self._x_max_y_min = None  # type Dict[str, Tuple[int, int]]

        # Logging
        self._setup_logging()

        # The colour palette for the classes
        self.palette = palette.get_voc_palette(len(self.training_classes))

        super(DSTLDataset, self).__init__(dataset=, **kwargs)

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Logs to file

        # Check if log directory exist, if not create it
        log_dir = os.path.join(self.root, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file_name = datetime.datetime.now().strftime('%Y-%m-%d_%H.log')
        handler = logging.FileHandler(os.path.join(log_dir, log_file_name),
                                      mode='a')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def _set_files(self):
        # TODO use seed to split, and depending on if self.val is true or not
        #  get the left of slice or right of slice
        ids = list(self._wkt_data.keys())
        step_size = math.ceil(self.patch_size - ((self.overlap_percentage / 100.0) *
                                       self.patch_size))
        self.files = []  # type: List[Tuple[np.ndarray, np.ndarray, str]]

        # Preprocess Images, ensure their existences.
        for image_id in ids:
            stage_1_file_path = self.get_stage_1_file_path(image_id)
            if not os.path.exists(stage_1_file_path + ".data.npy"):
                self._preprocess_image_stage_1(image_id, stage_1_file_path)

            # Save mask for image
            mask_path = self.get_mask_file_path(image_id)
            if not os.path.exists(mask_path + ".mask.npy"):
                self._save_mask(image_id, stage_1_file_path, mask_path)

            stage_2_file_path = self.get_stage_2_file_path(image_id)
            if not os.path.exists(stage_2_file_path + ".data.npy"):
                self._preprocess_image_stage_2(image_id, stage_1_file_path,
                                               stage_2_file_path)

        # Load Images
        for index, image_id in enumerate(ids):
            image = np.load(Path(self.get_stage_2_file_path(image_id) +
                                 ".data.npy"),  allow_pickle=True)
            self.logger.debug(f'Image shape: {image.shape}')

            height, width = image.shape[1], image.shape[2]
            chunk_offsets = self._gen_chunk_offsets(width, height, step_size)
            mask = np.load(Path(self.get_mask_file_path(image_id) + ".mask.npy"),  allow_pickle=True)
            self.logger.debug(f'Mask shape: {mask.shape}')

            for c_index, (x, y) in enumerate(chunk_offsets):

                # Get the masks for the classes that we want to validate and train on.
                patch_y_mask = mask[:, y:y + self.patch_size,
                               x:x + self.patch_size]

                # Merging bands together with the strategies to produce
                # the input patch image.
                patch = image[:, y:y + self.patch_size, x:x + self.patch_size]

                self.files.append((patch, patch_y_mask, image_id))
                self.print_progress_bar(c_index + 1, len(chunk_offsets),
                                      prefix=f"Chunking Image {image_id}...",
                                      suffix=f"{((100 / len(chunk_offsets)) * (c_index + 1)):.2f}%")

            self.logger.info(f"Total Data Loaded {(100 / len(ids)) * (index + 1)}% ...")

    def print_progress_bar(self, iteration, total, prefix='', suffix='',
                           length=50,
                           fill='█'):
        percent = ("{:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)

        sys.stdout.write(
            '\r{} |{}| {}% {}'.format(prefix, bar, percent, suffix))
        sys.stdout.flush()

    def get_mask_file_path(self, image_id: str) -> str:
        transformative_config = {
            "img_ref_scale": self.img_ref_scale,
            "classes": self.training_classes,
        }
        hash_id = generate_unique_config_hash(transformative_config)
        filename = f"{image_id}_{hash_id}"
        return os.path.join(self.cache_dir, "masks", image_id)

    def get_stage_1_file_path(self, image_id):
        transformative_config = {
            "img_ref_scale": self.img_ref_scale,
            "align_images": self.align_images,
            "interpolation_method": self.interpolation_method,
        }
        hash_id = generate_unique_config_hash(transformative_config)
        filename = f"{image_id}_{hash_id}"
        return os.path.join(self.cache_dir, "stage_1", filename)

    def get_stage_2_file_path(self, image_id):
        transformative_config = {
            "img_ref_scale": self.img_ref_scale,
            "align_images": self.align_images,
            "interpolation_method": self.interpolation_method,
            "training_band_groups": [group.to_dict()
                                     for group in self.training_band_groups]
        }
        hash_id = generate_unique_config_hash(transformative_config)
        filename = f"{image_id}_{hash_id}"
        return os.path.join(self.cache_dir, "stage_2", filename)

    def _save_mask(self, image_id, file_path, mask_path):
        image = np.load(Path(file_path + ".data.npy"),  allow_pickle=True)
        im_size = image.shape[1:]
        class_to_polygons = self._load_polygons(image_id, im_size)
        mask = np.array(
            [mask_for_polygons(im_size, class_to_polygons.get(str(cls + 1)))
             for cls in self.training_classes])

        # Add the mask for all where there is no mask
        mask = np.concatenate((mask, np.expand_dims(
            np.logical_not(np.any(mask, axis=0)), axis=0)), axis=0)

        np.save(Path(mask_path + ".mask.npy"), mask, allow_pickle=True)

    def _load_data(self, index: int):
        return self.files[index]

    def __getitem__(self, index):
        patch, patch_y_mask, image_id = self._load_data(index)
        if self.val and self.augment:
            patch, patch_y_mask = self._val_augmentation(patch, patch_y_mask)
        elif self.augment:
            patch, patch_y_mask = self._augmentation(patch, patch_y_mask)

        patch_y_mask = torch.from_numpy(patch_y_mask.astype(np.bool_)).long()
        # patch = Image.fromarray(np.uint8(patch))

        if self.return_id:
            return patch, patch_y_mask, image_id
        return patch, patch_y_mask

    def _gen_chunk_offsets(self, width: int, height: int, step_size: int) -> \
            List[Tuple[int, int]]:
        """
         Returns a list of (x, y) offsets corresponding to chunks of the image
         To account for the left over pixels it will generate a chunk a step back
         from the edge leftovers.
         :param height: The height of the image.
         :param step_size: The step size to use when generating the chunks.
         :return: A list of (x, y) offsets.
         """

        x_offsets = list(range(0, width - step_size + 1, step_size))
        y_offsets = list(range(0, height - step_size + 1, step_size))
        if width % step_size != 0:
            x_offsets.append(width - step_size)
        if height % step_size != 0 or height:
            y_offsets.append(height - step_size)

        # Erase the end offsets that are greater than the image size minus
        # the patch size.
        upper_limit = width - self.patch_size
        x_offsets = [x for x in x_offsets if x <= upper_limit]
        upper_limit = height - self.patch_size
        y_offsets = [y for y in y_offsets if y <= upper_limit]

        chunk_offsets = [(x, y) for x in x_offsets for y in y_offsets]

        return chunk_offsets

    def _load_polygons(self, image_id: str, im_size: Tuple[int, int]) \
            -> Dict[str, MultiPolygon]:
        """
        Load the polygons for the image id and scale them to the image size.
        :param im_id: The image id.
        :param im_size: The image size.
        :return: A dictionary of class type to polygon.
        """
        height, width = im_size
        self.logger.debug(f'Loading polygons for image: {image_id}')
        x_max, y_min = self._get_x_max_y_min(image_id)
        x_scaler, y_scaler = self._get_scalers(height, width, x_max, y_min)

        items_ = {
            str(poly_type): shapely.affinity.scale(shapely.wkt.loads(poly),
                                                   xfact=x_scaler,
                                                   yfact=y_scaler,
                                                   origin=(0, 0, 0)) for
            poly_type, poly in self._wkt_data[image_id].items()}
        self.logger.debug(f'Loaded polygons for image: {image_id}')
        return items_

    def _get_scalers(self, h, w, x_max, y_min) -> Tuple[float,
    float]:
        """
        Get the scalers for the x and y axis as according to the DSTL
        competition documentation there is some scaling and preprocessing
        that needs to occur to correc the training data..... so annoying
        """
        w_ = w * (w / (w + 1))
        h_ = h * (h / (h + 1))
        x_scaler = w_ / x_max
        y_scaler = h_ / y_min
        return x_scaler, y_scaler

    def _get_x_max_y_min(self, im_id: str) -> Tuple[float, float]:
        """
        Get the max x and y values from grid sizes file that is provided from the dataset in competition.
        According to the DSTL competition documentation there is some scaling and preprocessing
        that needs to occur to correct the training data..... so annoying
        :param im_id:
        :return:
        """
        if self._x_max_y_min is None:
            with open(os.path.join(self.root,
                                   'grid_sizes.csv/grid_sizes.csv')) as f:
                self._x_max_y_min = {im_id: (float(x), float(y))
                                     for im_id, x, y in
                                     islice(csv.reader(f), 1, None)}
        return self._x_max_y_min[im_id]

    def _scale_percentile(self, matrix: np.ndarray) -> np.ndarray:
        """ Fixes the pixel value range to 2%-98% original distribution of values.
        """
        w, h, d = matrix.shape
        matrix = matrix.reshape([w * h, d])
        # Get 2nd and 98th percentile
        mins = np.percentile(matrix, 1, axis=0)
        maxs = np.percentile(matrix, 99, axis=0) - mins
        matrix = (matrix - mins[None, :]) / maxs[None, :]
        return matrix.reshape([w, h, d]).clip(0, 1)

    def _preprocess_for_alignment(self, image: np.ndarray) -> np.ndarray:
        # attempts to remove single-dimensional entries
        image = np.squeeze(image)
        # checks if the shape of the image is 2D, indicating a grayscale image (single channel).
        if len(image.shape) == 2:
            # If the image is grayscale, it expands the dimensions to make it a 3D array (assuming a single channel).
            # This is done so that the image can be concatenated with the other bands.
            image = self._scale_percentile(np.expand_dims(image, 2))
        else:  # If the image is not grayscale (assumed to have 3 channels, indicating color), it converts it to grayscale.
            assert image.shape[2] == 3, image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.astype(np.float32)

    def _preprocess_image_stage_1(self, image_id: str, file_path: str):

        key = lambda x: f'{image_id}_{x}'

        # Get paths
        rgb_path = os.path.join(self.root, 'three_band/', f'{image_id}.tif')
        p_path = os.path.join(self.image_dir, f"{key('P')}.tif")
        m_path = os.path.join(self.image_dir, f"{key('M')}.tif")
        a_path = os.path.join(self.image_dir, f"{key('A')}.tif")

        # Open streams
        im_rgb_src = rasterio.open(rgb_path, driver='GTiff', dtype=np.float32)
        im_p_src = rasterio.open(p_path, driver='GTiff', dtype=np.float32)
        im_m_src = rasterio.open(m_path, driver='GTiff', dtype=np.float32)
        im_a_src = rasterio.open(a_path, driver='GTiff', dtype=np.float32)

        # Load data from streams
        im_rgb = im_rgb_src.read().transpose([1, 2, 0])
        im_p = im_p_src.read().transpose([1, 2, 0])
        im_m = im_m_src.read().transpose([1, 2, 0])
        im_a = im_a_src.read().transpose([1, 2, 0])

        w, h = im_rgb.shape[:2]

        # TODO This has not been tested yet
        if self.align_images:
            im_p, _ = self._aligned(im_rgb, im_p, key=key('P'))
            im_m, aligned = self._aligned(im_rgb, im_m, im_m[:, :, :3],
                                          key=key('M'))
            im_ref = im_m[:, :, -1] if aligned else im_rgb[:, :, 0]
            im_a, _ = self._aligned(im_ref, im_a, im_a[:, :, 0], key=key('A'))

        # Get the reference image for scaling the other image bands to it.
        # Allows for experimenting with different reference band scales.
        if self.img_ref_scale == 'RGB':
            ref_img = im_rgb
        elif self.img_ref_scale == 'P':
            ref_img = im_p
        elif self.img_ref_scale == 'M':
            ref_img = im_m
        elif self.img_ref_scale == 'A':
            ref_img = im_a
        else:
            raise ValueError(f'Invalid reference file type: {self.reference_file_type}')

        # Resize the images to be the same size as RGB.
        # Sometimes panchromatic is a couple of pixels different to RGB
        if im_p.shape != ref_img.shape[:2]:
            im_p = cv2.resize(im_p, (h, w),
                              interpolation=self.interpolation_method)
        im_p = np.expand_dims(im_p, 2)
        im_m = cv2.resize(im_m, (h, w), interpolation=self.interpolation_method)
        im_a = cv2.resize(im_a, (h, w), interpolation=self.interpolation_method)

        # Scale images between 0-1 based off their maximum and minimum bounds
        # P and M images are 11bit integers, A is 14bit integers, scale values to be
        # between 0-1 floats
        im_p = (im_p / 2047.0) * 255.0
        im_m = (im_m / 2047.0) * 255.0
        im_a = (im_a / 16383.0) * 255.0
        image = np.concatenate([im_p, im_m, im_a], axis=2).transpose([2, 0, 1])
        self.logger.debug(f"Image shape: {image.shape}")

        # Save images
        np.save(Path(file_path + ".data"), image)
        self.logger.debug(f"Saving image to {file_path}.data.npy")

        # Close streams
        im_rgb_src.close()
        im_p_src.close()
        im_m_src.close()
        im_a_src.close()

        del im_rgb, im_p, im_m, im_a, im_rgb_src, im_p_src, im_m_src, im_a_src, \
            rgb_path, p_path, m_path, a_path

        return image

    def _preprocess_image_stage_2(self, image_id: str, src_path: str, dst_path: str):
        image = np.load(src_path + ".data.npy",  allow_pickle=True)
        image = np.array([
            image[group.bands - 1, :, :].squeeze() if group.merge_3d is None else
            array_3d_merge(image[group.bands - 1, :, :].squeeze(), group.merge_3d)
            for group in self.training_band_groups
        ], dtype=np.float32)

        # Save images
        self.logger.debug(f"Saving image to {dst_path}.data.npy")
        np.save(Path(dst_path + ".data"), image, allow_pickle=True)

    def _aligned(self, im_ref, im, im_to_align=None, key=None):
        w, h = im.shape[:2]
        im_ref = cv2.resize(im_ref, (h, w),
                            interpolation=self.interpolation_method)
        im_ref = self._preprocess_for_alignment(im_ref)
        if im_to_align is None:
            im_to_align = im
        im_to_align = self._preprocess_for_alignment(im_to_align)
        assert im_ref.shape[:2] == im_to_align.shape[:2]
        try:
            cc, warp_matrix = self._get_alignment(im_ref, im_to_align, key)
        except cv2.error as e:
            self.logger.info(f'Error getting alignment: {e}')
            return im, False
        else:
            im = cv2.warpAffine(im, warp_matrix, (h, w),
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            im[im == 0] = np.mean(im)
            return im, True

    def _get_alignment(self, im_ref, im_to_align, key):
        self.logger.info(f'Getting alignment for {key}')
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-8)
        cc, warp_matrix = cv2.findTransformECC(
            im_ref, im_to_align, warp_matrix, warp_mode, criteria)

        matrix_str = str(warp_matrix).replace('\n', '')
        self.logger.info(
            f"Got alignment for {key} with cc {cc:.3f}: {matrix_str}")
        return cc, warp_matrix


class DSTL(BaseDataLoader):
    def __init__(self, training_band_groups, batch_size,
                 num_workers=1, val=False, shuffle=False, flip=False,
                 rotate=False, blur=False, augment=False, return_id=False,
                 **params):

        # Scale the bands to be between 0 - 255
        # Min Max for Type P: [[0, 2047]]
        # Min Max for Type RGB: [[1, 2047], [157, 2047], [91, 2047]]
        # Min Max for Type M: [[156, 2047], [115, 2047], [87, 2047], [55, 2047], [1, 2047], [84, 2047], [160, 2047], [111, 2047]]
        # Min Max for Type A: [[671, 15562], [489, 16383], [434, 16383], [390, 16383], [1, 16383], [129, 16383], [186, 16383], [1, 16383]]
        # P is 11bit, RGB is 11bit, M is 11bit, A is 14bit

        # TODO construct the std and means only from the bands that are being
        #  trained on.
        dstl_data_path = os.environ.get('DSTL_DATA_PATH')
        kwargs = {
            'root': dstl_data_path,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val,
            'mean': [],
            'std': [],
        }

        print("kwargs", kwargs)
        print("params", params)



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

        self.dataset = DSTLDataset(_wkt_data, training_band_groups, **params, **kwargs)

        print("LENGTH: ", len(self.dataset))
        # if split in ["train_rgb", "trainval_rgb", "val_rgb", "test_rgb"]:
        # self.dataset = DSTLDatasetRGB(**kwargs)
        # elif split in ["train", "trainval", "val", "test"]:
        #     self.dataset = VOCDataset(**kwargs)
        else:
            raise ValueError(f"Invalid split name {split}")
        super(DSTL, self).__init__(self.dataset, batch_size, shuffle,
                                   num_workers, train_indxs, val_indxs)

