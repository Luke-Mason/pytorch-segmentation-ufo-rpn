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
from utils import palette

# 10 is the unknown class because 0 - 9 are class ids in the DSTL dataset.
unknown_class_id = 10


class DSTLDataset(BaseDataSet):

    def __init__(self,
                 classes: List[int],
                 training_bands: List[int],
                 patch_size: int,
                 overlap_percentage: float,
                 align_images: bool,
                 interpolation_method: int,
                 **kwargs):
        """Constructor, initialiser.

        Args:
            classes (List[int]): The class labels to train on.
            training_bands (List[int]): The colour spectrum bands to train on.
            align_images (bool): Align the images.
            interpolation_method (int): The interpolation method to use.
        """
        # Attributes
        self.classes = classes
        self.num_classes = len(classes)
        self.training_bands = training_bands
        self.patch_size = patch_size
        self.overlap_percentage = overlap_percentage
        self.align_images = align_images
        self.interpolation_method = interpolation_method

        # Setup directories and paths
        self.root = kwargs['root']
        self.image_dir = os.path.join(self.root, 'sixteen_band/sixteen_band')
        self.cache_dir = os.path.join(self.root, 'cached')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Memeory Cache
        self.images = dict()  # type Dict[str, np.ndarray]
        self.labels = dict()  # type Dict[str, np.ndarray]
        self.info = dict()  # type Dict[str, Dict[str, Any]]
        self._wkt_data = None  # type List[Tuple[str, str]]
        self._x_max_y_min = None  # type Dict[str, Tuple[int, int]]

        # Logging
        self._setup_logging()

        # The colour palette for the classes
        self.palette = palette.get_voc_palette(len(self.classes))

        self.class_label_stats = json.loads(Path(
            'dataloaders/labels/dstl-stats.json').resolve().read_text())

        # TODO split still and validation
        all_train_ids = list(self._get_wkt_data())
        labeled_area = [
            (im_id,
             np.mean([self.class_label_stats[im_id][str(cls)]['area'] for cls in
                      self.classes]))
            for im_id in all_train_ids
        ]

        print('Labelled area: ', labeled_area)

        super(DSTLDataset, self).__init__(**kwargs)

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Logs to file

        # Check if log directory exist, if not create it
        log_dir = os.path.join(self.root, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file_name = datetime.datetime.now().strftime(
            '%Y-%m-%d_%H-%M-%S.log')
        handler = logging.FileHandler(os.path.join(log_dir, log_file_name))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Logs to screen
        # handler = logging.StreamHandler()
        # handler.setLevel(
        #     logging.INFO)  # Set the desired logging level for this handler
        # handler.setFormatter(logging.Formatter(
        #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        # self.logger.addHandler(handler)

    def _set_files(self):
        ids = list(self.class_label_stats.keys())

        # Preprocess
        self.logger.debug('Preprocessing images')
        for image_id in ids:
            preprocessed_filename = self.get_preprocessed_filename(image_id,
                                                                   self.align_images)
            preprocessed_file_path = os.path.join(self.cache_dir,
                                                  preprocessed_filename)
            if not os.path.exists(preprocessed_file_path):
                self.logger.info("Preprocessing image: " + image_id)
                self._preprocess_image(image_id, self.align_images,
                                       preprocessed_file_path)
                self.logger.info("Preprocessing image: " + image_id + " done.")
        self.logger.debug('Preprocessing images done.')
        step_size = self.patch_size - ((self.overlap_percentage / 100.0) *
                                       self.patch_size)
        step_size = math.ceil(step_size)

        self.logger.debug(f'Step size: {step_size}')

        # We want to know how many patches can be made out of all the images
        # in the dataset and then figure out the amount of patches that can
        # be created from it by returning the set of 'files' but really is a
        # set of patches in all files. Each patch is multipled by amount of bands
        self.logger.info('Calculating patches and chunk offsets...')
        patches = []  # type: List[Tuple[str, Tuple[int, int, int]]]
        for img_id in ids:
            file_name = self.get_preprocessed_filename(image_id,
                                                       self.align_images)

            image_path = os.path.join(self.cache_dir, file_name)
            width, height = self._image_size(image_path)

            # TODO merging or stacking strategy could change the amount of patches.
            chunk_offsets = self._gen_chunk_offsets(width, height,
                                                    step_size)
            self.logger.debug(
                f'Chunk offsets for {img_id}: {chunk_offsets}')

            # We want a patch to be a cutout of a band, so chunks * bands *
            # images is our total amount of patches.
            patches.extend(
                [(img_id, chunk, band_id) for chunk in chunk_offsets for band_id
                 in self.training_bands])

        self.logger.info('Calculating patches and chunk offsets done.')
        self.logger.debug(f'Amount of images: {len(self.class_label_stats)}')
        self.logger.debug(f'Amount of bands: {len(self.training_bands)}')
        self.logger.debug(f'Amount of patches: {len(patches)}')

        self.files = patches

    def _load_data(self, index: int):
        image_id, chunk, band_id = self.files[index]

        # If not loaded before, load it and generate mask for it
        if index not in self.images:
            self.logger.debug(f'Loading image {image_id} band: {band_id}')
            file_name = self.get_preprocessed_filename(image_id,
                                                       self.align_images)
            image_path = os.path.join(self.cache_dir, file_name)
            with rasterio.open(image_path, dtype=np.float32) as src:
                self.images[index] = np.array(src.read(band_id),
                                              dtype=np.float32)
                self.logger.debug(f'Loaded image {image_id} band: {band_id}')
                self.info[index] = (src.count, src.width, src.height)
                self.logger.debug(f'Image info: {self.info[index]}')
                self.labels[index] = self._gen_y_label_mask(image_id,
                                                            src.height,
                                                            src.width,
                                                            self.images[index])

        self.logger.debug(f'Using data for index: {index}')
        info = self.info[index]
        image_band = np.expand_dims(self.images[index], axis=2)
        label_mask = np.expand_dims(self.labels[index], axis=2)
        self.logger.debug(f'Image band shape: {image_band.shape}')
        self.logger.debug(f'Label mask shape: {label_mask.shape}')
        # TODO merge how?

        # Cut out the patch of the image
        x, y = chunk
        self.logger.debug(f'Chunk: {chunk}')

        patch = np.copy(
            image_band[y:y + self.patch_size, x:x + self.patch_size, :])
        patch_y_mask = np.copy(
            label_mask[y:y + self.patch_size, x:x + self.patch_size, :])

        self.logger.debug(f'Patch shape: {patch.shape}')
        self.logger.debug(f'Patch mask shape: {patch_y_mask.shape}')

        return patch, patch_y_mask, image_id

    def __getitem__(self, index):
        patch, patch_y_mask, image_id = self._load_data(index)
        if self.val:
            patch, patch_y_mask = self._val_augmentation(patch, patch_y_mask)
        elif self.augment:
            patch, patch_y_mask = self._augmentation(patch, patch_y_mask)

        patch_y_mask = torch.from_numpy(np.array(patch_y_mask,
                                                 dtype=np.float32)).long()
        # patch = Image.fromarray(np.float32(patch))
        if self.return_id:
            return self.normalize(
                self.to_tensor(patch)), patch_y_mask, image_id
        return self.normalize(self.to_tensor(patch)), patch_y_mask

    def _image_size(self, path: str):
        with rasterio.open(path) as src:
            return src.width, src.height

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
        if height % step_size != 0:
            y_offsets.append(height - step_size)
        chunk_offsets = [(x, y) for x in x_offsets for y in y_offsets]

        return chunk_offsets

    def _gen_y_label_mask(self, image_id: str, height: int, width: int,
                          image: np.ndarray):
        """
            For each patch generate the label mask over the classes so
        that we can get the loss for each class for each patch. You need to
        find the polygons that are within the patch and then generate the
        mask. Maybe calculate the coordinate per pixel and then use the chunk
        offset to get the polygon coordinate to do an intersection with.
            Returns the mask for the entire image so that it can be chipped.
        :param image: The image.
        :return: The label_mask for the image.
        """
        self.logger.debug(f'Generating label mask for image: {image_id}')
        class_to_polygons = self._load_polygons(image_id, height, width)
        polygons = self._mask_from_polygons(image, class_to_polygons)
        self.logger.debug(f'Generated label mask for image: {image_id}')
        return polygons

    def _mask_from_polygons(self, image: np.ndarray,
                            polygons_map: Dict[
                                int, MultiPolygon]) -> np.ndarray:
        """ Return numpy mask for given polygons.
        polygons should already be converted to image coordinates.
        """
        # The semantic segmentation map where each class id is an element of
        # the mask.
        label_mask = np.full(image.shape, unknown_class_id, dtype=np.float32)
        for class_id, polygons in polygons_map.items():
            if not polygons:
                return label_mask
            int_coords = lambda x: np.array(x).round().astype(np.int32)
            exteriors = [int_coords(poly.exterior.coords) for poly in
                         polygons.geoms]
            interiors = [int_coords(pi.coords) for poly in polygons.geoms
                         for pi in poly.interiors]

            # Cut out parts of current mask
            extracted_values = extract_mask_values_using_polygons(image,
                                                                  interiors)

            # Fill pixels with the polygon exterior convex with the class id
            cv2.fillPoly(label_mask, exteriors, class_id)

            # Apply polygon interior holes with saved mask data, where 0 are non values
            label_mask[extracted_values != -1] = extracted_values[
                extracted_values != -1]

        return label_mask

    def _get_filename(self, img_id: str, band_id: int):
        file_type, _ = self._get_file_type_and_band_index(band_id)
        return f"/{img_id}_{file_type}.tif"

    def _get_image_info(self, filename) -> Tuple[int, int, int]:
        info = self.info[filename]
        if info is None:
            with rasterio.open(filename) as src:
                info = (src.width, src.height, src.count)
                self.info[filename] = info

        return info

    def _get_wkt_data(self) -> Dict[str, Dict[int, str]]:
        if self._wkt_data is None:
            self._wkt_data = {}

            # Load the CSV into a DataFrame
            df = pd.read_csv(
                os.path.join(self.root, 'train_wkt_v4.csv/train_wkt_v4.csv'))

            for index, row in df.iterrows():
                im_id = row['ImageId']
                class_type = row['ClassType']
                poly = row['MultipolygonWKT']
                # Add the polygon to the dictionary
                self._wkt_data.setdefault(im_id, {})[int(class_type)] = poly

        return self._wkt_data

    def _load_polygons(self, image_id: str, height: int, width: int) \
            -> Dict[int, MultiPolygon]:
        """
        Load the polygons for the image id and scale them to the image size.
        :param im_id: The image id.
        :param im_size: The image size.
        :return: A dictionary of class type to polygon.
        """
        self.logger.debug(f'Loading polygons for image: {image_id}')
        x_max, y_min = self._get_x_max_y_min(image_id)
        x_scaler, y_scaler = self._get_scalers(height, width, x_max, y_min)

        items_ = {
            int(poly_type): shapely.affinity.scale(shapely.wkt.loads(poly),
                                                   xfact=x_scaler,
                                                   yfact=y_scaler,
                                                   origin=(0, 0, 0)) for
            poly_type, poly in self._get_wkt_data()[image_id].items()}
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

    def get_preprocessed_filename(self, image_id: str, align_images=True):
        return f"{image_id}{'_aligned' if align_images else ''}_interp_{self.interpolation_method}.tif"

    def _preprocess_image(self, image_id: str, align_images: bool,
                          file_path: str):
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
        if align_images:
            im_p, _ = self._aligned(im_rgb, im_p, key=key('P'))
            im_m, aligned = self._aligned(im_rgb, im_m, im_m[:, :, :3],
                                          key=key('M'))
            im_ref = im_m[:, :, -1] if aligned else im_rgb[:, :, 0]
            im_a, _ = self._aligned(im_ref, im_a, im_a[:, :, 0], key=key('A'))

        # Resize the images to be the same size as RGB.
        # Sometimes panchromatic is a couple of pixels different to RGB
        if im_p.shape != im_rgb.shape[:2]:
            im_p = cv2.resize(im_p, (h, w),
                              interpolation=self.interpolation_method)
        im_p = np.expand_dims(im_p, 2)
        im_m = cv2.resize(im_m, (h, w), interpolation=self.interpolation_method)
        im_a = cv2.resize(im_a, (h, w), interpolation=self.interpolation_method)

        # Scale images between 0-1 based off their maximum and minimum bounds
        # P and M images are 11bit integers, A is 14bit integers, scale values to be
        # between 0-1 floats
        im_p = im_p / 2047.0
        im_m = im_m / 2047.0
        im_a = im_a / 16383.0
        image = np.concatenate([im_p, im_m, im_a], axis=2).transpose([2, 0, 1])
        self.logger.debug(f"Image shape: {image.shape}")
        # Save images
        with rasterio.open(file_path, 'w',
                           count=image.shape[0],
                           height=image.shape[1],
                           width=image.shape[2],
                           dtype=np.float32) as src:
            self.logger.debug(f"Saving image to {file_path}")
            src.write(image)

        # Close streams
        im_rgb_src.close()
        im_p_src.close()
        im_m_src.close()
        im_a_src.close()

        del im_rgb, im_p, im_m, im_a, im_rgb_src, im_p_src, im_m_src, im_a_src, \
            rgb_path, p_path, m_path, a_path

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
    def __init__(self, data_dir, batch_size, split, crop_size=None,
                 base_size=None, scale=True, num_workers=1, val=False,
                 shuffle=False, flip=False, rotate=False, blur=False,
                 augment=False, val_split=None, return_id=False):

        # Scale the bands to be between 0 - 255
        # Min Max for Type P: [[0, 2047]]
        # Min Max for Type RGB: [[1, 2047], [157, 2047], [91, 2047]]
        # Min Max for Type M: [[156, 2047], [115, 2047], [87, 2047], [55, 2047], [1, 2047], [84, 2047], [160, 2047], [111, 2047]]
        # Min Max for Type A: [[671, 15562], [489, 16383], [434, 16383], [390, 16383], [1, 16383], [129, 16383], [186, 16383], [1, 16383]]
        # P is 11bit, RGB is 11bit, M is 11bit, A is 14bit

        # Preprocessed means and standard deviations for all 20 bands (P-1, RGB-3, M-8, A-8)
        self.MEAN = [
            503.58188970939926, 427.0634343071277, 464.8759277981281,
            329.2235720071117, 296.5393404081892,
            329.3482441387011, 464.96379164466254, 487.47826285625166,
            427.30320381069293, 531.2880633344082, 689.7117900623391,
            531.1469588868024, 4303.20706921944, 4558.573217857693,
            4250.339562130438, 3783.3650561624863, 2953.575996307124,
            2650.5233486470447, 2586.8447783175434, 2486.5040951355154]

        self.STD = [125.6897460373406, 185.7050473919206,
                    143.04137788295327, 93.08276278131164,
                    36.02164708172559, 93.00104122153027, 142.86657417238337,
                    146.35837445638936, 184.8630489536806,
                    119.47909534713612, 205.92902871666232,
                    107.97644188228902,
                    1498.283301243493, 2033.058786394439, 1775.6160960734042,
                    1735.6225240596295, 1558.8727446060877, 1375.4157376742348,
                    1379.5742405109147, 1429.8340679224425]
        # TODO construct the std and means only from the bands that are being
        #  trained on.

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            # 'augment': augment,
            # 'crop_size': crop_size,
            # 'base_size': base_size,
            # 'scale': scale,
            # 'flip': flip,
            # 'blur': blur,
            # 'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        print("kwargs", kwargs)

        params = {
            "classes": [1],  # TODO Test with all classes
            "training_bands": [2, 3, 4],  # TODO Test with all bands
            "patch_size": 512,
            "overlap_percentage": 10,
            "align_images": False,
            "interpolation_method": cv2.INTER_LANCZOS4,
        }

        if split in ["train", "trainval", "val", "test"]:
            self.dataset = DSTLDataset(**params, **kwargs)

        # if split in ["train_rgb", "trainval_rgb", "val_rgb", "test_rgb"]:
        # self.dataset = DSTLDatasetRGB(**kwargs)
        # elif split in ["train", "trainval", "val", "test"]:
        #     self.dataset = VOCDataset(**kwargs)
        else:
            raise ValueError(f"Invalid split name {split}")
        super(DSTL, self).__init__(self.dataset, batch_size, shuffle,
                                   num_workers, val_split)


def extract_mask_values_using_polygons(mask: np.ndarray,
                                       polygons: MultiPolygon):
    """ Return numpy mask for given polygons.
        polygons should already be converted to image coordinates.
        non values are given -1.
    """
    # Mark the values to extract with a 1.
    mark_value = 1
    marked_mask = np.zeros(mask.shape, dtype=np.int8)
    cv2.fillPoly(marked_mask, polygons, mark_value)

    # Extract the values from the main mask using the marked mask
    extracted_values_mask = np.full(marked_mask.shape, -1, dtype=np.float32)
    for index, element in np.ndenumerate(marked_mask):
        if element == mark_value:
            extracted_values_mask[index] = mask[index]
    return extracted_values_mask
