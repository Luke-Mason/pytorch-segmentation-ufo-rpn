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
import numpy as np
import pandas as pd
import rasterio
import shapely.affinity
import shapely.geometry
import shapely.wkt
from base import BaseDataSet, BaseDataLoader
from shapely.geometry import MultiPolygon
from utils import palette

# 10 is the unknown class because 0 - 9 are class ids in the DSTL dataset.
unknown_class_id = 10


class DSTLDataset(BaseDataSet):

    def __init__(self, classes: List[int], training_bands: List[int],
                 merging_strategy: str, hold_imagery_in_memory: bool,
                 **kwargs):
        """Constructor, initialiser.

        Args:
            classes (List[int]): The class labels to train on.
            training_bands (List[int]): The colour spectrum bands to train on.
            merging_strategy (str): (optional) The merge strategy for the bands.
            hold_imagery_in_memory (bool): hold the imagery in memory, cache different stages in memory.
        """
        self._setup_logging()

        self.training_bands = training_bands
        self.hold_imagery_in_memory = hold_imagery_in_memory
        self.classes = classes
        self.root = kwargs['root']
        self.image_dir = os.path.join(self.root, 'sixteen_bands/sixteen_bands')
        self.images = {}  # type Dict[str, np.ndarray]
        self.info = dict()  # type Dict[str, Dict[str, Any]]

        # The colour palette for the classes
        self.palette = palette.get_voc_palette(len(self.classes))

        all_train_ids = list(self._get_wkt_data())
        self.class_label_stats = json.loads(Path(
            'dataloaders/labels/dstl-stats.json').resolve().read_text())

        # TODO split still and validation
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
        log_file_name = datetime.datetime.now().strftime(
            '%Y-%m-%d_%H-%M-%S.log')
        handler = logging.FileHandler(os.path.join('logs', log_file_name))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def _set_files(self):
        ids = list(self.class_label_stats.keys())
        largest_dims = self._get_file_dims(ids)

        # TODO fix the edge case the the image is lower resolution than the patch size.
        step_size = self.patch_size - (
                self.overlap_percentage * self.patch_size)
        self.logger.debug(f'Step size: {step_size}')

        # We want to know how many patches can be made out of all the images
        # in the dataset and then figure out the amount of patches that can
        # be created from it by returning the set of 'files' but really is a
        # set of patches in all files. Each patch is multipled by amount of bands
        patches = []  # type: List[Tuple[str, Tuple[int, int, int]]]
        for img_id in ids:
            for band_id in self.training_bands:
                # TODO scale the lower resolution bands to the largest band resolution for image.

                file_name = self._get_filename(img_id, band_id)
                image_path = os.path.join(self.image_dir, file_name)
                width, height = self._image_size(image_path)

                # TODO merging or stacking strategy could change the amount of patches.
                # TODO create a logger that logs all the details of the dataloader.
                chunk_offsets = self._gen_chunk_offsets(width, height,
                                                        step_size)
                self.logger.debug(
                    f'Chunk offsets for {img_id}: {chunk_offsets}')

                patches.extend(
                    [(img_id, chunk, band_id) for chunk in chunk_offsets])

        self.logger.debug(f'Amount of images: {len(self.class_label_stats)}')
        self.logger.debug(f'Amount of bands: {len(self.training_bands)}')
        self.logger.debug(f'Amount of patches: {len(patches)}')

        self.files = patches

    def _load_data(self, index: int):
        image_id, chunk, band_id = self.files[index]
        file_type, band_idx = self._get_file_type_and_band_index(band_id)
        file_name = self._get_filename(image_id, band_id)

        # Caches - we store all imagery data and info loaded in RAM.
        info = self.info[file_name]
        image_idx = f'{image_id}-{band_idx}'
        image = self.images[image_idx]

        # If not loaded before, load it and generate mask for it
        if image is None:
            image_path = os.path.join(self.image_dir, file_name)
            with rasterio.open(image_path) as src:
                info = (src.width, src.height, src.count)
                image = np.array(src.read(band_idx), dtype=np.float32)
                label = self._gen_label_mask(image_id, src.height, src.width,
                                             image)

            self.info[file_name] = info
            self.images[image_idx] = image
            self.labels[image_idx] = label

        # TODO scale each band data to 0 - 255
        # TODO merge how?

        # Cut out the patch of the image
        x, y = chunk
        patch = np.copy(image[y:y + self.patch_size, x:x + self.patch_size, :])

        # For each patch generate the label mask over the classes so
        # that we can get the loss for each class for each patch. You need to
        # find the polygons that are within the patch and then generate the
        # mask. Maybe calculate the coordinate per pixel and then use the chunk
        # offset to get the polygon coordinate to do an intersection with.

        return image, label, image_id

    def _get_file_dims(self, image_ids: List[str]):
        if self.largest_dims is None:
            self.largest_dims = dict()
            for img_id in image_ids:
                # Get the largest resolution available for the image that is needed according
                # the the training bands because the image is split across 3 different files
                # depending on the band spectrum desired in order to scale the lower res to be the higher res.
                file_names = set(
                    [self._get_filename(img_id, band_id) for band_id in
                     self.training_bands])

                # Get the largest resolution from the bands in files.
                largest_image_psqr = 0
                for file in file_names:
                    image_path = os.path.join(self.image_dir, file)
                    width, height = self._image_size(image_path)
                    if width * height > largest_image_psqr:
                        largest_image_psqr = width * height
                        self.largest_dims[img_id] = (width, height)

        return self.largest_dims

    def _image_size(self, path):
        stats = os.stat(path)
        return stats.st_size

    def _get_file_type_and_band_index(self, band_id: int):
        """
        Returns the subdirectory that is responsible for images with that band id,
        and also returns the index of the band within that image.
        i.e band id 7 mught be band 1 in the images within the multispectral subdirectory.

        Args:
            band_id (int): The id of the band across all spectrums of a satellite image with
            the panchromatic being the first band, and the highest band being 1 + 16 = 17th band.

        Raises:
            error: Error if the band id does not fall within the 1 - 17 range.

        Returns:
            _type_: Returns the subdirectory that is responsible for images with that band id,
        and also returns the index of the band within that image.
        """
        if band_id == 1:
            return 'P', band_id
        elif band_id in range(2, 10):
            return 'M', band_id - 1
        elif band_id in range(10, 18):
            return 'A', band_id - 9
        else:
            self.logger.critical(
                f'WRONG BAND ID, must be inclusive of 1 - 17, recieved {band_id}')
            return ''

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

    def _gen_label_mask(self, image_id: str, height: int, width: int,
                        image: np.ndarray):
        """
        Returns the mask for the entire image so that it can be chipped.
        :param image: The image.
        :return: The label_mask for the image.
        """
        class_to_polygons = self._load_polygons(image_id, height, width)
        return self._mask_from_polygons(image, class_to_polygons)

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
        info = self.info.get(filename)
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
                os.path.join(self.root, '/train_wkt_v4.csv/train_wkt_v4.csv'))

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
        x_max, y_min = self._get_x_max_y_min(image_id)
        x_scaler, y_scaler = self._get_scalers(height, width, x_max, y_min)

        return {
            int(poly_type): shapely.affinity.scale(shapely.wkt.loads(poly),
                                                   xfact=x_scaler,
                                                   yfact=y_scaler,
                                                   origin=(0, 0, 0))
            for poly_type, poly in self._get_wkt_data()[image_id].items()
        }

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

        if split in ["train", "trainval", "val", "test"]:
            self.dataset = DSTLDataset([1], [2, 3, 4], 'grey_scale', True,
                                       **kwargs)

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


# TODO use this
def load_image(im_id: str, rgb_only=False, align=True) -> np.ndarray:
    im_rgb = tiff.imread(
        dataset_path + 'three_band/{}.tif'.format(im_id)).transpose([1, 2, 0])
    if rgb_only:
        return im_rgb
    im_p = np.expand_dims(tiff.imread(
        dataset_path + 'sixteen_band/sixteen_band/{}_P.tif'.format(im_id)), 2)
    im_m = tiff.imread(
        dataset_path + 'sixteen_band/sixteen_band/{}_M.tif'.format(
            im_id)).transpose([1, 2, 0])
    im_a = tiff.imread(
        dataset_path + 'sixteen_band/sixteen_band/{}_A.tif'.format(
            im_id)).transpose([1, 2, 0])
    w, h = im_rgb.shape[:2]
    if align:
        key = lambda x: '{}_{}'.format(im_id, x)
        im_p, _ = _aligned(im_rgb, im_p, key=key('p'))
        im_m, aligned = _aligned(im_rgb, im_m, im_m[:, :, :3], key=key('m'))
        im_ref = im_m[:, :, -1] if aligned else im_rgb[:, :, 0]
        im_a, _ = _aligned(im_ref, im_a, im_a[:, :, 0], key=key('a'))
    if im_p.shape != im_rgb.shape[:2]:
        im_p = cv2.resize(im_p, (h, w), interpolation=cv2.INTER_CUBIC)
    im_p = np.expand_dims(im_p, 2)
    im_m = cv2.resize(im_m, (h, w), interpolation=cv2.INTER_CUBIC)
    im_a = cv2.resize(im_a, (h, w), interpolation=cv2.INTER_CUBIC)
    return np.concatenate([im_rgb, im_p, im_m, im_a], axis=2)


def _preprocess_for_alignment(im):
    im = np.squeeze(im)
    if len(im.shape) == 2:
        im = scale_percentile(np.expand_dims(im, 2))
    else:
        assert im.shape[2] == 3, im.shape
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im.astype(np.float32)


def _aligned(im_ref, im, im_to_align=None, key=None):
    w, h = im.shape[:2]
    im_ref = cv2.resize(im_ref, (h, w), interpolation=cv2.INTER_CUBIC)
    im_ref = _preprocess_for_alignment(im_ref)
    if im_to_align is None:
        im_to_align = im
    im_to_align = _preprocess_for_alignment(im_to_align)
    assert im_ref.shape[:2] == im_to_align.shape[:2]
    try:
        cc, warp_matrix = _get_alignment(im_ref, im_to_align, key)
    except cv2.error as e:
        logger.info('Error getting alignment: {}'.format(e))
        return im, False
    else:
        im = cv2.warpAffine(im, warp_matrix, (h, w),
                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        im[im == 0] = np.mean(im)
        return im, True


def _get_alignment(im_ref, im_to_align, key):
    if key is not None:
        cached_path = Path(dataset_root_path + 'align_cache').joinpath(
            '{}.alignment'.format(key))
        if cached_path.exists():
            with cached_path.open('rb') as f:
                return pickle.load(f)
    logger.info('Getting alignment for {}'.format(key))
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-8)
    cc, warp_matrix = cv2.findTransformECC(
        im_ref, im_to_align, warp_matrix, warp_mode, criteria)
    if key is not None:
        with cached_path.open('wb') as f:
            pickle.dump((cc, warp_matrix), f)
    logger.info('Got alignment for {} with cc {:.3f}: {}'
                .format(key, cc, str(warp_matrix).replace('\n', '')))
    return cc, warp_matrix
