# Written by Luke Mason
# Date: 2020-06-03
# Purpose: To create a dataloader for the DSTL dataset.


import csv
import json
import os
from itertools import islice
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import rasterio
import shapely.affinity
import shapely.geometry
import shapely.wkt
from PIL import Image
from base import BaseDataSet, BaseDataLoader
from shapely.geometry import MultiPolygon
from utils import palette
import cv2

# 10 is the unknown class because 0 - 9 are class ids in the DSTL dataset.
unknown_class_id = 10


class DSTLDataset(BaseDataSet):

    def __init__(self, classes: List[int], bands: List[int], merging_strategy: str, hold_imagery_in_memory: bool,
                 mean: List[float], **kwargs):
        """
        :param hold_imagery_in_memory: Whether to hold the imagery in memory or not.
        :param file_type: The file type of the imagery. Either 'A', 'P',
        'M', 'RGB' or a combination of them i.e ARGBM.
        :param image_sub_dir: The sub directory of the imagery within the root
        directory.
        :param mean: The mean value of the imagery bands.
        :param kwargs: The keyword arguments.
        """

        self.bands = bands

        # The label data but is represented as in WKT format.
        # img_id -> class_id, WKT
        self._wkt_data = None  # type: Dict[str, Dict[int, str]] or None

        # The map of image ids to max x and y values.
        self._x_max_y_min = None  # type: Dict[str, Tuple[float, float]] or None
        self.hold_imagery_in_memory = hold_imagery_in_memory
        self.root = kwargs.get('root')
        self.image_dir = os.path.join(self.root, 'sixteen_bands/sixteen_bands')
        self.images = {}  # type Dict[str, np.ndarray]
        self.info = dict()  # type Dict[str, Dict[str, Any]]
        # The classes to detect
        self.classes = classes

        # The number of classes
        self.num_classes = len(self.classes)

        # The colour palette for the classes
        self.palette = palette.get_voc_palette(self.num_classes)

        all_train_ids = list(self._get_wkt_data())
        print(all_train_ids)
        self.class_label_stats = json.loads(Path(
            'dataloaders/labels/dstl-stats.json').resolve().read_text())
        labeled_area = [
            (im_id,
             np.mean([self.class_label_stats[im_id][str(cls)]['area'] for cls in
                      self.classes]))
            for im_id in all_train_ids
        ]

        print('Labelled area: ', labeled_area)

        super(DSTLDataset, self).__init__(mean=mean, **kwargs)

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
            return ''

    def _set_files(self):
        ids = list(self.class_label_stats.keys())

        # We want to know how many patches can be made out of all the images
        # in the dataset and then figure out the amount of patches that can
        # be created from it by returning the set of 'files' but really is a
        # set of patches in all files. Each patch is multipled by amount of bands
        patches = []  # type: List[Tuple[str, Tuple[int, int, int]]]
        step_size = self.patch_size - (
                self.overlap_percentage * self.patch_size)
        for img_id in ids:
            for band_id in self.bands:
                file_name = self._get_filename(img_id, band_id)
                image_path = os.path.join(self.image_dir, file_name)
                width, height = self._image_size(image_path)

                # TODO merging or stacking strategy could change the amount of patches.
                # TODO create a logger that logs all the details of the dataloader.
                chunk_offsets = self._chunk_offsets(width, height, step_size)
                patches.extend([(img_id, chunk, band + 1) for chunk in
                                chunk_offsets for band in range(self.bands)])

        print('Amount of images: ', len(self.class_label_stats))
        print('Amount of patches: ', len(patches))

        self.files = patches

    def _load_data(self, index: int):
        image_id, chunk, band_id = self.files[index]
        file_type, band_idx = self._get_file_type_and_band_index(band_id)
        file_name = self._get_filename(image_id, file_type)

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
                label = self._gen_label_mask(image_id, src.height, src.width, image)

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

    def _image_size(self, path):
        stats = os.stat(path)
        return stats.st_size

    def _chunk_offsets(self, width: int, height: int, step_size: int) -> List[
        Tuple[int, int]]:
        """
        Returns a list of (x, y) offsets corresponding to chunks of the image
        TODO - make this utilise left over pixels that don't fit the step size.
        :param height: The height of the image.
        :param step_size: The step size to use when generating the chunks.
        :return: A list of (x, y) offsets.
        """
        x_offsets = range(0, width - step_size + 1, step_size)
        y_offsets = range(0, height - step_size + 1, step_size)
        chunk_offsets = [(x, y) for x in x_offsets for y in y_offsets]

        return chunk_offsets

    def _gen_label_mask(self, image_id: str, height: int, width: int, image: np.ndarray):
        """
        Returns the mask for the entire image so that it can be chipped.
        :param image: The image.
        :return: The label_mask for the image.
        """
        # The semantic segmentation map where each class id is an element of
        # the mask.
        #  np.full(image.shape, unknown_class_id, dtype=np.int32)
        class_to_polygons = self._load_polygons(image_id, height, width)
        label_mask = np.array(
            [self._mask_for_polygons(image, class_to_polygons[cls + 1])
             for cls in range(self.hps.total_classes)], dtype=np.uint8)
        # TODO get class mask for each class
        # for cls_idx, cls in enumerate(self.classes):
        #     poly = self._wkt_data[image_id][cls]
        #     if poly is None:
        #         continue
        #     label_mask[cls_idx] = self._poly_to_mask(poly, image_id)

        return label_mask

    def _mask_for_polygons(self, image: np.ndarray, polygons: MultiPolygon) -> np.ndarray:
        """ Return numpy mask for given polygons.
        polygons should already be converted to image coordinates.
        """
        label_mask = image[:, :, unknown_class_id]

        if not polygons:
            return label_mask
        int_coords = lambda x: np.array(x).round().astype(np.int32)
        exteriors = [int_coords(poly.exterior.coords) for poly in polygons.geoms]
        interiors = [int_coords(pi.coords) for poly in polygons.geoms
                    for pi in poly.interiors]
        cv2.fillPoly(label_mask, exteriors, 1)
        cv2.fillPoly(label_mask, interiors, 0)
        return label_mask

    def _get_filename(self, img_id: str, file_type: str):
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
            xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
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


class DSTLDatasetRGB(DSTLDataset):

    def __init__(self, **kwargs):
        super(DSTLDatasetRGB, self).__init__([1], [2,3,4], 'grey_scale', True, **kwargs)


class DSTL(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None,
                 base_size=None, scale=True, num_workers=1, val=False,
                 shuffle=False, flip=False, rotate=False, blur=False,
                 augment=False, val_split=None, return_id=False):

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

        # std = np.array([
        #             62.00827863,  46.65453694,  24.7612776,   54.50255552,
        #             13.48645938,  24.76103598,  46.52145521,  62.36207267,
        #             61.54443128,  59.2848377,   85.72930307,  68.62678882,
        #             448.43441827, 634.79572682, 567.21509273, 523.10079804,
        #             530.42441592, 461.8304455,  486.95994727, 478.63768386],
        #             dtype=np.float32)
        #         mean = np.array([
        #             413.62140162,  459.99189475,  325.6722122,   502.57730746,
        #             294.6884949,   325.82117752,  460.0356966,   482.39001004,
        #             413.79388678,  527.57681818,  678.22878001,  529.64198655,
        #             4243.25847972, 4473.47956815, 4178.84648439, 3708.16482918,
        #             2887.49330138, 2589.61786722, 2525.53347208, 2417.23798598],
        #             dtype=np.float32)

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
            self.dataset = DSTLDatasetP(**kwargs)

        # if split in ["train_rgb", "trainval_rgb", "val_rgb", "test_rgb"]:
        # self.dataset = DSTLDatasetRGB(**kwargs)
        # elif split in ["train", "trainval", "val", "test"]:
        #     self.dataset = VOCDataset(**kwargs)
        else:
            raise ValueError(f"Invalid split name {split}")
        super(DSTL, self).__init__(self.dataset, batch_size, shuffle,
                                   num_workers, val_split)
