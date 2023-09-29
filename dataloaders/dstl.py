# Originally written by Kazuto Nakashima
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
from PIL import Image
from typing import Dict
from pathlib import Path
from itertools import islice
import json
import csv


class DSTLDataset(BaseDataSet):

    def __init__(self, mean: [float], **kwargs):
        self.classes = [1, 10]
        self.num_classes = len(self.classes)
        self.palette = palette.get_voc_palette(self.num_classes)
        self._wkt_data = None

        all_im_ids = list(self.get_wkt_data())
        class_stats_dir = os.path.join(self.root, 'cls-stats.json')
        class_label_stats = json.loads(Path(class_stats_dir).read_text())
        labeled_area = [(im_id, np.mean([class_label_stats[im_id][str(cls)]['area']
                                    for cls in self.classes]))
                   for im_id in all_im_ids]


        super(DSTLDataset, self).__init__(mean=mean, **kwargs)

    def get_wkt_data(self) -> Dict[str, Dict[int, str]]:
        _wkt_data = {}
        with open(self.root + '/train_wkt_v4.csv/train_wkt_v4.csv') as f:
            for im_id, poly_type, poly in islice(csv.reader(f), 1, None):
                _wkt_data.setdefault(im_id, {})[int(poly_type)] = poly
        return _wkt_data


class DSTLDatasetP(DSTLDataset):

    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.get_voc_palette(self.num_classes)
        super(DSTLDatasetP, self).__init__(**kwargs)

    def _set_files(self):
        self.image_dir = os.path.join(self.root, '/sixteen_band/sixteen_band')
        file_list = os.path.join(self.image_dir, self.split + ".txt")

        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]

    def _load_data(self, index):
        # TODO do we get 80% train 10% valid and 10% test for each image,
        #  or do we split the images them selves into 80% train 10% valid and 10% test?

        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '.png')
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id


class DSTLDatasetRGB(DSTLDataset):

    def __init__(self, **kwargs):
        self.num_classes = 1
        self.palette = palette.get_voc_palette(self.num_classes)
        super(DSTLDatasetRGB, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, 'three_band/three_band')
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.label_dir = os.path.join(self.root, 'SegmentationClass')

        file_list = os.path.join(self.root, "ImageSets/Segmentation",
                                 self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '.png')
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id

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

        self.STD =  [125.6897460373406, 185.7050473919206,
                     143.04137788295327,  93.08276278131164,
         36.02164708172559, 93.00104122153027,142.86657417238337,
                     146.35837445638936, 184.8630489536806,
                     119.47909534713612,  205.92902871666232,
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

        if split in ["train_rgb", "trainval_rgb", "val_rgb", "test_rgb"]:
            self.dataset = DSTLDatasetRGB(**kwargs)
        # elif split in ["train", "trainval", "val", "test"]:
        #     self.dataset = VOCDataset(**kwargs)
        else:
            raise ValueError(f"Invalid split name {split}")
        super(DSTL, self).__init__(self.dataset, batch_size, shuffle,
                                  num_workers, val_split)

