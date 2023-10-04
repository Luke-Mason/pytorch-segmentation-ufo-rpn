import os

import numpy as np
import rasterio

means = []
stds = []
num_bands = 17

dataset_path = ('/opt/home/s3630120/dstl-satellite-imagery-feature-detection'
                '/sixteen_band/sixteen_band/cached/')
mean_data_list = dict({bnd: (0, 0) for bnd in range(num_bands)})
data_list = dict({bnd: np.array([]) for bnd in range(num_bands)})

for img_id in ids:
    print(img_id)
    path = dataset_path + f'{img_id}_interp_4.tif'
    if path is None or not os.path.exists(path):
        print(f"Could not find file for image_id: {img_id}")
        continue
    with rasterio.open(path) as src:
        for bnd in range(num_bands):
            data = src.read(bnd + 1)
            data_list[bnd] = np.append(data_list[bnd], data)

            total, count = mean_data_list[bnd]
            mean_data_list[bnd] = (
                total + np.sum(data), count + (data.shape[0] * data.shape[1]))

for bnd in range(num_bands):
    print(
        f"Mean for Band {bnd + 1}: {mean_data_list[bnd][0] / mean_data_list[bnd][1]}")

for bnd in range(num_bands):
    print(f"Std for Band {bnd + 1}: {np.std(data_list[bnd])}")
