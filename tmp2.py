import numpy as np
import pandas as pd
import rasterio

means = []
stds = []
num_bands = 17

# File to load metadata for the training data set
# file_path = dataset_path + 'dstl_data.csv'
# root_path = '/mnt/e/ML_DATA/DSTL/dstl-satellite-imagery-feature-detection/'
root_path = '/opt/home/s3630120/dstl-satellite-imagery-feature-detection/'
dataset_path = root_path + 'cached/'
file_path = root_path + 'train_wkt_v4.csv/train_wkt_v4.csv'

df = pd.read_csv(file_path)
ids = df['ImageId'].unique().tolist()

data_list = dict({bnd: np.array([]) for bnd in range(num_bands)})
for img_id in ids:
    print(img_id)
    path = dataset_path + f'{img_id}_interp_4.tif'
    with rasterio.open(path) as src:
        data = src.read()
        data = data.reshape(src.count, src.width * src.height)
        for bnd in range(num_bands):
            data_list[bnd] = np.concatenate((data_list[bnd], data[bnd, :]),
                                            axis=0)

for bnd in range(num_bands):
    print(f"Std for Band {bnd + 1}: {np.std(data_list[bnd])}")
