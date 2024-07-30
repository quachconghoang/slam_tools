from pathlib import Path
import os
import torch
import kornia as K
import pandas as pd
import numpy as np
import h5py


def get_imc2024_path():
    root_path = Path('/')
    if os.getenv('LOCAL_DATASETS'):
        root_path = Path(os.getenv('LOCAL_DATASETS'))
    input_path = root_path / 'kaggle' / 'input'
    # working_path = root_path / 'kaggle' / 'working'
    imc_path = input_path / 'image-matching-challenge-2024'
    return imc_path

def image_path_gen(row):
    row['image_path'] = 'train/' + row['dataset'] + '/images/' + row['image_name']
    return row

IMC_PATH = get_imc2024_path()
train_df = pd.read_csv(IMC_PATH / 'train/train_labels.csv')
test_df = train_df.apply(image_path_gen, axis=1).drop_duplicates(subset=['image_path'])

for img_path in test_df.image_path.to_list():
    if not os.path.exists(IMC_PATH / img_path):
        test_df = test_df[test_df.image_path != img_path]
        print('Not Found - Removed: ', IMC_PATH / img_path)
img_gt_paths = test_df.image_path.to_list()

### find duplicate in img_gt_paths

### list all png file in imc_path
### convert to string and remove ***/train/ from path
# png_files = [f for f in Path(IMC_PATH/'train').rglob('*.png')]
# png_files = [str(f).replace((str(IMC_PATH)+'/'),'') for f in png_files]
# for png_file in png_files:
#     if not png_file in img_gt_paths:
#         print('Not have groundtruth : ', png_file)
# print(len(png_files), len(img_gt_paths))

scenes = ['church', 'dioscuri', 'lizard', 'multi-temporal-temple-baalshamin',
          'pond', 'transp_obj_glass_cup', 'transp_obj_glass_cylinder']

### get church dataset in test_df
# church_df = test_df[test_df['dataset'] == 'church']
# imgs = church_df.image_name.to_list()

# DEVICE: torch.device = K.utils.get_cuda_device_if_available(0)
# print(DEVICE)


