import json
from pathlib import Path

root_dir = '/home/hoangqc/Datasets/nerf/piazza_san_marco/ns/'

#load transform.json
with open(root_dir + 'transforms.json', 'r') as f:
    transforms = json.load(f)

valid_idx = []
for i in range(len(transforms['frames'])):
    transform = transforms['frames'][i]
    path_img = transform['file_path']
    # check path is not available -> remove from JSON
    if Path(root_dir + path_img).exists():
        valid_idx.append(i)


#reduce transforms['frames'] with valid_idx
transforms['frames'] = [transforms['frames'][i] for i in valid_idx]

#save new json file
with open(root_dir + 'transforms_r.json', 'w') as f:
    json.dump(transforms, f)
print('Done')