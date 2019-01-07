from config import test_cfg
from test import load_model_test
from torchclassify import utils
import os.path as osp
import math
import os

test_output, test_imgs_path = load_model_test()

if not osp.isdir(test_cfg['result_dir']):
    os.makedirs(test_cfg['result_dir'])
result_filename = test_cfg['result_filename']
if result_filename is None:
    result_filename = utils.get_timestamp()+'.csv'
result_filepath = osp.join(test_cfg['result_dir'], result_filename)

# custom code for output
result = []
for output, img_path in zip(test_output, test_imgs_path):
    img_id = int(osp.basename(img_path).split('.')[0])
    cat_output, dog_output = output
    dog_p = math.exp(dog_output)/(math.exp(dog_output)+math.exp(cat_output))
    result.append((img_id, dog_p))
result.sort()

with open(result_filepath, 'w') as f:
    f.write("id,label\n")
    for img_id, dog_p in result:
        f.write(f"{img_id},{dog_p}\n")
