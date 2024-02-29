import os
from PIL import Image
import shutil
import glob
import numpy as np


def get_broadth_ratio(mask):
    mask[mask>0] = 1
    accumulated_row_mask = np.sum(mask, axis=0)
    accumulated_row_mask[accumulated_row_mask>0] = 1
    broadth_ratio = np.sum(accumulated_row_mask)/accumulated_row_mask.shape[0]
    return broadth_ratio

img_list = glob.glob('/home/ubuntu/workspace/zero123/3drec/data/real_images/*.png')
save_dir = '/home/ubuntu/workspace/zero123/3drec/data/real_images_zero123'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

rendering_example = Image.open('/home/ubuntu/workspace/zero123/3drec/data/sofa_zero123_rendering/B07B4MMV3N/000.png')
rendering_example = np.array(rendering_example)[:,:,3]
print(rendering_example.shape)
target_ratio = get_broadth_ratio(rendering_example)

for i, file_name in enumerate(img_list):
    print(file_name)
    img = Image.open(file_name)
    height, width = img.size
    print(img.size)
    mask = np.array(img)[:,:,3]
    real_image_ratio = get_broadth_ratio(mask)
    print("real image ratio:", real_image_ratio)
    size = max(height, width)
    print(img.size)
    scale_ratio = real_image_ratio*size/512/target_ratio

    new_size = int(scale_ratio*512)
    print(new_size)
    new_im = Image.new('RGBA', (new_size, new_size), (0,0,0,0))
    new_im.paste(img, (int((new_size - height) / 2), int((new_size - width) / 2)))

    img = img.resize((512, 512))

    new_im.save(os.path.join(save_dir, "%03d.png"%i))

    # Image.fromarray(np.array(new_im)[:,:,3]).convert("1").save(os.path.join(save_dir, "%03d_mask.png"%i))
    # break
