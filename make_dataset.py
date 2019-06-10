#%%
import os, glob
from PIL import Image
import numpy as np
import h5py
#%%
def conv_output(inp, filters):
    out = inp
    for f in filters:
        out = out - f + 1
    return out

scale = 4
data_path = 'D:\\python\\dataset\\horse2zebra\\trainB\\*.jpg'
save_path = 'D:\\python\\dataset\\horse2zebra\\srcnn_train.hdf5'
imgSize = 33
filter_list = [9, 1, 5] # convnet filter size sequence
stride = 14
labelSize = conv_output(imgSize, filter_list)
padding = (imgSize - labelSize) // 2

img_list = glob.glob(data_path)
print('total train data : {}'.format(len(img_list)))
#img_list = np.random.choice(img_list, size = 50, replace = False)
#%%
n_images = int(((256 - 33 + 1) / 14) ** 2 * len(img_list))
dataset = h5py.File(save_path)
dataset.create_dataset('images', shape = (n_images, imgSize, imgSize, 3))
dataset.create_dataset('labels', shape = (n_images, labelSize, labelSize, 3))

#%%
# subimage generate
index = 0
for i, img in enumerate(img_list):
    img_high = Image.open(img)
    if img_high.mode == 'L':
        img_high = img_high.convert('RGB')
    size = img_high.size[0]
    img_low = img_high.resize((size//scale, size//scale), Image.BICUBIC)
    img_low = img_low.resize((size, size), Image.BICUBIC)

    for r in range(0, size-imgSize, stride):
        for c in range(0, size-imgSize, stride):
            crop_image = np.array(img_low)[r:r+imgSize, c:c+imgSize:]
            dataset['images'][index] = crop_image
            crop_label = np.array(img_high)[r+padding:r+padding+labelSize, c+padding:c+padding+labelSize]
            dataset['labels'][index] = crop_label
            index += 1
    print("{} of {} : {}".format(i, len(img_list), img), end = '\r')

dataset.close()
