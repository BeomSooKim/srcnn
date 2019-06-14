#%%
import os, glob
import cv2
import numpy as np
import h5py
#%%
def conv_output(inp, filters):
    out = inp
    for f in filters:
        out = out - f + 1
    return out

onRGB = True
scale = 3
data_path = 'D:\\dataset\\horse2zebra\\trainB\\*.jpg'
save_path = 'D:\\dataset\\horse2zebra\\srcnn_train_{}.hdf5'
save_path = save_path.format('Y')
size = 256

imgSize = 33
filter_list = [9, 1, 5] # convnet filter size sequence
stride = 14
labelSize = conv_output(imgSize, filter_list)
padding = (imgSize - labelSize) // 2

img_list = glob.glob(data_path)
print('total train data : {}'.format(len(img_list)))
#img_list = np.random.choice(img_list, size = 50, replace = False)
#%%
n_images = int(((256 - imgSize + 1) / stride) ** 2 * len(img_list))
dataset = h5py.File(save_path)

dataset.create_dataset('images', shape = (n_images, imgSize, imgSize))
dataset.create_dataset('labels', shape = (n_images, labelSize, labelSize))

#%%
# subimage generate
index = 0
for i, img in enumerate(img_list):
    img_high = cv2.imread(img)

    img_high = cv2.cvtColor(img_high, cv2.COLOR_BGR2YCrCb)
    img_high = img_high[:,:,0]
    img_low = cv2.resize(img_high, (size // scale, size // scale), cv2.INTER_CUBIC)
    img_low = cv2.resize(img_low, (size, size), cv2.INTER_CUBIC)
    size = img_high.shape[0]

    for r in range(0, size-imgSize, stride):
        for c in range(0, size-imgSize, stride):
            crop_image = np.array(img_low)[r:r+imgSize, c:c+imgSize]
            dataset['images'][index] = crop_image
            crop_label = np.array(img_high)[r+padding:r+padding+labelSize, c+padding:c+padding+labelSize]
            dataset['labels'][index] = crop_label
            index += 1
    print("{} of {} : {}".format(i+1, len(img_list), img), end = '\r')

dataset.close()
#%%

