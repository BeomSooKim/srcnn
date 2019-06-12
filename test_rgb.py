#%%
import cv2
from IPython.display import display
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
#%%
model = load_model("D:\\python\\models\\srcnn\\epoch_050.hdf5")
# test images
img_high = cv2.imread("D:\\python\\dataset\\horse2zebra\\testB\\n02391049_1630.jpg")
#%%
shape = img_high.shape
img_low = cv2.resize(img_high, (shape[0] // 2, shape[1] // 2), cv2.INTER_CUBIC)
img_low = cv2.resize(img_low, (shape[0], shape[1]), cv2.INTER_CUBIC)

highlow = np.hstack([img_high, img_low])
plt.imshow(highlow[:,:,::-1])
plt.show()
Y = np.zeros((1, shape[0], shape[1], 3), dtype = float)
Y[0,:,:,:] = img_low / 255.0
pred = model.predict(Y)

pred = np.clip(pred*255.0, 0, 255).astype(np.uint8)[0,:,:,:]
#%%
img_sr = img_low.copy()
img_sr[6:-6, 6:-6, :] = pred
highlow = np.hstack([highlow, img_sr])
fig = plt.figure(figsize = (10, 30))
plt.imshow(highlow[:,:,::-1])