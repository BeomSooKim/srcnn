#%%
import cv2
from IPython.display import display
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
#%%
model = load_model("D:\\python\\models\\srcnn\\epoch_050.hdf5")
# test images
img_high = cv2.imread("D:\\python\\dataset\\horse2zebra\\testB\\n02391049_10980.jpg")
#%%
img_low = cv2.cvtColor(img_high, cv2.COLOR_BGR2YCrCb)
shape = test_image.shape
Y_img = cv2.resize(img_low[:,:,0], (shape[0] // 2, shape[1] // 2), cv2.INTER_CUBIC)
Y_img = cv2.resize(Y_img, (shape[0], shape[1]), cv2.INTER_CUBIC)
img_low[:,:,0] = Y_img
img_low = cv2.cvtColor(img_low, cv2.COLOR_YCrCb2BGR)
highlow = np.hstack([img_high, img_low])
plt.imshow(highlow)
plt.show()

Y = np.zeros((1, Y_img.shape[0], Y_img.shape[1], 1), dtype = float)
Y[0,:,:,0] = Y_img / 255.0
pred = model.predict(Y)

pred = np.clip(pred*255.0, 0, 255).astype(np.uint8)[0,:,:,0]

img_pred = cv2.cvtColor(img_high, cv2.COLOR_BGR2YCrCb)
img_pred[6:-6, 6:-6, 0] = pred
img_pred = cv2.cvtColor(img_pred, cv2.COLOR_YCrCb2BGR)
plt.imshow(img_pred)
plt.show()