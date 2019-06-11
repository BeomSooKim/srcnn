#%%
from PIL import Image
from IPython.display import display
from keras.models import load_model
import numpy as np 
# test images
test_image =  Image.open("D:\\python\\dataset\\horse2zebra\\testB\\n02391049_100.jpg")
test_image_sm = test_image.resize((64, 64))
img_bicubic = test_image_sm.resize((256, 256),Image.BICUBIC)
display(img_bicubic)
#%%
trained = load_model('D:\\python\\models\\srcnn\\epoch_050.hdf5')
#%%
bicubic_arr = np.array(img_bicubic)
#%%
arr = np.array(img_bicubic) / 255.0
dd = trained.predict(arr[np.newaxis, :,:,:])[0]
ddd = np.clip(dd * 255.0, 0, 255).astype(np.uint8)
Image.fromarray(ddd)