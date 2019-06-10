#%%
from PIL import Image
from IPython.display import display
from keras.models import load_model
import numpy as np 
# test images
test_image =  Image.open("D:\\dataset\\horse2zebra\\testB\\n02391049_100.jpg")
test_image_sm = test_image.resize((128, 128))
img_bicubic = test_image_sm.resize((256, 256),Image.BICUBIC)
display(img_bicubic)
#%%
trained = load_model('D:\\dl\\models\\srcnn\\epoch_005.hdf5')
#%%
bicubic_arr = np.array(img_bicubic)
pred_img = np.zeros(bicubic_arr.shape)
for r in range(0, 256-33+1, 21):
    for c in range(0, 256-33+1, 21):
        sub_img = bicubic_arr[r:r+33, c:c+33]

        pred = trained.predict(sub_img[np.newaxis,:,:,:] / 255.0)
        pred_img[r+6:r+6+21, c+6:c+6+21, :] = pred

#%%
pred_img = np.clip(pred_img * 255, 0, 255).astype(np.uint8)
Image.fromarray(pred_img)
#%%