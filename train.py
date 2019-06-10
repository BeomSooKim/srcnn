#%%
from keras.layers import Conv2D, Activation, Input
from keras.initializers import RandomNormal, Constant
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
import keras.backend as K
import h5py
import numpy as np

#%%
epochs = 100
base_lr = 0.001
momentum = 0.9
batchSize = 256
save_period = 5

def srcnn(input_shape):
    inp = Input(input_shape)
    x = Conv2D(filters = 64, kernel_size = 9, strides = 1, 
                kernel_initializer = RandomNormal(0, 0.001), 
                bias_initializer = Constant(0))(inp)
    x = Activation('relu')(x)
    x = Conv2D(filters = 32, kernel_size = 1, strides = 1, 
                kernel_initializer = RandomNormal(0, 0.001), 
                bias_initializer = Constant(0))(x)
    x = Activation('relu')(x)
    out = Conv2D(filters = 3, kernel_size = 5, strides = 1, 
                kernel_initializer = RandomNormal(0, 0.001), 
                bias_initializer = Constant(0))(x)
    x = Activation('linear')(x)
    return Model(inp, out)

model = srcnn((None, None, 3))
model.summary()
model.compile(SGD(base_lr, momentum, 0.8), loss = 'mse')

dataset = h5py.File('D:\\dataset\\horse2zebra\\srcnn_train.hdf5')
nImages = len(dataset['images'])

index = np.arange(nImages)
losses = []
stepsize = nImages // batchSize
for ep in range(epochs+1):
    loss_for_epoch = []
    np.random.shuffle(index)    
    for step in range(stepsize):
        idx = index[step*batchSize:(step+1)*batchSize]
        images = dataset['images'][sorted(list(idx))]
        labels = dataset['labels'][sorted(list(idx))]

        images = images / 255.
        labels = labels / 255.

        loss = model.train_on_batch(images, labels)
        print("{} of {} iter loss : {:.6f}".format(step, stepsize, loss), end = '\r')
        loss_for_epoch.append(loss)
    loss_epoch = np.array(loss_for_epoch).mean()
    losses.append(loss_epoch)
    print('{} of {} epoch loss : {:.6f}'.format(ep+1, epochs, loss_epoch))
    if ep+1 % save_period == 0:
        model.save('D:\\dl\\models\\srcnn\\epoch_{:03d}.hdf5'.format(ep+1))
    if ep+1 % 20 == 0:
        K.set_value(model.optimizer.lr, model.optimizer.lr * 0.3)
dataset.close()
#%%
from PIL import Image
from IPython.display import display
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
