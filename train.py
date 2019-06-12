#%%
from keras.layers import Conv2D, Activation, Input
from keras.initializers import RandomNormal, Constant
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
import keras.backend as K
import h5py
import numpy as np

#%%
epochs = 50
base_lr = 0.001
momentum = 0.9
batchSize = 512
save_period = 5
decay_period = 20
onRGB = True

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
    out = Conv2D(filters = 3 if onRGB else 1, kernel_size = 5, strides = 1, 
                kernel_initializer = RandomNormal(0, 0.001), 
                bias_initializer = Constant(0))(x)
    x = Activation('linear')(x)
    return Model(inp, out)

model = srcnn((None, None, 3 if onRGB else 1))
model.summary()
model.compile(SGD(base_lr, momentum), loss = 'mse')

dataset = h5py.File('D:\\python\\dataset\\horse2zebra\\srcnn_train_rgb.hdf5')
nImages = len(dataset['images'])

index = np.arange(nImages)
losses = []
stepsize = nImages // batchSize
for ep in range(1, epochs+1):
    loss_for_epoch = []
    np.random.shuffle(index)    
    for step in range(stepsize):
        idx = index[step*batchSize:(step+1)*batchSize]
        images = dataset['images'][sorted(list(idx))]
        labels = dataset['labels'][sorted(list(idx))]

        images = images / 255.
        labels = labels / 255.
        if not onRGB:
            loss = model.train_on_batch(images[:,:,:,np.newaxis], labels[:,:,:,np.newaxis])
        else:
            loss = model.train_on_batch(images, labels)
        print("{} of {} iter loss : {:.6f}".format(step, stepsize, loss), end = '\r')
        loss_for_epoch.append(loss)
    loss_epoch = np.array(loss_for_epoch).mean()
    losses.append(loss_epoch)
    print('{} of {} epoch loss : {:.6f}'.format(ep, epochs, loss_epoch))
    if ep % save_period == 0:
        model.save('D:\\python\\models\\srcnn\\epoch_{:03d}.hdf5'.format(ep))
    #if ep+1 % decay_period == 0:
     #   K.set_value(model.optimizer.lr, model.optimizer.lr * 0.3)
dataset.close()
#%%
