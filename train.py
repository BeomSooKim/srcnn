#%%

from keras.optimizers import SGD, Adam
import keras.backend as K
from keras.models import load_model
import h5py
import numpy as np
from utils import srcnn

#%%
start_epoch = 140
epochs = 20
base_lr = 0.000005
momentum = 0.9
batchSize = 512
save_period = 5
patience = 10
decay_factor = 0.5
epsilon = 1e-4

#model = srcnn((None, None, 1))
model = load_model('D:/dl/models/srcnn/epoch_120.hdf5')
model.summary()
model.compile(Adam(base_lr, momentum), loss = 'mse')

dataset = h5py.File('D:\\dataset\\horse2zebra\\srcnn_train_Y.hdf5')
nImages = len(dataset['images'])

index = np.arange(nImages)
losses = []
stepsize = nImages // batchSize
stay_count = 0
minloss = 1.0

for ep in range(1, epochs+1):
    loss_for_epoch = []
    np.random.shuffle(index)    
    for step in range(stepsize):
        idx = index[step*batchSize:(step+1)*batchSize]
        images = dataset['images'][sorted(list(idx))]
        labels = dataset['labels'][sorted(list(idx))]

        images = images / 127.5 - 1
        labels = labels / 127.5 - 1
        
        loss = model.train_on_batch(images[:,:,:,np.newaxis], labels[:,:,:,np.newaxis])
        
        print("{} of {} iter loss : {:.6f}".format(step, stepsize, loss), end = '\r')
        loss_for_epoch.append(loss)
    loss_epoch = np.array(loss_for_epoch).mean()
    losses.append(loss_epoch)
    print('{} of {} epoch loss : {:.6f}'.format(ep, epochs, loss_epoch))
    if ep % save_period == 0:
        model.save('D:\\dl\\models\\srcnn\\epoch_{:03d}.hdf5'.format(ep+start_epoch))
    #if ep % decay_period == 0:
    #   K.set_value(model.optimizer.lr, model.optimizer.lr * decay_factor)
    #if minloss - loss_epoch < epsilon:
    #    stay_count +=1
    #    if stay_count == patience:
    #        print('decrease learning rate from {:.4f} to {:.4f}'.format(K.eval(model.optimizer.lr), K.eval(model.optimizer.lr)*decay_factor))
    #        K.set_value(model.optimizer.lr, K.eval(model.optimizer.lr) * decay_factor)
    #        stay_count = 0
    #else:
    #    stay_count = 0
    #if minloss >= loss_epoch:
    #        minloss = loss_epoch
    #print('stay count {} / min loss : {:.4f} / loss epoch : {:.4f}'.format(stay_count, minloss, loss_epoch))    
dataset.close()