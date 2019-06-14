from keras.layers import Conv2D, Activation, Input
from keras.initializers import RandomNormal, Constant
from keras.models import Model, load_model

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
    out = Conv2D(filters = 1, kernel_size = 5, strides = 1, 
                kernel_initializer = RandomNormal(0, 0.001), 
                bias_initializer = Constant(0))(x)
    x = Activation('linear')(x)
    
    return Model(inp, out)