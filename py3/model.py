# 2020/02/29 by ausk

import utils
import warnings
warnings.filterwarnings("ignore")

import os, sys, time
import numpy as np
import cv2

from keras import layers, models
from keras import backend as K
import tensorflow as tf

# 处理块的名字
def handle_block_names(prefix, stage):
    name = "{}_{}".format(prefix, stage)
    conv_name = '{}_conv'.format(name)
    bn_name   = '{}_bn'.format(name)
    relu_name = '{}_relu'.format(name)
    return conv_name, bn_name, relu_name

# 默认的 ConvReLU6
def ConvReLU( nfilter, kernel_size, prefix="block", stage=1, use_bn=False):
    conv_name, bn_name, relu_name = handle_block_names(prefix, stage)
    def layer(x):
        conv_name, bn_name, relu_name = handle_block_names(prefix, stage)
        x = layers.Conv2D( nfilter, kernel_size, padding="same", kernel_initializer = 'glorot_uniform', activation = None, name=conv_name, use_bias=not(use_bn))(x)
        if use_bn:
            x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=bn_name)(x)
        #x = layers.Activation('relu', name=relu_name)(x)
        x = layers.ReLU(6.0, name=relu_name)(x)
        return x
    return layer

# 编码模块
def EncoderBlock( nfilter, stage, kernel_size=(3,3), use_bn=False, use_drop=False, use_pool = True):
    prefix = "encoder"
    def layer(x):
        if use_pool:
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = ConvReLU( nfilter, kernel_size, use_bn=use_bn, prefix=prefix, stage = str(stage)+"1")(x)
        #x = ConvReLU( nfilter, kernel_size, use_bn=use_bn, prefix=prefix, stage = str(stage)+"2")(x)
        if use_drop:
            x = layers.Dropout(0.5)(x)
        return x
    return layer

# 解码模块
def DecoderBlock( nfilter, stage, kernel_size=(3,3), use_bn=False, use_up=True, skip=None):
    prefix = "decoder"
    def layer(x):
        nonlocal skip
        if use_up:
            x = layers.UpSampling2D(size=(2,2), name="{}_{}_up".format(prefix, str(stage)))(x)
            #x = ConvReLU( nfilter, kernel_size, use_bn=use_bn, prefix=prefix, stage = str(stage)+"0")(x)
        if skip is not None:
            x = layers.Concatenate()([x, skip])
        x = ConvReLU( nfilter, kernel_size, use_bn=use_bn, prefix=prefix, stage = str(stage)+"1")(x)
        #x = ConvReLU( nfilter, kernel_size, use_bn=use_bn, prefix=prefix, stage = str(stage)+"2")(x)
        return x
    return layer

# 搭建 UNet
def myunet(input_size=(256, 256, 1), nfilter0=64,ratio=1.2):
    inputs = layers.Input(input_size)
    stages = 5
    nfilters = [int(nfilter0 * (ratio**i) + 0.5) for i in range(stages)]
    feats = []
    x = inputs
    for stage in range(1, stages+1):
        nfilter = nfilters[stage-1]
        use_drop = stage > 4
        use_pool = stage > 1
        use_bn =  True #stage > 1
        x = EncoderBlock( nfilter, stage, kernel_size=(3,3), use_bn=use_bn, use_drop=use_drop, use_pool=use_pool)(x)
        feats.append(x)

    for stage in range(stages, 0, -1):
        nfilter = nfilters[stage-1]
        use_drop = False
        use_bn = True #stage > 1
        use_up = stage > 1
        skip = feats[stage-2] if stage>2 else None
        x = DecoderBlock( nfilter, stage, kernel_size=(3,3), use_bn=use_bn, use_up=use_up, skip=skip)(x)

    x = ConvReLU(nfilter0, kernel_size=(3,3), prefix="final", stage=1, use_bn=True)(x)
    x = layers.Conv2D(1, 1, activation = 'sigmoid', name="final")(x)
    model = models.Model(inputs = inputs, outputs = x)
    model.summary()
    return model

# UNet in H5
class UNetH5:
    def __init__(self, h5fpath, input_shape):
        assert len(input_shape)==2 or input_shape[2] == 1
        self.input_shape = input_shape
        self.input_size  = (int(input_shape[1]), int(input_shape[0]))
        tf.keras.backend.set_learning_phase(0)
        K.set_learning_phase(0)
        self.model = models.load_model(h5fpath, custom_objects={})

    def predict_generator(self, test_gen, batch_size, verbose=0):
        return self.model.predict_generator(test_gen, batch_size, verbose=verbose)

    def predict(self, img):
        #assert img.shape == input_shape, "Make sure you have input {} ndarray"
        nh, nw = img.shape[:2]
        if img.ndim == 3 and img.shape[-1] !=1 :
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.shape[:2] != self.input_shape[:2]:
            img = cv2.resize(img, self.input_size)
        ximg = (np.float32(img)/255)[None, ..., None]

        xout = self.model.predict(ximg)[0] #, ..., 0]
        xout = np.clip(xout * 255, 0, 255).astype(np.uint8)
        return xout

# UNet in PB
class UNetPB:
    def __init__(self, pbfpath, input_shape):
        self.input_shape = input_shape
        self.pbfpath     = pbfpath
        self.graph       = utils.loadGraph(pbfpath)
        self.input_node  = self.graph.get_tensor_by_name("input_1:0")
        self.output_node = self.graph.get_tensor_by_name("output_0:0")
        self.session = tf.Session(graph=self.graph)

    def predict(self, img):
        nh,nw = self.input_shape[:2]
        img = cv2.resize(img, (nw, nh))

        if img.ndim ==2:
            ximg = img[None, ..., None]
        else:
            ximg = img[None, ...]
        ximg = np.float32(ximg)/255.0

        xout = self.session.run(self.output_node, feed_dict={self.input_node: ximg})[0]
        xout = np.clip(xout * 255, 0, 255).astype(np.uint8)
        return xout