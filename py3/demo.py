# 2020/02/29 by ausk

import warnings
warnings.filterwarnings("ignore")

import os, sys
import argparse

import numpy as np
import cv2

import tensorflow as tf
from keras import backend as K
#from keras.losses import *
from keras.optimizers import Adam
import keras.models
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from data import trainGenerator,testGenerator, saveResult
from model import myunet, UNetH5, UNetPB
import utils


class cfg:
    losses = {
        #"dice": dice_coef_loss,
        "bce": "binary_crossentropy",
    }
    loss_name = "bce"

    custom_objects = {} # {"dice_coef_loss": dice_coef_loss}
    train_dpath = "dataset/membrane/train/"
    test_dpath  = "dataset/membrane/test/"

    model_dpath = "models/{}".format(loss_name)
    pretrained_fpath = "{}/myunet_final.h5".format(model_dpath)
    mfpath = model_dpath + '/ep{epoch:03d}-loss{loss:.3f}-acc{accuracy:.3f}.h5'

    input_size = (256,256,1)
    epochs = 20
    tag = "myunet"
    prefix = "{}_".format(tag) if tag else ""
    postfix = "_{}".format(tag) if tag else ""

    @classmethod
    def setparam(cls, tag, loss_name, lr=1E-4):
        cls.tag=tag
        cls.prefix = "{}_".format(tag) if tag else ""
        cls.postfix = "_{}".format(tag) if tag else ""
        cls.loss_name = loss_name
        cls.loss = cls.losses.get(loss_name)
        cls.model_dpath = "models/{}{}".format(cls.prefix, loss_name)
        utils.mksured(cls.model_dpath)
        cls.pretrained_fpath = "{}/myunet_final.h5".format(cls.model_dpath)
        cls.mfpath = cls.model_dpath + '/ep{epoch:03d}-loss{loss:.3f}-acc{accuracy:.3f}.h5'
        cls.lr=lr

    @classmethod
    def get_lastepoch(cls):
        try:
            fpaths = glob.glob(cls.model_dpath+"/ep*.h5")
            print(fpaths)
            last_epoch = sorted(int(fpath.split("/")[-1][2:5]) for fpath in fpaths)[-1]
        except:
            last_epoch = 0
        return last_epoch


# 训练 UNet
def train_unet(cfg, resume=True):
    input_size = cfg.input_size
    pretrained_fpath = cfg.pretrained_fpath
    model_dpath = cfg.model_dpath
    loss = cfg.loss
    train_dpath = cfg.train_dpath
    lr = cfg.lr
    epochs = cfg.epochs
    mfpath = cfg.mfpath #model_dpath + '/ep{epoch:03d}-loss{loss:.3f}-acc{accuracy:.3f}.h5'
    last_epoch = cfg.get_lastepoch()
    print(last_epoch)

    # (1) 创建训练生成器
    augdict = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
    train_gen = trainGenerator(4, train_dpath,'image','label',augdict, save_to_dir = None)

    if False:
        for i,batch in enumerate(train_gen):
            if i >= 3:
                break

    # (2) 编译和训练模型, (可选地加载预训练模型，继续训练）
    model = myunet(input_size)
    model.compile(optimizer = Adam(lr =lr), loss = loss, metrics = ['accuracy'])
    #model.summary()

    # 加载预训练模型
    if resume and (pretrained_fpath) and os.path.exists(pretrained_fpath):
        try:
            model.load_weights(pretrained_fpath)
        except Exception as exc:
            print("Exception: {}".format(exc))

    # 训练模型
    model_checkpoint = ModelCheckpoint(mfpath, monitor='loss', verbose=1, save_best_only=True)
    model_tb = keras.callbacks.TensorBoard(log_dir=model_dpath, histogram_freq=0, write_graph=False, write_images=False)
    model.fit_generator(train_gen, steps_per_epoch=1000, epochs=epochs, callbacks=[model_checkpoint, model_tb], initial_epoch=last_epoch)
    model.save(pretrained_fpath)

# 测试 UNet
def test_unet( cfg ):
    pretrained_fpath = cfg.pretrained_fpath
    test_dpath = cfg.test_dpath
    postfix = cfg.postfix

    tf.keras.backend.set_learning_phase(0)
    K.set_learning_phase(0)

    # 从文件中加载模型
    print("Loading: {}".format(pretrained_fpath))
    #model = keras.models.load_model(pretrained_fpath, custom_objects=custom_objects)
    model = UNetH5(pretrained_fpath, (256, 256, 1))

    # 创建数据生成器，预测并保存结果
    test_gen = testGenerator(test_dpath)
    results = model.predict_generator(test_gen,30,verbose=1)
    saveResult(test_dpath,results, postfix=postfix)


# 保存为 PB 文件
def run_savepb(h5fpath, pbfpath=None):
    if pbfpath is None:
        pbfpath = h5fpath[:-2] + "pb"
    #K.clear_session()
    #K.set_session(tf.Session())
    print(h5fpath)
    model = keras.models.load_model(h5fpath)
    #model.load_weights(h5fpath, by_name=True)
    utils.savepb(model, pbfpath)

def test_loadpb(pbfpath):
    unet = UNetPB(pbfpath, (256, 256, 1))
    for i in range(30):
        fpath = "dataset/membrane/test/{}.png".format(i)
        fpath2 = "dataset/membrane/test/{}_pb.png".format(i)
        img = cv2.imread(fpath)
        print(fpath, img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        with utils.timeit():
            result = unet.predict(gray)

        print(result.min(), result.max(), result.shape)
        cv2.imwrite(fpath2, result)

def get_args():
    #tag = "myunet"
    parser = argparse.ArgumentParser(description="Train/Test Unet-like CNN")
    parser.add_argument("--devices", "-d", default="0", type=str)
    parser.add_argument("--op", default="train", type=str)

    args = parser.parse_args(sys.argv[1:])
    assert args.op in ("train", "test", "savepb", "testpb")
    return args


if __name__ == "__main__":
    args = get_args()
    devices = args.devices
    op = args.op

    # (0) 设置参数
    #assert tag in ("myunet", )
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    cfg.setparam(tag="myunet",loss_name="bce", lr=1E-4)

    # (1) 训练
    if op == "train":
        train_unet(cfg)

    # (2) 测试
    if op == "test":
        cfg.pretrained_fpath = "models/myunet_64_1.2.h5"
        test_unet(cfg)

    # (3) 转 pb 模型
    if op == "savepb":
        h5fpath = cfg.pretrained_fpath
        h5fpath = "models/myunet_64_1.2.h5"
        pbfpath = "models/myunet_64_1.2.pb"
        run_savepb(h5fpath)

    # (4) 测试 pb 模型
    if op == "testpb":
        pbfpath = "models/myunet_64_1.2.pb"
        test_loadpb(pbfpath)
