# 2020/02/29 by ausk

import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def trainGenerator(batch_size,train_dpath,image_dname,mask_dname,aug_dict,
                    image_color_mode = "grayscale", mask_color_mode = "grayscale",
                    image_save_prefix  = "image",mask_save_prefix  = "mask",
                    save_to_dir = None,target_size = (256,256),seed = 1):

    image_generator = ImageDataGenerator(**aug_dict).flow_from_directory(
        train_dpath,
        classes     = [image_dname],
        class_mode  = None,
        color_mode  = image_color_mode,
        target_size = target_size,
        batch_size  = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = image_save_prefix,
        seed = seed)

    mask_generator = ImageDataGenerator(**aug_dict).flow_from_directory(
        train_dpath,
        classes     = [mask_dname],
        class_mode  = None,
        color_mode  = mask_color_mode,
        target_size = target_size,
        batch_size  = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = mask_save_prefix,
        seed = seed)

    def preprocess(img,mask):
        if(np.max(img) > 1):
            img = img / 255
            mask = np.where(mask >127.5, 1.0, 0.0)
        return (img,mask)

    for (img,mask) in zip(image_generator, mask_generator):
        img,mask = preprocess(img,mask)
        yield (img,mask)


#读图，灰度化，归一化，缩放，改变维度 [1, nh, nw, 1]
def testGenerator(test_dpath, num_image = 30, input_shape = (256,256, 1)):
    for i in range(num_image):
        fpath = "{}/{}.png".format(test_dpath, i)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        nh, nw = img.shape[:2]
        if img.shape[0]>input_shape[0] or img.shape[1] > input_shape[1]:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
        img = cv2.resize(img, (input_shape[1], input_shape[0]), interpolation = interpolation)
        img = np.float32(img)/255
        img = img[None, ..., None]
        yield img

# 保存结果
def saveResult(dpath, npyfile,num_class = 2, postfix=""):
    if not postfix:
        postfix = "_res"
    for i,item in enumerate(npyfile):
        img = np.clip(item[:,:,0]*255, 0, 255)
        fname = "{}{}.png".format(i, postfix)
        cv2.imwrite(os.path.join(dpath, fname),img)
