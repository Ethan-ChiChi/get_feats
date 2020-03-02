# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:51:35 2020

@author: Dylan
"""
import numpy as np
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
#import skimage
#import skimage.io
#import skimage.transform
import vgg16
import os
#import matplotlib.pyplot as plt
from scipy.linalg import norm
import time
#import matplotlib.image as mpimg
import cv2

image_path = './test_data'
#image_lists = os.listdir(image_path)

def load_image(path):
    # load image
#    img = skimage.io.imread(path)
    img = cv2.imread(path)
#    print(img)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    #图片长宽最小值
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224, 3
    resized_img = cv2.resize(crop_img, (224,224))
#    resized_img = skimage.transform.resize(crop_img, (224, 224))
    
    return resized_img


def get_feats(image_dict):
    vgg16_feats = np.zeros((len(image_dict), 4096))
    with tf.Session() as sess:
        images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
        vgg = vgg16.Vgg16()
        vgg.build(images)
        for i in range(len(image_dict)):
            print(i)
            img_list = load_image(image_dict[i])
            batch = img_list.reshape((1, 224, 224, 3))
            print(batch.shape)
            feature = sess.run(vgg.fc7, feed_dict={images: batch})#提取fc7层的特征
            print(feature.shape)
            feature = np.reshape(feature, [4096])
            feature = feature.reshape(-1)
            print(feature.shape)
            feature /= norm(feature) # 特征归一化
 #           print(feature)
            vgg16_feats[i, :] = feature #每张图片的特征向量为1行
 #           print(vgg16_feats.shape)
#            print(len(vgg16_feats))
#            print(len(vgg16_feats[0]))
#    vgg16_feats = np.save('X:/github/vgg-tensorflow/vgg16_feats', vgg16_feats)
    return vgg16_feats

def write_feats(feats, image_dict):
    f = open('./vgg16_feats.txt', 'w')
    for i in range(len(feats)):
        result = []
        for j in range(len(feats[0])):
            result.append(str(feats[i][j]))
        id_index = get_id_index(image_dict)
        f.write(str(id_index[i]) +'\t' +','.join(result) + '\n')
    f.close

def get_id_index(image_dict):        
    id_index = list(image_dict.keys())
    return id_index

def get_image_dict(path):
    unique_image = {}
    index = 0
    image_lists = os.listdir(path)
#    print(image_lists)
    for image_file in image_lists:
        unique_image[index] = os.path.join(path, image_file)
        index += 1
    return unique_image

if __name__ == '__main__':
    
    start_time = time.time()

    image_dict = get_image_dict(image_path)
    vgg16_feats = get_feats(image_dict)
    write_feats(vgg16_feats,image_dict)
    
    print(("features_got: %ds" % (time.time() - start_time)))
####    
#    
#image_data =get_data(image_path)
#
#
#data_index = batch_size
#batch = image_data[:batch_size]
#data_index += batch_size
