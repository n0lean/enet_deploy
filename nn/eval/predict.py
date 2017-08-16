from __future__ import absolute_import

from skimage import io
from model.enet import *
import numpy as np
from skimage import measure
from skimage import transform
from utils.label_transform import self_labeled

def predict(img_path, weight_path, in_shape=(256, 512, 3), class_num=5, is_cityscapes=False):
    model = ENet(in_shape, class_num)
    model.model.load_weights(weight_path)
    # model.model.summary()
    img = io.imread(img_path)
    io.imsave('ori.jpg', img)
    if is_cityscapes:
        img = measure.block_reduce(img, (4,4,1))
    else:
        img = measure.block_reduce(img, (2,2,1))
    img1 = img.copy()
    img1 = np.array([img1])

    res = model.model.predict(img1, 1, 1)
    return res, img


def vis(flatten, w, h):
    # for vis the difference
    img = np.argmax(flatten, axis=-1)
    img = np.reshape(img, (w, h))
    img2 = np.zeros((w, h,3))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] == 1:
                img2[i,j] = (255, 0, 0)
            elif img[i, j] == 0:
                img2[i, j] = (0,255,0)
            elif img[i, j] == 2:
                img2[i, j] = (0,0,255)
            elif img[i, j] == 3:
                img2[i, j] = (125,125,0)
            elif img[i, j] == 4:
                img2[i, j] = (0,0,0)
    img2 = np.array(img2, dtype=np.uint8)
    return img2


if __name__ == '__main__':
    #img_path = '../data/cityscapes/img/train/darmstadt/darmstadt_000000_000019_leftImg8bit.png'
    # 210 is good
    num = '250'
    img_path = '../data/self_labeled/img/train/0000' + num + '.jpg'
    weight_path = '../final.h5'
    label_path = '../data/self_labeled/labels/train/0000' + num + '_annotated_image.png'
    labels = io.imread(label_path)
    label_trans = self_labeled()
    res, img = predict(img_path, weight_path, (256, 512, 3), is_cityscapes=False)
    res = vis(res, 256, 512)
    # res = color.gray2rgb(res)
    # res = res.reshape((res.shape[0],res.shape[1],1))
    # res = np.concatenate((res,res,res),axis=-1)
    res = res * 0.6 + img * 0.3
    res = np.array(res, dtype=np.uint8)
    io.imsave('res.jpg', res)
    #io.imsave('ori.jpg', img)