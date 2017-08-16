from __future__ import absolute_import

from nn.model.enet import *
from nn.utils.line_generate import vis_line, generate_line
from skimage import io
from skimage import transform
import numpy as np
import time
import os
import csv
from shutil import copy


def nn_init(weight_path, shape=(256,512,1), classes=2):
    enet = ENet(shape, classes=classes)
    enet.model.load_weights(weight_path)
    enet.model.predict_on_batch()
    return enet


def nn_predict(img, enet):
    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
    img0 = img.copy()
    img = np.array([img])
    res = enet.model.predict_on_batch(img)
    res = np.argmax(res, axis=-1)
    line = generate_line(res, False)
    img0 = vis_line(img0, line, True)
    img0 = np.reshape(img0, (img.shape[0], img.shape[1]))
    img0 = np.array(img0, dtype=np.uint8)

    return line, img0


if __name__ == '__main__':
    weight_path = ''

    im_path = './workspace/input/cfl.jpg'
    lock1_path = './workspace/input/lock1.txt'
    lock2_path = './workspace/output/lock2.txt'

    # Model Init
    enet = nn_init(weight_path)

    # Main Loop
    while 1:
        print('Inside Main While Loop')
        if os.path.isfile(im_path) & (not os.path.isfile(lock1_path)):
            open(lock1_path, 'w').close()
            start_time = time.time()

            # Load Images
            img = io.imread(im_path, as_grey=True)
            if img.shape != (256,512):
                img = transform.resize(img, (256, 512))

            os.rename(im_path, './workspace/output/img.jpg')
            os.remove(lock1_path)

            # print load time
            print('Load time: {}'.format(time.time()-start_time))

            # Run Model
            start_time = time.time()
            line, res = nn_predict(img, enet)
            print('Model time: {}'.format(time.time()-start_time))

            # Save Result
            start_time = time.time()
            if not os.path.isfile(lock2_path):
                open(lock2_path, 'w').close()
                with open('./workspace/output/vector.csv', 'w') as f:
                    writer = csv.writer(f, delimier=',')
                    writer.writerow(line)
                copy('./workspace/output/vector.csv', './workspace/output/vector2.cs')
                os.remove(lock2_path)

            # visualize
            io.imsave('workspace/output/vector.jpg', res)


