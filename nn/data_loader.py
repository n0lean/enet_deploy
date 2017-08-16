from __future__ import absolute_import
from skimage import io
from skimage import measure
import glob
import os
import random
import utils.label_transform
import numpy as np
from skimage import transform
from utils import line_generate

class Dataset(object):
    def __init__(self, data_dir, label_dir, train=True, make_random=True, val_ratio=0.3, is_cityscape=False, is_gray=False):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.is_cityscape = is_cityscape
        self.is_gray = is_gray
        self.addr = self.build()
        self.label_trans = utils.label_transform.cityscape2mine()
        if make_random:
            random.shuffle(self.addr)
        if train:
            self.train_addr = self.addr[0:int(len(self.addr)*val_ratio)]
            self.val_addr = self.addr[int(len(self.addr)*val_ratio):]
        else:
            self.train_addr = self.addr

    def build(self):
        sep = os.path.sep
        # ---------------   label_dir
        # ***/label/train/*/*labelIds.png
        res = []
        label_addr = glob.glob(self.label_dir + sep + '*' + sep + '*labelIds.png')
        for i in label_addr:
            cityname = i.split(sep)[-2]
            imgname_temp = i.split(sep)[-1].split('_')[:3]
            img_name = imgname_temp[0] + '_' + imgname_temp[1] + '_' + imgname_temp[2] + '_leftImg8bit.png'
            res.append({
                'img_addr': self.data_dir + sep + cityname + sep + img_name,
                'label_addr': i
            })
        print('load {} data'.format(len(res)))
        return res

    def train_generator(self):
        # for train and eval
        idx = 0
        while 1:
            if idx == len(self.train_addr)-1:
                random.shuffle(self.train_addr)
                idx = 0
            img = io.imread(self.train_addr[idx]['img_addr'], as_grey=self.is_gray)
            label = io.imread(self.train_addr[idx]['label_addr'])
            if self.is_cityscape:
                if self.is_gray:
                    img = measure.block_reduce(img, (4, 4), func=np.max)
                else:
                    img = measure.block_reduce(img, (4,4,1), func=np.max)
                # img = maximum_filter(img, (512, 1024, 3))
                # label = maximum_filter(label, (512, 1024))
                label = measure.block_reduce(label, (4,4), func=np.max)
                label = self.label_trans.img_label_trans(label)
            else:
                if self.is_gray:
                    img = measure.block_reduce(img, (2, 2), func=np.max)
                else:
                    img = measure.block_reduce(img, (2, 2, 1), func=np.max)
                label = measure.block_reduce(label, (2, 2), func=np.max)
                label = self.label_trans.img_label_trans(label)

            idx += 1
            yield (img, label)

    def val_generator(self):
        # for train only
        idx = 0
        while 1:
            if idx == len(self.val_addr) - 1:
                random.shuffle(self.val_addr)
                idx = 0
            img = io.imread(self.val_addr[idx]['img_addr'], as_grey=self.is_gray)
            label = io.imread(self.val_addr[idx]['label_addr'])
            if self.is_cityscape:
                if self.is_gray:
                    img = measure.block_reduce(img, (4, 4), func=np.max)
                else:
                    img = measure.block_reduce(img, (4,4,1), func=np.max)
                # img = maximum_filter(img, (512, 1024, 3))
                # label = maximum_filter(label, (512, 1024))
                label = measure.block_reduce(label, (4, 4), func=np.max)
                label = self.label_trans.img_label_trans(label)
            else:
                if self.is_gray:
                    img = measure.block_reduce(img, (4, 4), func=np.max)
                else:
                    img = measure.block_reduce(img, (4,4,1), func=np.max)
                # img = maximum_filter(img, (512, 1024, 3))
                # label = maximum_filter(label, (512, 1024))
                label = measure.block_reduce(label, (4, 4), func=np.max)
                label = self.label_trans.img_label_trans(label)
            idx += 1

            yield (img, label)

    @staticmethod
    def batched_gen(gen, batch_size=32, flatten=True, is_gray=False, predict_line=False):
        imgs = []
        labels = []
        for img, label in gen:
            imgs.append(img)
            labels.append(label)
            if len(imgs) == batch_size:
                if flatten:
                    if predict_line:
                        pass
                    else:
                        data_shape = labels[0].shape[0] * labels[0].shape[1]
                        nc = labels[0].shape[2]
                        labels = np.concatenate(labels, axis=0)
                        # labels = np.reshape(labels, (batch_size, data_shape, nc))
                        labels = np.reshape(labels, (batch_size, data_shape, nc))
                imgs = np.array(imgs)
                if is_gray:
                    shape = imgs.shape
                    imgs = np.reshape(imgs, (shape[0], shape[1], shape[2], 1))
                labels = np.array(labels)
                yield (imgs, labels)
                imgs = []
                labels = []


class Self_labeled_dataset(Dataset):
    def __init__(self, data_dir, label_dir, train=True, make_random=True
                 , val_ratio=0.3, binary_label=False, is_gray=True, predict_line=False):
        Dataset.__init__(self, data_dir, label_dir, train=train, make_random=make_random,
                         val_ratio=val_ratio, is_cityscape=False, is_gray=is_gray)
        self.label_trans = utils.label_transform.self_labeled(binary_dict=binary_label)
        if binary_label:
            self.classes = 2
        else:
            self.classes = 5
        self.predict_line = predict_line

    def build(self):
        sep = os.path.sep
        # ---------------   label_dir
        # ***/label/train/*.png
        res = []
        img_addr = glob.glob(self.data_dir + sep + '*.jpg')
        for i in img_addr:
            # 0000200_annotated_image.png
            imgname_temp = i.split(sep)[-1].split('.')[0]
            img_name = imgname_temp + '_annotated_image.png'
            res.append({
                'img_addr': i,
                'label_addr': self.label_dir + sep + img_name
            })
        print('load {} data'.format(len(res)))
        return res


    def train_generator(self):
        # for train and eval
        idx = 0
        while 1:
            if idx == len(self.train_addr)-1:
                random.shuffle(self.train_addr)
                idx = 0
            img = io.imread(self.train_addr[idx]['img_addr'],as_grey=self.is_gray)
            label = io.imread(self.train_addr[idx]['label_addr'])
            if self.is_cityscape:
                # TODO useless code:
                img = measure.block_reduce(img, (4,4,1), func=np.max)
                # img = maximum_filter(img, (512, 1024, 3))
                # label = maximum_filter(label, (512, 1024))
                label = measure.block_reduce(label, (4,4), func=np.max)
                label = self.label_trans.img_label_trans(label)
            else:
                if label.shape != (512, 1024):
                    # img = measure.block_reduce(img, (2, 2, 1), func=np.max)
                    # label = measure.block_reduce(label, (2, 2), func=np.max)
                    #print(label.max())
                    label = self.label_trans.img_label_trans2(label)
                    label = transform.resize(label, (256, 512), preserve_range=True)
                    label = np.array(label, dtype=np.int)
                    if self.predict_line:
                        label = line_generate.generate_line(label, label_trans=False)
                    else:
                        label = self.label_trans.label2one_hot(label, dimension=self.classes)
                    img = transform.resize(img, (256, 512))
                else:
                    if self.is_gray:
                        img = measure.block_reduce(img, (2, 2), func=np.max)
                    else:
                        img = measure.block_reduce(img, (2, 2, 1), func=np.max)
                    label = measure.block_reduce(label, (2, 2), func=np.max)
                    if self.predict_line:
                        label = line_generate.generate_line(label)
                    else:
                        label = self.label_trans.img_label_trans(label, dimension=self.classes)
            idx += 1
            yield (img, label)

    def val_generator(self):
        # for train only
        idx = 0
        while 1:
            if idx == len(self.val_addr) - 1:
                random.shuffle(self.val_addr)
                idx = 0
            img = io.imread(self.val_addr[idx]['img_addr'], as_grey=self.is_gray)
            label = io.imread(self.val_addr[idx]['label_addr'])
            if self.is_cityscape:
                img = measure.block_reduce(img, (4, 4, 1), func=np.max)
                # img = maximum_filter(img, (512, 1024, 3))
                # label = maximum_filter(label, (512, 1024))
                label = measure.block_reduce(label, (4, 4), func=np.max)
                label = self.label_trans.img_label_trans(label)
            else:
                if label.shape != (512, 1024):
                    # img = measure.block_reduce(img, (2, 2, 1), func=np.max)
                    # label = measure.block_reduce(label, (2, 2), func=np.max)
                    label = self.label_trans.img_label_trans2(label)
                    label = transform.resize(label, (256, 512), preserve_range=True)
                    label = np.array(label, dtype=np.int)
                    if self.predict_line:
                        label = line_generate.generate_line(label, label_trans=False)
                    else:
                        label = self.label_trans.label2one_hot(label, dimension=self.classes)
                    img = transform.resize(img, (256, 512))
                else:
                    if self.is_gray:
                        img = measure.block_reduce(img, (2, 2), func=np.max)
                    else:
                        img = measure.block_reduce(img, (2, 2, 1), func=np.max)
                    label = measure.block_reduce(label, (2, 2), func=np.max)
                    if self.predict_line:
                        label = line_generate.generate_line(label)
                    else:
                        label = self.label_trans.img_label_trans(label, dimension=self.classes)

                    # label = transform.resize(label, (256, 512), preserve_range=True)
                    # label = np.array(label, dtype=np.int)
                    # print(label.max())
                    # img = transform.resize(img, (256, 512))
            idx += 1
            yield (img, label)


if __name__ == '__main__':
    data = Self_labeled_dataset('./data/self_labeled/img/train', './data/self_labeled/labels/train',is_gray=True,predict_line=True)
    d = data.train_generator()
    d2 = Dataset.batched_gen(d, 1, predict_line=True)
    for i in range(10):
        res, res2 = next(d2)
        print(res2.shape)
        # res2 = np.argmax(res2[0].reshape((256,512,5)),axis=-1)*40
        # io.imsave('test.jpg', res2)
        input()