from __future__ import absolute_import
from utils import label_transform
from skimage import draw


def generate_line(lbl, label_trans=True):
    # lbl should be original label
    # 1: ground, 0: other
    if label_trans:
        l_trans = label_transform.self_labeled(binary_dict=True)
        lbl = l_trans.img_label_trans2(lbl)
    w, h = lbl.shape
    line = []
    # TODO: optimize
    for i in range(h):
        col = lbl[:, i]
        temp = 0
        for j in range(w):
            if col[j] == 0:
                temp = j
        line.append(temp)
    return line


def vis_line(img, line, is_gray=False):
    for i in range(len(line)):
        cc, rr = draw.circle(line[i], i, 3,shape=img.shape)
        if is_gray:
            img[cc, rr] = 0
        else:
            img[cc, rr] = (255, 0, 0)
    return img

if __name__ == '__main__':
    from skimage import io
    path = '/home/pchen11/Project/enet/data/self_labeled/labels/train/frame0299_annotated_image.png'
    lbl = io.imread(path)
    line = generate_line(lbl)
    img = vis_line(lbl, line, True)
    io.imsave('test.jpg', img)
