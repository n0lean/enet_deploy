from __future__ import absolute_import
from model.line_model import *
from data_loader import *
from utils.line_generate import vis_line

if __name__ == '__main__':
    weight_path = '../checkpoint/line/weights.line.99-646.00.h5'
    enet = LineNet((256, 512, 1), is_gray=True)
    enet.model.load_weights(weight_path)
    # '/home/pchen11/Project/Test/labels/'
    # img_dir = '../data/self_labeled/img/train'
    # label_dir = '../data/self_labeled/labels/train'
    img_dir = '/home/pchen11/Project/enet/data/self_labeled/img/train'
    label_dir = '/home/pchen11/Project/enet/data/self_labeled/labels/train'
    dataset = Self_labeled_dataset(img_dir, label_dir, train=False, binary_label=False, predict_line=True)

    train_gen = dataset.train_generator()
    batched = dataset.batched_gen(train_gen, 16, is_gray=True, predict_line=True)

    # res = enet.model.evaluate_generator(batched, 6)
    inp, true = next(batched)
    inp1 = inp.copy()
    res = enet.model.predict_on_batch(inp1)
    stat = enet.model.evaluate(inp1, true, batch_size=16)
    print(stat)
    # res = enet.model.predict_generator(batched, 8)
    # print(enet.model.metrics_names)
    for idx, i in enumerate(res):
        i = np.array(i, dtype=np.uint8)
        img = inp[idx] * 255
        img = vis_line(img, i, is_gray=True)
        # i = vis(i, 256, 512)
        # io.imsave('./res/' + str(idx) + '.jpg', i)
        # i = inp[idx] * 0.4 + i * 0.6
        img = np.array(img, dtype=np.uint8)
        img = np.reshape(img, (img.shape[0],img.shape[1]))
        io.imsave('./res/' + str(idx) + '.jpg', img)

