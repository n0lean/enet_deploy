from __future__ import absolute_import
from model.enet import *
from data_loader import *
from eval.predict import vis
from utils.line_generate import vis_line, generate_line

if __name__ == '__main__':
    # weight_path = '../checkpoint/post5aug/post5aug_best.h5'
    weight_path = '../checkpoint/post2/post2_best.h5'

    # enet = ENet((256, 512, 1), classes=5, is_gray=True)
    enet = ENet((256, 512, 1), classes=2)

    enet.model.load_weights(weight_path)
    # '/home/pchen11/Project/Test/labels/'
    # img_dir = '../data/self_labeled/img/train'
    # label_dir = '../data/self_labeled/labels/train'
    img_dir = '/home/pchen11/Project/enet/data/self_labeled/img/train'
    label_dir = '/home/pchen11/Project/enet/data/self_labeled/labels/train'
    # dataset = Self_labeled_dataset(img_dir, label_dir,train=False, binary_label=False)
    dataset = Self_labeled_dataset(img_dir, label_dir,train=False, binary_label=True)

    train_gen = dataset.train_generator()
    batched = dataset.batched_gen(train_gen, 16, is_gray=True)

    # res = enet.model.evaluate_generator(batched, 6)
    inp, true = next(batched)
    inp1 = inp.copy()
    res = enet.model.predict_on_batch(inp1)
    stat = enet.model.evaluate(inp1, true, batch_size=16)
    print(stat)
    # res = enet.model.predict_generator(batched, 8)
    # print(enet.model.metrics_names)
    for idx, i0 in enumerate(res):
        i = i0.copy()
        i = np.argmax(i, axis=-1)
        i = np.reshape(i, (256, 512))
        line = generate_line(i, False)
        img = inp[idx] * 255
        img = vis_line(img, line, True)
        img = np.reshape(img, (img.shape[0],img.shape[1]))
        img = np.array(img, dtype=np.uint8)
        io.imsave('./res/' + str(idx) + '.jpg', img)

        i0 = vis(i0, 256, 512)
        i0 = inp[idx] * 0.4 + i0 * 0.6
        i0 = np.array(i0, dtype=np.uint8)
        io.imsave('./res2/' + str(idx) + '.jpg', i0)

