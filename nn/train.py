from __future__ import absolute_import

from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint

from data_loader import *
from model.enet import *


def callbacks(log_dir, checkpoint_dir, model_name):
    """ 
    :param log_dir: 
    :param checkpoint_dir: 
    :param model_name: 
    :return: 
    """
    cbs = []

    tb = TensorBoard(log_dir=log_dir,
                     histogram_freq=1,
                     # write_graph=True,
                     write_images=True)
    cbs.append(tb)

    best_model = os.path.join(checkpoint_dir, '{}_best.h5'.format(model_name))
    save_best = ModelCheckpoint(best_model, save_best_only=True)
    cbs.append(save_best)

    checkpoint_file = os.path.join(checkpoint_dir, 'weights.' + model_name + '.{epoch:02d}-{val_loss:.2f}.h5')
    checkpoints = ModelCheckpoint(filepath=checkpoint_file, verbose=1)
    cbs.append(checkpoints)

    return cbs


def train():
    # hard-coded addrs
    # './data/cityscape/img/train', './data/cityscape/label/train'
    img_dir = './data/cityscapes/img/train'
    label_dir = './data/cityscapes/labels/train'
    assert K.backend() == 'tensorflow'
    # ss = K.tf.Session(config=K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_grouwth=True)))
    ss = K.tf.Session()
    K.set_session(ss)
    ss.run(K.tf.global_variables_initializer())

    dataset = Dataset(img_dir, label_dir, is_cityscape=True, is_gray=True)

    # model = ENet((256, 512, 1), 5, mode='unpooling',is_gray=True)
    model = ENet((256, 512, 1), 2, mode='unpooling', is_gray=True)

    model.model.summary()

    train_gen = dataset.train_generator()
    val_gen = dataset.val_generator()

    model.model.fit_generator(generator=dataset.batched_gen(train_gen, 16, is_gray=True),
                              # 3000 data
                              steps_per_epoch=100,
                              verbose=1,
                              epochs=20,
                              callbacks=callbacks('./log/pre5/', './checkpoint/pre5/', 'pre_train'),
                              validation_data=dataset.batched_gen(val_gen, 8, is_gray=True),
                              initial_epoch=0,
                              validation_steps=8)

if __name__ == '__main__':
    train()
