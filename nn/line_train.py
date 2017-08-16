from __future__ import absolute_import
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
from data_loader import *
from model.line_model import *


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
    img_dir = './data/self_labeled/img/train'
    label_dir = './data/self_labeled/labels/train'
    # ss = K.tf.Session(config=K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_grouwth=True)))
    ss = K.tf.Session()
    K.set_session(ss)
    ss.run(K.tf.global_variables_initializer())

    dataset = Self_labeled_dataset(img_dir, label_dir, is_gray=True, predict_line=True)

    # model = ENet((256, 512, 1), 5, mode='unpooling', front_trainable=False, is_gray=True)

    opt = Adam(0.001/50.)

    model = LineNet((256,512,1), optimizer=opt)
    model.model.load_weights('./checkpoint/line.h5', False)


    train_gen = dataset.train_generator()
    val_gen = dataset.val_generator()

    model.model.fit_generator(generator=dataset.batched_gen(train_gen, 16, is_gray=True, predict_line=True),
                              steps_per_epoch=14,
                              verbose=1,
                              epochs=100,
                              callbacks=callbacks('./log/line', './checkpoint/line', 'line'),
                              validation_data=dataset.batched_gen(val_gen, 4, is_gray=True, predict_line=True),
                              initial_epoch=0,
                              validation_steps=4)

if __name__ == '__main__':
    train()