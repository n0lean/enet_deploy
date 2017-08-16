from __future__ import absolute_import

from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from keras.models import Model
from keras.engine.topology import Input

from model import enet

class LineNet(enet.ENet):
    def __init__(self, shape, is_gray=True, optimizer='Adam',
                 loss='mse'):
        self.shape = shape
        self.loss = loss
        self.optimizer = optimizer
        self.is_gray = is_gray
        self.model = self.build()

    def build(self):
        inputs = Input(shape=self.shape)
        front, pooling_index = self.build_en2(inputs)
        end = self.backend(front)

        model = Model(input=inputs, outputs=end)
        model.compile(optimizer=self.optimizer, loss=self.loss,
                      metrics=['mse'])

        return model

    def backend(self, inputs):
        # 32, 64, 128
        inputs = MaxPooling2D((4, 4))(inputs)
        inputs = Conv2D(256, (3, 3))(inputs)

        inputs = MaxPooling2D((2, 2))(inputs)
        inputs = Conv2D(256, (3, 3))(inputs)

        inputs = Flatten()(inputs)
        end = Dense(512)(inputs)
        return end




if __name__ == '__main__':
    model = LineNet((512, 1024, 1))