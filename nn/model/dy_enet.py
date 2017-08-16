from __future__ import absolute_import

from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.core import SpatialDropout2D, Permute
from keras.layers.core import Activation, Reshape
from keras.layers.merge import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.engine.topology import Input

from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D


class ENet(object):
    def __init__(self, shape, classes, mode='full_native',
                 optimizer='Adam', loss='categorical_crossentropy'):
        self.shape = shape
        self.classes = classes
        self.mode = mode
        self.optimizer = optimizer
        self.loss = loss
        self.model = self.build()


    def build(self):
        inputs = Input(shape=self.shape)
        if self.mode == 'full_native':
            front = self.build_en(inputs)
            end = self.build_de(front, self.classes)
            # to avoid reshape
            # end = Reshape((self.shape[0] * self.shape[1], self.classes))(end)
            end = Activation('softmax')(end)

            model = Model(inputs=inputs, outputs=end)
            model.compile(optimizer=self.optimizer, loss=self.loss,
                               metrics=['accuracy', 'mse'])
        return model

    def initial_block(self, inputs, nb_filters=13, nb_row=3, nb_col=3,
                      strides=(2,2)):
        conv = Conv2D(nb_filters,(nb_row, nb_col), padding='same', strides=strides)(inputs)
        max_pool = MaxPooling2D()(inputs)
        merged = concatenate([conv, max_pool], axis=3)
        return merged

    def encoder_bottleneck(self, inputs, output, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):

        internal = output // internal_scale
        encoder = inputs

        input_stride = 2 if downsample else 1
        encoder = Conv2D(internal, (input_stride, input_stride),
                         strides=(input_stride, input_stride),
                         use_bias=False)(encoder)

        encoder = BatchNormalization(momentum=0.1)(encoder)
        encoder = PReLU(shared_axes=[1,2])(encoder)

        if not asymmetric and not dilated:
            encoder = Conv2D(internal, (3,3), padding='same')(encoder)
        elif asymmetric:
            encoder = Conv2D(internal, (1, asymmetric), padding='same', use_bias=False)(encoder)
            encoder = Conv2D(internal, (asymmetric, 1), padding='same')(encoder)
        elif dilated:
            encoder = Conv2D(internal, (3, 3), dilation_rate=(dilated, dilated), padding='same')(encoder)
        else:
            raise (Exception('You shouldn\'t be here'))

        encoder = BatchNormalization(momentum=0.1)(encoder)
        encoder = PReLU(shared_axes=[1,2])(encoder)

        encoder = Conv2D(output, (1,1), use_bias=False)(encoder)

        encoder = BatchNormalization(momentum=0.1)(encoder)
        encoder = SpatialDropout2D(dropout_rate)(encoder)

        other = inputs
        if downsample:
            other = MaxPooling2D()(other)
            other = Permute((1, 3, 2))(other)
            pad_feature_maps = output - inputs.get_shape().as_list()[3]
            tb_pad =  (0, 0)
            lr_pad = (0, pad_feature_maps)
            other = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
            other = Permute((1, 3, 2))(other)

        encoder = add([encoder, other])
        encoder = PReLU(shared_axes=[1, 2])(encoder)
        return encoder



    def build_en(self, inputs, dropout_rate=0.01):
        enet = self.initial_block(inputs)
        enet = self.encoder_bottleneck(enet, 64, downsample=True, dropout_rate=dropout_rate)  # bottleneck 1.0
        for i in range(4):
            enet = self.encoder_bottleneck(enet, 64, dropout_rate=dropout_rate)  # bottleneck 1.i

        enet = self.encoder_bottleneck(enet, 128, downsample=True)  # bottleneck 2.0
        # bottleneck 2.x and 3.x
        for i in range(2):
            enet = self.encoder_bottleneck(enet, 128)  # bottleneck 2.1
            enet = self.encoder_bottleneck(enet, 128, dilated=2)  # bottleneck 2.2
            enet = self.encoder_bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.3
            enet = self.encoder_bottleneck(enet, 128, dilated=4)  # bottleneck 2.4
            enet = self.encoder_bottleneck(enet, 128)  # bottleneck 2.5
            enet = self.encoder_bottleneck(enet, 128, dilated=8)  # bottleneck 2.6
            enet = self.encoder_bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.7
            enet = self.encoder_bottleneck(enet, 128, dilated=16)  # bottleneck 2.8
        return enet


    def decoder_bottleneck(self, encoder, output, upsample=False, reverse_module=False):
        internal = output // 4

        x = Conv2D(internal, (1, 1), use_bias=False)(encoder)
        x = BatchNormalization(momentum=0.1)(x)
        x = Activation('relu')(x)
        if not upsample:
            x = Conv2D(internal, (3, 3), padding='same', use_bias=True)(x)
        else:
            x = Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization(momentum=0.1)(x)
        x = Activation('relu')(x)

        x = Conv2D(output, (1, 1), padding='same', use_bias=False)(x)

        other = encoder
        if encoder.get_shape()[-1] != output or upsample:
            other = Conv2D(output, (1, 1), padding='same', use_bias=False)(other)
            other = BatchNormalization(momentum=0.1)(other)
            if upsample and reverse_module is not False:
                other = UpSampling2D(size=(2, 2))(other)

        if upsample and reverse_module is False:
            decoder = x
        else:
            x = BatchNormalization(momentum=0.1)(x)
            decoder = add([x, other])
            decoder = Activation('relu')(decoder)

        return decoder


    def build_de(self, encoder, nc):
        enet = self.decoder_bottleneck(encoder, 64, upsample=True, reverse_module=True)  # bottleneck 4.0
        enet = self.decoder_bottleneck(enet, 64)  # bottleneck 4.1
        enet = self.decoder_bottleneck(enet, 64)  # bottleneck 4.2
        enet = self.decoder_bottleneck(enet, 16, upsample=True, reverse_module=True)  # bottleneck 5.0
        enet = self.decoder_bottleneck(enet, 16)  # bottleneck 5.1

        enet = Conv2DTranspose(filters=nc, kernel_size=(2, 2), strides=(2, 2), padding='same')(enet)
        return enet

if __name__ == '__main__':
    enet = ENet((None, None, 3), 4)
    enet.model.summary()