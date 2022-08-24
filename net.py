from __future__ import print_function

from keras import backend as K
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.models import Model

from keras.layers import UpSampling2D
from typing import Tuple, List

from data import channels
from data import image_cols
from data import image_rows
from data import modalities

batch_norm = False
smooth = 1.0


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def conv_layer(input, filters: int, kernel_size: Tuple[int, int], padding: str, axis: int, activation: str):
    convn = Conv2D(filters, kernel_size, padding=padding)(input)
    if batch_norm:
        convn = BatchNormalization(axis=axis)(convn)
    return Activation(activation)(convn)

def larm(input, filters: int, kernel_size: Tuple[int, int], padding: str, axis: int, activation: str, pool_size: Tuple[int, int]):
    convn = conv_layer(input, filters, kernel_size, padding, axis, activation)
    convn = conv_layer(convn, filters, kernel_size, padding, axis, activation)
    pooln = MaxPooling2D(pool_size=pool_size)(convn)
    return convn, pooln

def rarm(input, concat, filters: int, kernel_size_t: Tuple[int, int], kernel_size: Tuple[int, int], strides: Tuple[int, int], padding: str, axis: int, activation: str):
    upn = Conv2DTranspose(filters, kernel_size_t, strides=strides, padding=padding)(input)
    if concat != []:
        upn = concatenate([upn] + concat, axis=axis)
    convn = conv_layer(upn, filters, kernel_size, padding, axis, activation)
    convn = conv_layer(convn, filters, kernel_size, padding, axis, activation)
    return convn

def change_dim(input, src_dim: int, dest_dim: int):
    if input == None:
        return None
    elif src_dim < dest_dim:
        size = (dest_dim // src_dim, dest_dim // src_dim)
        return UpSampling2D(data_format='channels_last', size=size, interpolation='nearest')(input)
    elif src_dim > dest_dim:
        size = (src_dim // dest_dim, src_dim // dest_dim)
        return MaxPooling2D(data_format='channels_last', pool_size=size)(input)
    else: #src_dim == dest_dim
        return input

def unet(to_conv6: List[str], to_conv7: List[str], to_conv8: List[str], to_conv9: List[str]):
    inputs = Input((image_rows, image_cols, channels * modalities))

    conv1, pool1 = larm(inputs, 32, (3, 3), 'same', 3, 'relu', (2, 2))
    conv2, pool2 = larm(pool1, 64, (3, 3), 'same', 3, 'relu', (2, 2))
    conv3, pool3 = larm(pool2, 128, (3, 3), 'same', 3, 'relu', (2, 2))
    conv4, pool4 = larm(pool3, 256, (3, 3), 'same', 3, 'relu', (2, 2))

    conv5 = conv_layer(pool4, 512, (3, 3), 'same', 3, 'relu')
    conv5 = conv_layer(conv5, 512, (3, 3), 'same', 3, 'relu')

    concat_dict = {'conv4': (conv4, 32), 'conv3': (conv3, 64), 'conv2': (conv2, 128), 'conv1': (conv1, 256), 'none': (None, 0), '': (None, 0)}
    to_conv6 = [change_dim(concat_dict[to_conv6_][0], concat_dict[to_conv6_][1], 32) for to_conv6_ in to_conv6]
    to_conv7 = [change_dim(concat_dict[to_conv7_][0], concat_dict[to_conv7_][1], 64) for to_conv7_ in to_conv7]
    to_conv8 = [change_dim(concat_dict[to_conv8_][0], concat_dict[to_conv8_][1], 128) for to_conv8_ in to_conv8]
    to_conv9 = [change_dim(concat_dict[to_conv9_][0], concat_dict[to_conv9_][1], 256) for to_conv9_ in to_conv9]

    to_conv6 = [c for c in to_conv6 if not c is None]
    to_conv7 = [c for c in to_conv7 if not c is None]
    to_conv8 = [c for c in to_conv8 if not c is None]
    to_conv9 = [c for c in to_conv9 if not c is None]

    conv6 = rarm(conv5, to_conv6, 256, (2, 2), (3, 3), (2, 2), 'same', 3, 'relu')
    conv7 = rarm(conv6, to_conv7, 128, (2, 2), (3, 3), (2, 2), 'same', 3, 'relu')
    conv8 = rarm(conv7, to_conv8, 64, (2, 2), (3, 3), (2, 2), 'same', 3, 'relu')
    conv9 = rarm(conv8, to_conv9, 32, (2, 2), (3, 3), (2, 2), 'same', 3, 'relu')

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model
