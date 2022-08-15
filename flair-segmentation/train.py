from __future__ import print_function

import os
import sys

import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
#from keras import backend as K
#from keras.callbacks import TensorBoard
#from keras.optimizers import Adam
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.callbacks import TensorBoard
from tensorflow.compat.v1.keras.optimizers import Adam

from tensorflow.compat.v1.keras.utils import plot_model

from data import load_data
from data import oversample
from net import dice_coef
from net import dice_coef_loss
from net import unet

from typing import Dict

#train_images_path = "./data/train/""..\..\archive\test"
#valid_images_path = "./data/valid/"
#init_weights_path = "./weights_128.h5"
small = True
train_images_path = "..\\..\\archive\\train" + ("_small" if small else "")
valid_images_path = "..\\..\\archive\\validate" + ("_small" if small else "")
#init_weights_path = "weights_64.h5"
weights_path = "."
log_path = "."

gpu = "0"

#epochs = 128
epochs = 1
batch_size = 32
base_lr = 1e-5

def load():
    print('aasu: loading and pre-processing data...', end='')
    imgs_train, imgs_mask_train, _ = load_data(train_images_path)
    mean = np.mean(imgs_train)
    std = np.std(imgs_train)
    imgs_train -= mean
    imgs_train /= std
    imgs_valid, imgs_mask_valid, _ = load_data(valid_images_path)
    imgs_valid -= mean
    imgs_valid /= std
    imgs_train, imgs_mask_train = oversample(imgs_train, imgs_mask_train)
    print('completed')
    return imgs_train, imgs_valid, imgs_mask_valid, imgs_mask_train

def train(imgs_train, imgs_valid, imgs_mask_valid, imgs_mask_train):
    #model = unet()
    #d = {'ct_conv6': 'conv4', 'ct_conv7': 'conv3', 'ct_conv8': 'conv2', 'ct_conv9': 'conv1'}
    #model = unet('conv4', 'conv3', 'conv2', 'conv1')

    #models.append(unet(ct_conv6, ct_conv7, ct_conv8, ct_conv9))
    #fnames.append(f'{ct_conv6}_{ct_conv7}_{ct_conv8}_{ct_conv9}')
    fname = f'{ct_conv6}_{ct_conv7}_{ct_conv8}_{ct_conv9}'
    print(f'aasu: initializing model {fname} ...', end='')
    model = unet(ct_conv6, ct_conv7, ct_conv8, ct_conv9)
    model.compile(optimizer=Adam(lr=base_lr), loss=dice_coef_loss, metrics=[dice_coef])
    
    #model = unet(architecture['ct_conv6'], architecture['ct_conv7'], architecture['ct_conv8'], architecture['ct_conv9'])
    #model.summary()
    #fname = '_'.join(val for val in architecture.values())

    #if os.path.exists(init_weights_path):
    #    #model.load_weights(init_weights_path)
    #    model.load_weights(init_weights_path, by_name=True, skip_mismatch=True)

    #for model in unets.keys():
    #    model.compile(optimizer=Adam(lr=base_lr), loss=dice_coef_loss, metrics=[dice_coef])
    #optimizer = Adam(lr=base_lr)
    #model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])

    if not os.path.exists(log_path):
        os.mkdir(log_path)
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    training_log = TensorBoard(log_dir=log_path)

    #print('aasu: start training? (y/[n])', end='')
    #if input() != 'y':
    #    return
    #model.fit(
    #    imgs_train,
    #    imgs_mask_train,
    #    validation_data=(imgs_valid, imgs_mask_valid),
    #    batch_size=batch_size,
    #    epochs=epochs,
    #    shuffle=True,
    #    callbacks=[training_log],
    #)

    print(f'...completed\naasu: training model {fname} ...', end='')
    train_history = model.fit(
        imgs_train,
        imgs_mask_train,
        validation_data=(imgs_valid, imgs_mask_valid),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=[training_log],
        verbose=1,
        use_multiprocessing=False
    )
    print('...completed\naasu: outputting...', end='')
    #model.save_weights(os.path.join(weights_path, "weights_{}.h5".format(epochs)))
    model.save_weights(os.path.join(weights_path, f'weights_{fname}.h5'))
    plot_model(
        model=model,
        to_file=f'model_{fname}.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='LR',
        expand_nested=False,
        dpi=96,
    )
    with open(f'history.txt', 'a') as fp:
        fp.write(f'{fname}: {str(train_history.history)}\n')
    print('completed')


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    #if len(sys.argv) > 1:
    #    gpu = sys.argv[1]
    #device = "/gpu:" + gpu
    device = "/gpu:0"

    try:
        with open(f'history.txt', 'w') as fp:
            pass
        with tf.device(device):
            imgs_train, imgs_valid, imgs_mask_valid, imgs_mask_train = load()
            for ct_conv9 in ['conv4', 'conv3', 'conv2', 'conv1']:
                for ct_conv8 in ['conv4', 'conv3', 'conv2', 'conv1']:
                    for ct_conv7 in ['conv4', 'conv3', 'conv2', 'conv1']:
                        for ct_conv6 in ['conv4', 'conv3', 'conv2', 'conv1']:
                            train(imgs_train, imgs_valid, imgs_mask_valid, imgs_mask_train)
    except KeyboardInterrupt:
        print('aasu: interrupted')
