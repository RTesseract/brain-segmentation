from __future__ import print_function

import os

import numpy as np

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.utils import plot_model

from data import load_data, oversample, load_concat
from net import dice_coef, dice_coef_loss, unet

from gc import collect

def train(
    epochs: int,
    batch_size: int,
    augment: bool,
    verbose: bool,
    fname: str,
    to_conv6: list,
    to_conv7: list,
    to_conv8: list,
    to_conv9: list,
    log_path: str,
    train_images_path: str,
    valid_images_path: str,
    weights_path: str,
    log_folder_path: str,
    architecture_path: str
):
    imgs_train, imgs_mask_train, _ = load_data(train_images_path)
    mean = np.mean(imgs_train)
    std = np.std(imgs_train)
    imgs_train -= mean
    imgs_train /= std
    imgs_valid, imgs_mask_valid, _ = load_data(valid_images_path)
    imgs_valid -= mean
    imgs_valid /= std
    imgs_train, imgs_mask_train = oversample(imgs_train, imgs_mask_train, augment=augment)
    print('aasu: images augmented')

    model = unet(to_conv6, to_conv7, to_conv8, to_conv9)
    
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    #plot_model(
    #    model=model,
    #    to_file=os.path.join(architecture_path, f'{fname}.png'),
    #    show_shapes=True,
    #    show_layer_names=True,
    #    rankdir='LR',
    #    expand_nested=False,
    #    dpi=96,
    #)

    collect()
    print(f'aasu: starting {fname}')
    model.fit(
        imgs_train,
        imgs_mask_train,
        validation_data=(imgs_valid, imgs_mask_valid),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=False,
        callbacks=[
            CSVLogger(
                filename=os.path.join(log_folder_path, f'{fname}.csv'),
                separator=',',
                append=False
            ),
            #TensorBoard(
            #    log_dir=log_path,
            #    histogram_freq=1
            #),
            EarlyStopping(
                monitor='val_loss',
                verbose=1,
                patience=4
            )
        ],
        verbose=verbose,
        use_multiprocessing=False
    )

    model.save_weights(os.path.join(weights_path, f'{fname}.h5'))
    print(f'aasu: ending {fname}')


def train_main(device: str, small: bool):
    tf.set_random_seed(0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)

    train_images_path = "../archive/train"
    valid_images_path = "../archive/validate"
    weights_path = "../out/weights"
    log_folder_path = "../out/logs"
    architecture_path = "../out/architectures"

    if small:
        train_images_path += "_small"
        valid_images_path += "_small"
    #epochs = 128
    epochs = 1024
    batch_size = 32
    augment = True
    verbose = 1

    if not os.path.exists("../out"):
        os.mkdir("../out")
    if not os.path.exists(log_folder_path):
        os.mkdir(log_folder_path)
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    #if not os.path.exists(architecture_path):
    #    os.mkdir(architecture_path)
    
    fname, log_path = None, None
    with tf.device(device):
        for to_conv6, to_conv7, to_conv8, to_conv9 in load_concat():
            fname = f"{','.join(to_conv6)}_{','.join(to_conv7)}_{','.join(to_conv8)}_{','.join(to_conv9)}"
            #log_path = f'{log_folder_path}/{fname}'
            #if not os.path.exists(log_path):
            #    os.mkdir(log_path)
            train(
                epochs,
                batch_size,
                augment,
                verbose,
                fname,
                to_conv6,
                to_conv7,
                to_conv8,
                to_conv9,
                log_path,
                train_images_path,
                valid_images_path,
                weights_path,
                log_folder_path,
                architecture_path
            )