from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.callbacks import CSVLogger, TensorBoard, EarlyStopping
from tensorflow.compat.v1.keras.optimizers import Adam

from tensorflow.compat.v1.keras.utils import plot_model

from data import load_data, oversample, load_concat
from net import dice_coef, dice_coef_loss, unet

from gc import collect

train_images_path = "../archive/train"
valid_images_path = "../archive/validate"
weights_path = "../out/weights"
log_folder_path = "../out/logs"
architecture_path = "../../out/architectures"
history_path = "../out/history.csv"

base_lr = 1e-5

def train(augment: bool, verbose: bool):
    imgs_train, imgs_mask_train, _ = load_data(train_images_path)
    mean = np.mean(imgs_train)
    std = np.std(imgs_train)
    imgs_train -= mean
    imgs_train /= std
    imgs_valid, imgs_mask_valid, _ = load_data(valid_images_path)
    imgs_valid -= mean
    imgs_valid /= std
    imgs_train, imgs_mask_train = oversample(imgs_train, imgs_mask_train, augment=augment)

    model = unet(to_conv6, to_conv7, to_conv8, to_conv9)
    #TODO: init weight
    model.compile(optimizer=Adam(lr=base_lr), loss=dice_coef_loss, metrics=[dice_coef])
    plot_model(
        model=model,
        to_file=os.path.join(architecture_path, f'{fname}.png'),
        show_shapes=True,
        show_layer_names=True,
        rankdir='LR',
        expand_nested=False,
        dpi=96,
    )

    collect()
    print(f'aasu: starting {fname}')
    train_history = model.fit(
        imgs_train,
        imgs_mask_train,
        validation_data=(imgs_valid, imgs_mask_valid),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=False,
        callbacks=[
            CSVLogger(
                filename=os.path.join(log_path, f'{fname}.csv'),
                separator=',',
                append=True
            ),
            TensorBoard(
                log_dir=log_path,
                histogram_freq=1,
                write_graph=True,
                write_grads=True,
                write_images=True
            ),
            EarlyStopping(
                monitor='val_loss',
                verbose=1,
                patience=2
            )
        ],
        verbose=verbose,
        use_multiprocessing=False
    )

    model.save_weights(os.path.join(weights_path, f'{fname}.h5'))
    with open(history_path, 'a') as fp:
        for k, v in train_history.history.items():
            fp.write(f"{fname},{k},{len(v)},{','.join(v)}\n")
    print(f'aasu: ending {fname}')


if __name__ == "__main__":
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)

    device = sys.argv[1]
    if sys.argv[2] == 'small':
        train_images_path += "_small"
        valid_images_path += "_small"
    epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    augment = True if sys.argv[5] == 'True' else False
    verbose = int(sys.argv[6])

    if not os.path.exists("../../out"):
        os.mkdir("../../out")
    if not os.path.exists(log_folder_path):
        os.mkdir(log_folder_path)
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    if not os.path.exists(architecture_path):
        os.mkdir(architecture_path)
    history_path = os.path.join(architecture_path, 'history.txt')
    with open(history_path, 'w') as fp:
        fp.write(f"fname,field,len,values\n")
    
    fname, log_path = None, None
    try:
        with tf.device(device):
            for to_conv6, to_conv7, to_conv8, to_conv9 in load_concat():
                fname = f"{','.join(to_conv6)}_{','.join(to_conv7)}_{','.join(to_conv8)}_{','.join(to_conv9)}"
                log_path = f'{log_folder_path}/{fname}'
                if not os.path.exists(log_path):
                    os.mkdir(log_path)
                train(augment, verbose)
    except KeyboardInterrupt:
        print('\naasu: interrupted')
