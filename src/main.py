import argparse
# from azureml.core import Run

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from scipy import ndimage
import segmentation_models_3D as sm
import cv2
import tensorflow as tf
import pandas as pd

import utils
import losses
import unet
import images

def list_files(path):
    brains = os.listdir(path)
    files = []
    for brain in brains:
        brain_path = os.path.join(path, brain)
        files += [os.path.join(brain_path, file) for file in os.listdir(brain_path)]
    # print(len(files))

    return files


def main():
    # print(help('modules'))

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to data')
    args = parser.parse_args()

    data = args.data_path
    print(os.listdir(data))
    training_path = os.path.join(data, 'train/')
    testing_path = os.path.join(data, 'test/')
    validation_path = os.path.join(data, 'val/')

    if not os.path.isdir('outputs'):
        os.mkdir('outputs')

    #images.test_images(training_path)

    train_t2_list = glob.glob(training_path + '/*/*t2.nii.gz')
    train_t1ce_list = glob.glob(training_path + '/*/*t1ce.nii.gz')
    train_flair_list = glob.glob(training_path + '/*/*flair.nii.gz')
    train_mask_list = glob.glob(training_path + '/*/*seg.nii.gz')

    val_t2_list = glob.glob(validation_path + '/*/*t2.nii.gz')
    val_t1ce_list = glob.glob(validation_path + '/*/*t1ce.nii.gz')
    val_flair_list = glob.glob(validation_path + '/*/*flair.nii.gz')
    val_mask_list = glob.glob(validation_path + '/*/*seg.nii.gz')

    batch_size = 2
    for subregion in [1, 2, 3]:
        train_img_datagen = utils.image_loader(train_flair_list, train_t1ce_list, train_t2_list, train_mask_list, batch_size, subregion)
        val_img_datagen = utils.image_loader(val_flair_list, val_t1ce_list, val_t2_list, val_mask_list, batch_size, subregion)

        metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5), tf.keras.metrics.MeanIoU(num_classes=2),
                   losses.dice_coef]

        '''
        metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5), tf.keras.metrics.MeanIoU(num_classes=2),
                   losses.dice_coef, losses.dice_coef_necrotic, losses.dice_coef_edema, losses.dice_coef_enhancing]
        '''
        '''
        metrics = ['accuracy', tf.keras.metrics.MeanIoU(num_classes=2),
                   losses.dice_coef, losses.dice_coef_necrotic, losses.dice_coef_edema, losses.dice_coef_enhancing]
        '''
        LR = 0.0001
        optim = tf.keras.optimizers.Adam(LR)
        steps_per_epoch = len(train_flair_list) // batch_size
        val_steps_per_epoch = len(val_flair_list) // batch_size

        model = unet.unet_model(IMG_HEIGHT=128, IMG_WIDTH=128, IMG_DEPTH=128, IMG_CHANNELS=3, num_classes=2)
        model.compile(optimizer=optim, loss=losses.loss(), metrics=metrics)

        history = model.fit(train_img_datagen,
                            steps_per_epoch=steps_per_epoch,
                            epochs=30,
                            verbose=1,
                            callbacks=utils.callback(segmenting_subregion=subregion),
                            validation_data=val_img_datagen,
                            validation_steps=val_steps_per_epoch)

        model.save(f'outputs/model_{subregion}.h5')

        #Saves history as dictionary
        ''''
        with open(f'outputs/train_history_{subregion}', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        '''

        hist_df = pd.DataFrame(history.history)

        # Saves history as .csv
        hist_csv_file = f'outputs/history_{subregion}.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

    '''
    training_files = list_files(training_path)
    validation_files = list_files(validation)
    testing_files = list_files(testing)

    print(f'training_path: {len(training_files)}, validation: {len(validation_files)}, testing: {len(testing_files)}')
    '''

if __name__ == '__main__':
    main()
