import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
import wandb
import tensorflow as tf
import cv2
import re

import losses
from utils.data_processing import *

def show_predictions(my_model, flair, t1ce, t2, mask, channels, subregion, n_slice, epoch, classes=4):
    test_img = load_img(flair, t1ce, t2, img_channels=channels)

    test_prediction = my_model.predict(test_img)
    test_prediction_argmax = np.argmax(test_prediction, axis=-1)

    test_mask = load_mask(mask, segmenting_subregion=subregion, classes=classes)
    test_mask_argmax = np.argmax(test_mask, axis=-1)

    custom_cmap = get_custom_cmap()

    if channels == 3:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))
        ax1.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 0], 270), cmap='gray')
        ax1.set_title('Image flair')
        ax1.axis('off')
        ax2.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 1], 270), cmap='gray')
        ax2.set_title('Image t1ce')
        ax2.axis('off')
        ax3.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 2], 270), cmap='gray')
        ax3.set_title('Image t2')
        ax3.axis('off')
        ax4.imshow(ndimage.rotate(test_mask_argmax[0][:, :, n_slice], 270), cmap=custom_cmap)
        ax4.set_title('Mask')
        ax4.axis('off')
        ax5.imshow(ndimage.rotate(test_prediction_argmax[0][:, :, n_slice], 270), cmap=custom_cmap)
        ax5.set_title('Prediction')
        ax5.axis('off')
        # fig.savefig(f'outputs/test.png')
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
        ax1.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 0], 270), cmap='gray')
        ax1.set_title('Image flair')
        ax1.axis('off')
        ax2.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 1], 270), cmap='gray')
        ax2.set_title('Image t1ce')
        ax2.axis('off')
        ax3.imshow(ndimage.rotate(test_mask_argmax[0][:, :, n_slice], 270), cmap=custom_cmap)
        ax3.set_title('Mask')
        ax3.axis('off')
        ax4.imshow(ndimage.rotate(test_prediction_argmax[0][:, :, n_slice], 270), cmap=custom_cmap)
        ax4.set_title('Prediction')
        ax4.axis('off')
        # fig.savefig(f'outputs/test.png')

    fig.savefig(f'outputs/prediction_epoch{epoch}.png')
    plt.close()

    return ndimage.rotate(test_img[0][:, :, n_slice, 0], 270), \
           ndimage.rotate(test_mask_argmax[0][:, :, n_slice], 270), \
           ndimage.rotate(test_prediction_argmax[0][:, :, n_slice], 270)


class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, flair, t1ce, t2, mask, channels, subregion, n_slice, classes=4):
        self.flair = flair
        self.t1ce = t1ce
        self.t2 = t2
        self.mask = mask
        self.channels = channels
        self.subregion = subregion
        self.n_slice = n_slice
        self.classes = classes
        if subregion == 0:
            self.labels = {0: 'background', 1: 'necrotic', 2: 'edema', 3: 'enhancing'}
        elif subregion == 1:
            self.labels = {0: 'background', 1: 'necrotic'}
        elif subregion == 2:
            self.labels = {0: 'background', 1: 'edema'}
        else:
            self.labels = {0: 'background', 1: 'enhancing'}

    # util function for generating interactive image mask from components
    def wandb_mask(self, img, true_mask, pred_mask):
        return wandb.Image(img, masks={
            "ground truth": {
                "mask_data": true_mask,
                "class_labels": self.labels
            },
            "prediction": {
                "mask_data": pred_mask,
                "class_labels": self.labels
            }
        })

    def on_epoch_end(self, epoch, logs=None):
        image, true_mask, pred_mask = show_predictions(
            self.model, self.flair, self.t1ce, self.t2, self.mask,
            self.channels, self.subregion, self.n_slice, epoch, self.classes
        )

        # mask = self.wandb_mask(image.numpy(), pred_mask.numpy(), true_mask.numpy())
        mask = self.wandb_mask(image, true_mask, pred_mask)
        wandb.log({"predictions": mask}, commit=False)


def callback(flair, t1ce, t2, mask, channels, subregion, n_slice, classes):
    csv_logger = CSVLogger(f'outputs/training_{subregion}.log', separator=',', append=False)
    callbacks = [
        EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=1, mode='auto'),
        #ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001, verbose=1),
        PredictionCallback(flair, t1ce, t2, mask, channels, subregion, n_slice, classes),
        csv_logger
    ]

    return callbacks