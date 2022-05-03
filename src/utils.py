import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
import wandb
import tensorflow as tf
import cv2
import re

import losses


# based on https://stackoverflow.com/a/44007180
def get_dimensions(img, scale=.7):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    return int(left_x), int(right_x), int(top_y), int(bottom_y)


# normalisation based on https://stackoverflow.com/a/59601298
def normalise(image):
    scaler = MinMaxScaler()
    image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    return image


def load_img(flair_list, t1ce_list, t2_list, img_channels):
    images = []
    for flair_name, t1ce_name, t2_name in zip(flair_list, t1ce_list, t2_list):
        flair = nib.load(flair_name).get_fdata()
        t1ce = nib.load(t1ce_name).get_fdata()
        t2 = nib.load(t2_name).get_fdata()

        # crop the images and mask
        left_x, right_x, top_y, bottom_y = get_dimensions(flair[:, :, 0])
        # print(left_x, right_x, top_y, bottom_y)
        flair = flair[left_x:right_x, top_y:bottom_y, :]
        t1ce = t1ce[left_x:right_x, top_y:bottom_y, :]
        t2 = t2[left_x:right_x, top_y:bottom_y, :]

        volume_start = 13
        tmp_flair = np.zeros((128, 128, 128))
        tmp_t1ce = np.zeros((128, 128, 128))
        tmp_t2 = np.zeros((128, 128, 128))

        # resize the images
        inter = cv2.INTER_NEAREST
        for i in range(128):
            tmp_flair[:,:,i] = cv2.resize(flair[:,:,i+volume_start], (128, 128), interpolation=inter)
            tmp_t1ce[:,:,i] = cv2.resize(t1ce[:,:,i+volume_start], (128, 128), interpolation=inter)
            tmp_t2[:,:,i]= cv2.resize(t2[:,:,i+volume_start], (128, 128), interpolation=inter)
        flair = tmp_flair
        t1ce = tmp_t1ce
        t2 = tmp_t2

        # normalise
        flair = normalise(flair)
        t1ce = normalise(t1ce)
        t2 = normalise(t2)

        # stack the sequences
        if img_channels == 3:
            image = np.stack([flair, t1ce, t2], axis=3)
        else:
            image = np.stack([flair, t1ce], axis=3)

    images.append(image)
    images = np.array(images)

    return images


def load_mask(mask_list, segmenting_subregion=0, classes=4):
    images = []
    for mask_name in mask_list:
        mask = nib.load(mask_name).get_fdata()
        mask = mask.astype(np.uint8)
        mask[mask == 4] = 3  # Reassign mask values 4 to 3

        # crop the mask
        left_x, right_x, top_y, bottom_y = get_dimensions(mask[:, :, 0])
        mask = mask[left_x:right_x, top_y:bottom_y, :]

        volume_start = 13
        tmp_mask = np.zeros((128, 128, 128))

        # resize the mask
        inter = cv2.INTER_NEAREST
        for i in range(128):
            tmp_mask[:,:,i]= cv2.resize(mask[:,:,i+volume_start], (128, 128), interpolation=inter)
        mask = tmp_mask

        # segmenting label 1
        if segmenting_subregion == 1:
            mask[mask == 2] = 0
            mask[mask == 3] = 0
        # segmenting label 2
        elif segmenting_subregion == 2:
            mask[mask == 1] = 0
            mask[mask == 3] = 0
            mask[mask == 2] = 1
        # segmenting label 3
        elif segmenting_subregion == 3:
            mask[mask == 1] = 0
            mask[mask == 2] = 0
            mask[mask == 3] = 1

        # image = mask[56:184, 56:184, 13:141]
        # print(np.unique(temp_mask))

        if segmenting_subregion == 0:
            mask = to_categorical(mask, num_classes=4)
        elif classes == 2:
            mask = to_categorical(mask, num_classes=2)
        elif classes == 1:
            mask = tf.one_hot(mask, 1, on_value=0, off_value=1)

        images.append(mask)

    images = np.array(images)

    return images


def image_loader(flair_list, t1ce_list, t2_list, mask_list, batch_size, channels=3, segmenting_subregion=0, classes=4):
    img_len = len(flair_list)

    # keras needs the generator infinite
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < img_len:
            limit = min(batch_end, img_len)
            x = load_img(flair_list[batch_start:limit], t1ce_list[batch_start:limit], t2_list[batch_start:limit],
                         channels)
            y = load_mask(mask_list[batch_start:limit], segmenting_subregion, classes)

            yield (x, y)  # a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size


def show_predictions(my_model, flair, t1ce, t2, mask, channels, subregion, n_slice, epoch, classes=4):
    test_img = load_img(flair, t1ce, t2, img_channels=channels)
    test_prediction = my_model.predict(test_img)
    # print(np.unique(test_prediction))
    test_prediction_argmax = np.argmax(test_prediction, axis=-1)
    # print(np.unique(test_prediction_argmax))
    # print('original shape: ', test_prediction.shape)
    # print('new shape: ', test_prediction_argmax.shape)

    test_mask = load_mask(mask, segmenting_subregion=subregion, classes=classes)
    test_mask_argmax = np.argmax(test_mask, axis=-1)
    # print('mask shape: ', test_mask.shape)
    # print(test_mask.dtype)
    # print(test_prediction.dtype)
    # test_mask = tf.cast(test_mask, tf.float32)
    # print(test_mask.dtype)

    # print('dice:', losses.dice_coef(test_mask, test_prediction).numpy())
    # print('dice edema:', losses.dice_coef_edema(test_mask, test_prediction).numpy())
    # print('dice necrotic:', losses.dice_coef_necrotic(test_mask, test_prediction).numpy())
    # print('dice enhancing:', losses.dice_coef_enhancing(test_mask, test_prediction).numpy())

    if channels == 3:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))
        ax1.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 0], 270), cmap='gray')
        ax1.set_title('Image flair')
        ax2.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 1], 270), cmap='gray')
        ax2.set_title('Image t1ce')
        ax3.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 2], 270), cmap='gray')
        ax3.set_title('Image t2')
        ax4.imshow(ndimage.rotate(test_mask_argmax[0][:, :, n_slice], 270))
        ax4.set_title('Mask')
        ax5.imshow(ndimage.rotate(test_prediction_argmax[0][:, :, n_slice], 270))
        ax5.set_title('Prediction')
        # fig.savefig(f'outputs/test.png')
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
        ax1.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 0], 270), cmap='gray')
        ax1.set_title('Image flair')
        ax2.imshow(ndimage.rotate(test_img[0][:, :, n_slice, 1], 270), cmap='gray')
        ax2.set_title('Image t1ce')
        ax3.imshow(ndimage.rotate(test_mask_argmax[0][:, :, n_slice], 270))
        ax3.set_title('Mask')
        ax4.imshow(ndimage.rotate(test_prediction_argmax[0][:, :, n_slice], 270))
        ax4.set_title('Prediction')
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
    '''
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001, verbose=1),
                 csv_logger]
    '''
    callbacks = [
        EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=1, mode='auto'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001, verbose=1),
        PredictionCallback(flair, t1ce, t2, mask, channels, subregion, n_slice, classes),
        csv_logger
    ]

    return callbacks
