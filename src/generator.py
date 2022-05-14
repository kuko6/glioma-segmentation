import cv2
from matplotlib import image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import albumentations as A
import random

# ----------------------------------------------------------------------------------------- #
# Data generator used for providing the preprocessed, optionally augmented, batches of data.
# The preprocessing includes cropping the background, downsampling to 128x128x128
# and normalising the volumes with MinMax normalisation.
# The implemented augmentations are: 
#   - vertical flipping with 50% chance 
#   - random rotation in range (-10, 10) with 50% chance
#
# Inspired by:
#   - https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# ----------------------------------------------------------------------------------------- #

class BratsGen(tf.keras.utils.Sequence):
    def __init__(self, flair_list, t1ce_list, t2_list, t1_list, mask_list, img_dim=(128, 128, 128),
                 img_channels=3, classes=4, batch_size=2, segmenting_subregion=0, aug=False):
        self.batch_size = batch_size
        self.classes = classes
        self.img_channels = img_channels
        self.img_dim = img_dim
        self.flair_list = flair_list
        self.t1ce_list = t1ce_list
        self.t2_list = t2_list
        self.t1_list = t1_list
        self.mask_list = mask_list
        self.segmenting_subregion = segmenting_subregion
        self.aug = aug

    def __len__(self):
        return len(self.flair_list) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_flair = self.flair_list[i: i + self.batch_size]
        batch_t1ce = self.t1ce_list[i: i + self.batch_size]
        batch_t2 = self.t2_list[i: i + self.batch_size]
        batch_t1 = self.t1_list[i: i + self.batch_size]
        batch_mask = self.mask_list[i: i + self.batch_size]

        X, y = self.__data_generation(flair_list=batch_flair, t1ce_list=batch_t1ce, t2_list=batch_t2, t1_list=batch_t1, mask_list=batch_mask)

        return X, y

    # based on https://stackoverflow.com/a/44007180
    def __get_dimensions(self, img, scale=.7):
        center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
        width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
        left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
        top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
        return int(left_x), int(right_x), int(top_y), int(bottom_y)

    # normalisation based on https://stackoverflow.com/a/59601298
    def __normalise(self, image):
        scaler = MinMaxScaler()
        image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
        return image

    # augmentations
    def __augmentation(self, flair, t1ce, t2, t1, mask):
        verticalFlip = A.Compose(
            [A.VerticalFlip(p=1)], 
            additional_targets={'image1' : 'image', 'image2': 'image', 'image3': 'image'}
        ) # 50% chance
        randomRotate = A.Compose(
            [A.Rotate(p=1, limit=(-20, 20), border_mode=cv2.BORDER_CONSTANT)],
            additional_targets={'image1' : 'image', 'image2': 'image', 'image3': 'image'}
        ) # 60% chance

        if random.random() < 0.5:
            t = verticalFlip(image=flair, image1=t1, image2=t1ce, image3=t2, mask=mask)
            flair = t['image']
            t1 = t['image1']
            t1ce = t['image2']
            t2 = t['image3']
            mask = t['mask']

        if random.random() < 0.6:
            t = randomRotate(image=flair, image1=t1, image2=t1ce, image3=t2, mask=mask)
            flair = t['image']
            t1 = t['image1']
            t1ce = t['image2']
            t2 = t['image3']
            mask = t['mask']

        return flair, t1ce, t2, t1, mask

    def __data_generation(self, flair_list, t1ce_list, t2_list, t1_list, mask_list):
        images = []
        masks = []
        for flair_name, t1ce_name, t2_name, t1_name, mask_name in zip(flair_list, t1ce_list, t2_list, t1_list, mask_list):
            flair = nib.load(flair_name).get_fdata()
            t1ce = nib.load(t1ce_name).get_fdata()
            t2 = nib.load(t2_name).get_fdata()
            t1 = nib.load(t1_name).get_fdata()
            mask = nib.load(mask_name).get_fdata()

            # crop the images and mask
            left_x, right_x, top_y, bottom_y = self.__get_dimensions(flair[:, :, 0])
            flair = flair[left_x:right_x, top_y:bottom_y, :]
            t1ce = t1ce[left_x:right_x, top_y:bottom_y, :]
            t2 = t2[left_x:right_x, top_y:bottom_y, :]
            t1 = t1[left_x:right_x, top_y:bottom_y, :]
            mask = mask[left_x:right_x, top_y:bottom_y, :]

            volume_start = 13
            volume_end = 141
            total_slices = 128
            tmp_flair = np.zeros((128, 128, 128))
            tmp_t1ce = np.zeros((128, 128, 128))
            tmp_t2 = np.zeros((128, 128, 128))
            tmp_t1 = np.zeros((128, 128, 128))
            tmp_mask = np.zeros((128, 128, 128))

            # resize the images and mask
            inter = cv2.INTER_NEAREST
            for i in range(total_slices):
                tmp_flair[:, :, i] = cv2.resize(flair[:, :, i + volume_start], (128, 128), interpolation=inter)
                tmp_t1ce[:, :, i] = cv2.resize(t1ce[:, :, i + volume_start], (128, 128), interpolation=inter)
                tmp_t2[:, :, i] = cv2.resize(t2[:, :, i + volume_start], (128, 128), interpolation=inter)
                tmp_t1[:, :, i] = cv2.resize(t1[:, :, i + volume_start], (128, 128), interpolation=inter)
                tmp_mask[:, :, i] = cv2.resize(mask[:, :, i + volume_start], (128, 128), interpolation=inter)
            flair = tmp_flair
            t1ce = tmp_t1ce
            t2 = tmp_t2
            t1 = tmp_t1
            mask = tmp_mask

            # augmentations
            if self.aug:
                flair, t1ce, t2, t1, mask = self.__augmentation(flair=flair, t1ce=t1ce, t2=t2, t1=t1, mask=mask)

            # ==================== Sequences ==================== #
            # normalise
            flair = self.__normalise(flair)
            t1ce = self.__normalise(t1ce)
            t2 = self.__normalise(t2)
            t1 = self.__normalise(t1)

            # stack the sequences
            if self.img_channels == 4:
                image = np.stack([flair, t1ce, t2, t1], axis=3)
            elif self.img_channels == 3:
                image = np.stack([flair, t1ce, t2], axis=3)
            elif self.img_channels == 2:
                image = np.stack([flair, t1ce], axis=3)
            else:
                image = np.stack([flair], axis=3)

            # ==================== Mask ==================== #
            # change label 4 to label 3 because label 3 is empty
            mask[mask == 4] = 3

            # segmenting label 1
            if self.segmenting_subregion == 1:
                mask[mask == 2] = 0
                mask[mask == 3] = 0
            # segmenting label 2
            elif self.segmenting_subregion == 2:
                mask[mask == 1] = 0
                mask[mask == 3] = 0
                mask[mask == 2] = 1
            # segmenting label 3
            elif self.segmenting_subregion == 3:
                mask[mask == 1] = 0
                mask[mask == 2] = 0
                mask[mask == 3] = 1

            # encode
            if self.segmenting_subregion == 0:
                mask = to_categorical(mask, num_classes=4)
            elif self.classes == 2:
                mask = to_categorical(mask, num_classes=2)
            elif self.classes == 1:
                mask = tf.one_hot(mask, 1, on_value=0, off_value=1)
            # mask = mask.astype(np.uint8)

            images.append(image)
            masks.append(mask)

        images = np.array(images)
        masks = np.array(masks)

        # return images/np.max(images), masks
        return images, masks
