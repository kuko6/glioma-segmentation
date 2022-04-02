import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random


# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class BratsGen(tf.keras.utils.Sequence):
    def __init__(self, flair_list, t1ce_list, t2_list, mask_list, img_dim=(128, 128, 128),
                 img_channels=3, classes=4, batch_size=2, segmenting_subregion=0):
        self.batch_size = batch_size
        self.classes = classes
        self.img_channels = img_channels
        self.img_dim = img_dim
        self.flair_list = flair_list
        self.t1ce_list = t1ce_list
        self.t2_list = t2_list
        self.mask_list = mask_list
        self.segmenting_subregion = segmenting_subregion

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_flair = self.flair_list[i: i + self.batch_size]
        batch_t1ce = self.t1ce_list[i: i + self.batch_size]
        batch_t2 = self.t2_list[i: i + self.batch_size]
        batch_mask = self.mask_list[i: i + self.batch_size]

        X, y = self.__data_generation(batch_flair, batch_t1ce, batch_t2, batch_mask)

        return X, y

    # normalisation based on https://stackoverflow.com/a/59601298
    def __normalise(self, image):
        scaler = MinMaxScaler()
        image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
        return image

    def __data_generation(self, flair_list, t1ce_list, t2_list, mask_list):
        # X = np.zeros((self.batch_size*self.img_dim[2], self.img_dim[0], self.img_dim[1], self.img_channels))
        # y = np.zeros((self.batch_size*self.img_dim[2], self.img_dim[0], self.img_dim[1]))
        # Y = np.zeros((self.batch_size*self.img_dim[2], self.img_dim[0], self.img_dim[1], 4))

        images = []
        masks = []
        for flair_name, t1ce_name, t2_name, mask_name in zip(flair_list, t1ce_list, t2_list, mask_list):
            # ==================== Sequences ====================#
            flair = nib.load(flair_name).get_fdata()
            t1ce = nib.load(t1ce_name).get_fdata()
            t2 = nib.load(t2_name).get_fdata()
            mask = nib.load(mask_name).get_fdata()

            # normalise
            flair = self.__normalise(flair)
            t1ce = self.__normalise(t1ce)
            t2 = self.__normalise(t2)

            # stack the sequences
            if self.img_channels == 3:
                image = np.stack([flair, t1ce, t2], axis=3)
            else:
                image = np.stack([flair, t1ce], axis=3)

            # crop the image to 128x128x128
            image = image[56:184, 56:184, 13:141]

            # ==================== Mask ====================#
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

            # crop the mask to 128x128x128
            mask = mask[56:184, 56:184, 13:141]

            # encode
            if self.segmenting_subregion == 0:
                mask = to_categorical(mask, num_classes=4)
            else:
                # mask = to_categorical(mask, num_classes=2)
                mask = tf.one_hot(mask, 1, on_value=0, off_value=1)

            images.append(image)
            masks.append(mask)

            # X[j+(VOLUME_SLICES*c),:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
            # X[j+(VOLUME_SLICES*c),:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
            # y[j +VOLUME_SLICES*c,:,:] = cv2.resize(seg[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))

        images = np.array(images)
        masks = np.array(masks)

        # return images/np.max(images), masks
        return images, masks
