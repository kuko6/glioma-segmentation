import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import cv2

# -------------------------------------------------------------------------------- #
# Additional functions used for data processing or loading.
# Includes:
#   - custom cmap used for mask visualization
#   - functions used for loading images and masks and their helper functions
# -------------------------------------------------------------------------------- #

def get_custom_cmap():
    colorarray = [
        [0/256, 0/256, 0/256, 1], # Background
        #[200/256, 192/256, 249/256, 1], # Background
        [105/256, 173/256, 212/256, 1], # Necrotic
        [114/256, 195/256, 116/256, 1], # Edema
        [254/256, 249/256, 9/256, 1], # Enhancing
    ]
    cmap = ListedColormap(colorarray)

    return cmap


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


def load_img(flair_list, t1ce_list, t2_list, t1_list, img_channels):
    images = []
    for flair_name, t1ce_name, t2_name, t1_name in zip(flair_list, t1ce_list, t2_list, t1_list):
        flair = nib.load(flair_name).get_fdata()
        t1ce = nib.load(t1ce_name).get_fdata()
        t2 = nib.load(t2_name).get_fdata()
        t1 = nib.load(t1_name).get_fdata()

        # crop the images and mask
        left_x, right_x, top_y, bottom_y = get_dimensions(flair[:, :, 0])
        flair = flair[left_x:right_x, top_y:bottom_y, :]
        t1ce = t1ce[left_x:right_x, top_y:bottom_y, :]
        t2 = t2[left_x:right_x, top_y:bottom_y, :]
        t1 = t1[left_x:right_x, top_y:bottom_y, :]

        volume_start = 13
        tmp_flair = np.zeros((128, 128, 128))
        tmp_t1ce = np.zeros((128, 128, 128))
        tmp_t2 = np.zeros((128, 128, 128))
        tmp_t1 = np.zeros((128, 128, 128))

        # resize the images
        inter = cv2.INTER_NEAREST
        for i in range(128):
            tmp_flair[:,:,i] = cv2.resize(flair[:,:,i+volume_start], (128, 128), interpolation=inter)
            tmp_t1ce[:,:,i] = cv2.resize(t1ce[:,:,i+volume_start], (128, 128), interpolation=inter)
            tmp_t2[:,:,i]= cv2.resize(t2[:,:,i+volume_start], (128, 128), interpolation=inter)
            tmp_t1[:,:,i]= cv2.resize(t1[:,:,i+volume_start], (128, 128), interpolation=inter)
        flair = tmp_flair
        t1ce = tmp_t1ce
        t2 = tmp_t2
        t1 = tmp_t1

        # normalise
        flair = normalise(flair)
        t1ce = normalise(t1ce)
        t2 = normalise(t2)
        t1 = normalise(t1)

        # stack the sequences
        if img_channels == 4:
            image = np.stack([flair, t1ce, t2, t1], axis=3)
        elif img_channels == 3:
            image = np.stack([flair, t1ce, t2], axis=3)
        elif img_channels == 2:
            image = np.stack([flair, t1ce], axis=3)
        else:
            image = np.stack([flair], axis=3)

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