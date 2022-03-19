import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau


def load_img(flair_list, t1ce_list, t2_list):
    scaler = MinMaxScaler()
    images = []
    for flair_name, t1ce_name, t2_name in zip(flair_list, t1ce_list, t2_list):
        temp_image_flair = nib.load(flair_name).get_fdata()
        # print(np.max(temp_image_flair))
        temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(
            temp_image_flair.shape)
        # print("========================================")
        # print(np.max(temp_image_flair))
        temp_image_t1ce = nib.load(t1ce_name).get_fdata()
        temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(
            temp_image_t1ce.shape)

        temp_image_t2 = nib.load(t2_name).get_fdata()
        temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(
            temp_image_t2.shape)

        image = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
        image = image[56:184, 56:184, 13:141]

        images.append(image)

    images = np.array(images)

    return images


def load_mask(mask_list, segmenting_subregion):
    images = []
    for mask_name in mask_list:
        temp_mask = nib.load(mask_name).get_fdata()
        temp_mask = temp_mask.astype(np.uint8)
        temp_mask[temp_mask == 4] = 3  # Reassign mask values 4 to 3

        # segmenting label 1
        if segmenting_subregion == 1:
            temp_mask[temp_mask == 2] = 0
            temp_mask[temp_mask == 3] = 0
        # segmenting label 2
        elif segmenting_subregion == 2:
            temp_mask[temp_mask == 1] = 0
            temp_mask[temp_mask == 3] = 0
            temp_mask[temp_mask == 2] = 1
        # segmenting label 3
        elif segmenting_subregion == 3:
            temp_mask[temp_mask == 1] = 0
            temp_mask[temp_mask == 2] = 0
            temp_mask[temp_mask == 3] = 1
        image = temp_mask[56:184, 56:184, 13:141]

        #print(np.unique(temp_mask))

        if segmenting_subregion == 0:
            image = to_categorical(image, num_classes=4)
        else:
            image = to_categorical(image, num_classes=2)

        images.append(image)

    images = np.array(images)

    return images


def image_loader(flair_list, t1ce_list, t2_list, mask_list, batch_size, segmenting_subregion):
    img_len = len(flair_list)

    # keras needs the generator infinite
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < img_len:
            limit = min(batch_end, img_len)
            x = load_img(flair_list[batch_start:limit], t1ce_list[batch_start:limit], t2_list[batch_start:limit])
            y = load_mask(mask_list[batch_start:limit], segmenting_subregion)

            yield x, y  # a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size


def callback(segmenting_subregion=''):
    csv_logger = CSVLogger(f'outputs/training_{segmenting_subregion}.log', separator=',', append=False)

    callbacks = [EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=1, mode='auto'),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001, verbose=1),
                 csv_logger]

    return callbacks
