import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
import wandb
import tensorflow as tf


def load_img(flair_list, t1ce_list, t2_list, channels):
    scaler = MinMaxScaler()
    images = []
    for flair_name, t1ce_name, t2_name in zip(flair_list, t1ce_list, t2_list):
        # normalisation based on https://stackoverflow.com/a/59601298
        flair = nib.load(flair_name).get_fdata()
        flair = scaler.fit_transform(flair.reshape(-1, flair.shape[-1])).reshape(flair.shape)
        t1ce = nib.load(t1ce_name).get_fdata()
        t1ce = scaler.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)
        t2 = nib.load(t2_name).get_fdata()
        t2 = scaler.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)

        if channels == 3:
            image = np.stack([flair, t1ce, t2], axis=3)
        else:
            image = np.stack([flair, t1ce], axis=3)
        image = image[56:184, 56:184, 13:141]

        images.append(image)

    images = np.array(images)

    return images


def load_mask(mask_list, segmenting_subregion=0):
    images = []
    for mask_name in mask_list:
        temp_mask = nib.load(mask_name).get_fdata()
        #temp_mask = temp_mask.astype(np.uint8)
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
        # print(np.unique(temp_mask))

        if segmenting_subregion == 0:
            image = to_categorical(image, num_classes=4)
        else:
            #image = to_categorical(image, num_classes=2)
            image = tf.one_hot(image, 1, on_value=0, off_value=1)

        images.append(image)

    images = np.array(images)

    return images


def image_loader(flair_list, t1ce_list, t2_list, mask_list, batch_size, channels=3, segmenting_subregion=0):
    img_len = len(flair_list)

    # keras needs the generator infinite
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < img_len:
            limit = min(batch_end, img_len)
            x = load_img(flair_list[batch_start:limit], t1ce_list[batch_start:limit], t2_list[batch_start:limit], channels)
            y = load_mask(mask_list[batch_start:limit], segmenting_subregion)

            yield (x, y)  # a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size


def callback(segmenting_subregion=''):
    csv_logger = CSVLogger(f'outputs/training_{segmenting_subregion}.log', separator=',', append=False)

    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001, verbose=1),
                 csv_logger]
    '''
    callbacks = [EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=1, mode='auto'),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001, verbose=1),
                 csv_logger]
    '''
    return callbacks


# https://wandb.ai/ayush-thakur/image-segmentation/reports/Image-Segmentation-Using-Keras-and-W-B--VmlldzoyNTE1Njc
class SemanticLogger(tf.keras.callbacks.Callback):
    def __init__(self, testloader):
        super(SemanticLogger, self).__init__()
        self.val_images, self.val_masks = next(iter(testloader))
        self.segmentation_classes = ['background', 'necrotic', 'edema', 'enhancing']

    # returns a dictionary of labels
    def labels(self):
        l = {}
        for i, label in enumerate(self.segmentation_classes):
            l[i] = label
        return l

    # util function for generating interactive image mask from components
    def wandb_mask(self, bg_img, pred_mask, true_mask):
        return wandb.Image(bg_img, masks={
            "prediction": {
                "mask_data": pred_mask,
                "class_labels": self.labels()
            },
            "ground truth": {
                "mask_data": true_mask,
                "class_labels": self.labels()
            }
        })

    def on_epoch_end(self, logs, epoch):
        pred_masks = self.model.predict(self.val_images)
        pred_masks = np.argmax(pred_masks, axis=-1)

        val_images = tf.image.convert_image_dtype(self.val_images, tf.uint8)
        val_masks = tf.image.convert_image_dtype(self.val_masks, tf.uint8)
        #val_masks = tf.squeeze(val_masks, axis=-1)

        pred_masks = tf.image.convert_image_dtype(pred_masks, tf.uint8)

        mask_list = []
        for i in range(len(self.val_images)):
            mask_list.append(self.wandb_mask(val_images[i].numpy(),
                                        pred_masks[i].numpy(),
                                        val_masks[i].numpy()))

        wandb.log({"predictions": mask_list})
