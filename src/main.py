import argparse
# from azureml.core import Run

import os
import glob
import segmentation_models_3D as sm
import tensorflow as tf
import pandas as pd
import wandb
from wandb.keras import WandbCallback

import utils
import losses
import unet
from generator import BratsGen


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
    parser.add_argument('--wandb', type=str, help='wandb id')
    args = parser.parse_args()

    wandb_key = args.wandb
    wandb.login(key=wandb_key)
    # wandb.init(project="BraTS2021", entity="kuko")
    wandb.config = {
        "num_classes": 4,
        "img_channels": 2,
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 2,
        "loss": "categorical_crossentropy",
        "optimizer": "adam",
        "dataset": "BraTS2021"
    }
    config = wandb.config

    data = args.data_path
    print(os.listdir(data))
    training_path = os.path.join(data, 'train/')
    testing_path = os.path.join(data, 'test/')
    validation_path = os.path.join(data, 'val/')

    if not os.path.isdir('outputs'):
        os.mkdir('outputs')

    train_t2_list = sorted(glob.glob(training_path + '/*/*t2.nii.gz'))
    train_t1ce_list = sorted(glob.glob(training_path + '/*/*t1ce.nii.gz'))
    train_flair_list = sorted(glob.glob(training_path + '/*/*flair.nii.gz'))
    train_mask_list = sorted(glob.glob(training_path + '/*/*seg.nii.gz'))

    val_t2_list = sorted(glob.glob(validation_path + '/*/*t2.nii.gz'))
    val_t1ce_list = sorted(glob.glob(validation_path + '/*/*t1ce.nii.gz'))
    val_flair_list = sorted(glob.glob(validation_path + '/*/*flair.nii.gz'))
    val_mask_list = sorted(glob.glob(validation_path + '/*/*seg.nii.gz'))

    batch_size = config["batch_size"]
    if config['num_classes'] == 4:
        subregions = [0]
    else:
        # subregions = [1, 2, 3]
        subregions = [2]

    for subregion in subregions:
        run = wandb.init(project="BraTS2021",
                         name=f"{config['loss']}_{config['epochs']}_{config['img_channels']}ch_sub{subregion}",
                         entity="kuko",
                         reinit=True)

        train_img_datagen = BratsGen(train_flair_list, train_t1ce_list, train_t2_list, train_mask_list,
                                     (128, 128, 128), img_channels=config['img_channels'], classes=config['num_classes'],
                                     segmenting_subregion=subregion)

        val_img_datagen = BratsGen(val_flair_list, val_t1ce_list, val_t2_list, val_mask_list, (128, 128, 128),
                                   img_channels=config['img_channels'], classes=config['num_classes'],
                                   segmenting_subregion=subregion)
        '''
        train_img_datagen = utils.image_loader(train_flair_list, train_t1ce_list, train_t2_list, train_mask_list,
                                               batch_size=batch_size, channels=config['img_channels'],
                                               segmenting_subregion=subregion)

        val_img_datagen = utils.image_loader(val_flair_list, val_t1ce_list, val_t2_list, val_mask_list,
                                             batch_size=batch_size, channels=config['img_channels'],
                                             segmenting_subregion=subregion)
        '''
        if config['num_classes'] == 4:
            metrics = [sm.metrics.IOUScore(threshold=0.5),
                       tf.keras.metrics.MeanIoU(num_classes=4),
                       losses.dice_coef, losses.dice_coef2, losses.dice_coef_necrotic,
                       losses.dice_coef_edema, losses.dice_coef_enhancing]
        else:
            metrics = [sm.metrics.IOUScore(threshold=0.5),
                       tf.keras.metrics.MeanIoU(num_classes=2),
                       losses.dice_coef, losses.dice_coef2]

        LR = config["learning_rate"]
        optim = tf.keras.optimizers.Adam(LR)
        steps_per_epoch = len(train_flair_list) // batch_size
        val_steps_per_epoch = len(val_flair_list) // batch_size

        model = unet.unet_model(img_height=128, img_width=128, img_depth=128,
                                img_channels=config["img_channels"], num_classes=config["num_classes"])

        if config['num_classes'] == 4:
            # model.compile(optimizer=optim, loss=losses.loss(), metrics=metrics)
            model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=metrics)
        else:
            model.compile(optimizer=optim, loss="binary_crossentropy", metrics=metrics)

        history = model.fit(train_img_datagen,
                            steps_per_epoch=steps_per_epoch,
                            epochs=config["epochs"],
                            verbose=1,
                            callbacks=[utils.callback(segmenting_subregion=subregion),
                                       WandbCallback()],
                            validation_data=val_img_datagen,
                            validation_steps=val_steps_per_epoch)

        model.save(f'outputs/model_{subregion}.h5')

        # Saves history as dictionary
        ''''
        with open(f'outputs/train_history_{subregion}', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        '''

        hist_df = pd.DataFrame(history.history)

        # Saves history as .csv
        hist_csv_file = f'outputs/history_{subregion}.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        run.finish()

    # wandb.finish()
    '''
    training_files = list_files(training_path)
    validation_files = list_files(validation)
    testing_files = list_files(testing)

    print(f'training_path: {len(training_files)}, validation: {len(validation_files)}, testing: {len(testing_files)}')
    '''


if __name__ == '__main__':
    main()
