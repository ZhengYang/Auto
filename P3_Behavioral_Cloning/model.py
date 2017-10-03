import argparse
import os
import os.path as path
import glob

import numpy as np
import pandas as pd
import cv2

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Cropping2D, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


def train(train_samples, valid_samples, nb_epoch=3, batch_size=32, model_file_name='model.h5'):
    '''
    Train models using the generators.
    Save the model to a file after the trainings.
    '''

    train_generator = generator(train_samples, batch_size=batch_size)
    valid_generator = generator(valid_samples, batch_size=batch_size)

    # MODEL
    model = Sequential()

    # Preprocessing Layer
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    # Conv Layers
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    # FC Layers
    model.add(Flatten())

    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                        validation_data=valid_generator, nb_val_samples=len(valid_samples),
                        nb_epoch=nb_epoch)
    model.save(model_file_name)


def generator(samples, batch_size=32):
    '''
    Create a generator of samples, each emit will be in the size of the batch_size.
    '''
    num_samples = len(samples)
    while True:
        shuffled_samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = shuffled_samples[offset:offset+batch_size]
            images, angles = [], []
            for batch_sample in batch_samples:
                center_image = read_img(batch_sample['center_img'])
                center_angle = float(batch_sample['steer_angle'])
                images.append(center_image)
                angles.append(center_angle)
            X_train, y_train = np.array(images), np.array(angles)
            yield shuffle(X_train, y_train)


def read_img(img_path):
    '''
    Read image in RGB format.
    '''
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        print('Error loading image:', img_path)
        raise
    return img


def extract_samples(data_folder):
    '''
    Extracts center, left, righ image path and steering angles from the label file.
    Each sub-folder in data_folder will be merged into one big dataset.
    Correct the absolute path when data was gathered into correct relative path.
    Return the repackaged clean data.
    '''
    subfolders = glob.glob(path.join(data_folder, '*'))
    
    samples = []
    for subfolder in subfolders:
        image_base = path.join(subfolder, 'IMG')
        label_path = path.join(subfolder, 'driving_log.csv')

        samples_df = pd.read_csv(label_path, names=['center_img', 'left_img', 'right_img', 'steer_angle', 'throttle', 'break', 'speed'])

        for _, row in samples_df.iterrows():
            # original image path
            orig_center = row['center_img']
            orig_left   = row['left_img']
            orig_right  = row['right_img']
            orig_angle  = row['steer_angle'] 
            # extract image names
            base_center = path.basename(orig_center)
            base_left   = path.basename(orig_left) 
            base_right  = path.basename(orig_right)
            # prepend image path
            center = path.join(image_base, base_center)
            left   = path.join(image_base, base_left)
            right  = path.join(image_base, base_right)
            # repackage the cleaned data into samples
            sample = {}
            sample['center_img'] = center
            sample['left_img'] = left
            sample['right_img'] = right
            sample['steer_angle'] = orig_angle
            samples.append(sample)
           
    return samples

    
def augment_data(data_folder):
    '''
    Augmentation the data by flipping it horizontally
    '''
    subfolders = glob.glob(path.join(data_folder, '*'))
    subfolders = set(subfolders)
 
    # Data augmentation
    for subfolder in subfolders:
        # original data folder (without _aug suffix) that has not been augmented 
        if not subfolder.endswith('_aug') and (subfolder+'_aug' not in subfolders):
            os.makedirs(path.join(subfolder+'_aug', 'IMG'))

            image_base = path.join(subfolder, 'IMG')
            label_path = path.join(subfolder, 'driving_log.csv')

            samples_df = pd.read_csv(label_path, names=['center_img', 'left_img', 'right_img', 'steer_angle', 'throttle', 'break', 'speed'])
            samples_df['steer_angle'] = -samples_df['steer_angle']
            samples_df.to_csv(path.join(subfolder+'_aug', 'driving_log.csv'), header=False, index=False)

            for _, row in samples_df.iterrows():
                orig_center = row['center_img']
                base_center = path.basename(orig_center)
                center = path.join(image_base, base_center)
                center = cv2.imread(center)
                center_flipped = np.fliplr(center)
                cv2.imwrite(path.join(subfolder+'_aug', 'IMG', base_center), center_flipped)


def main():
    '''
    Extract training data generated by the driving simulator
    and train a CNN to clone the driver hehivour.
    '''
    parser = argparse.ArgumentParser(description='Train behaviour model.')
    parser.add_argument(
        '--data_folder',
        type=str,
        default='data',
        help='Path to training data folder. (It must contain IMG folder and driving_log.csv)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of epochs in training'
    )
    parser.add_argument(
        '--batchsize',
        type=int,
        default=32,
        help='Batch size in training'
    )
    args = parser.parse_args()
    # training parameters
    data_folder = args.data_folder
    nb_epoch = args.epochs
    batchsize = args.batchsize
    print('Training models in:', args.data_folder)
    print('Number of epochs:  ', args.epochs)
    print('Batch size:        ', args.batchsize)

    # prepare data and train!
    augment_data(data_folder)
    samples = extract_samples(data_folder)
    train_samples, valid_samples = train_test_split(samples, test_size=0.2)
    print('Data for training/validation: {}/{}'.format(len(train_samples), len(valid_samples)))

    train(train_samples, valid_samples, nb_epoch=nb_epoch, batch_size=batchsize)


if __name__ == '__main__':
    main()
