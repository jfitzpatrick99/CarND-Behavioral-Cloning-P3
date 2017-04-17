# model.py - Train a neural network to drive a car in a simulator.
# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import sys
import csv
import cv2
import numpy as np
from os import path
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, \
    Dropout
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt


def csv_log_to_image_filename(data_dir, csv_filename):
    """
    Convert an image filename is the csv log to a relative filename on disk.

    `data_dir` The directory containing the collected data
    `csv_filename` The name of the csv log file
    """
    image_file = csv_filename.strip()
    image_file_parts = image_file.split("/")
    rel_image_file = \
        image_file_parts[len(image_file_parts)-2:len(image_file_parts)]
    return path.join(data_dir, "/".join(rel_image_file))


def generator(data_dir, samples, batch_size=32):
    """
    Generator function that returns data on-demand to train the network.

    `data_dir` Directory containing the data
    `samples` List of samples where each sample is a tuple of image filename and
    steering angle.
    `batch_size` Number of samples to yeild on each call to the generator

    A generator function is used to avoid loading the entire dataset into
    memory.
    """
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                filename = csv_log_to_image_filename(data_dir,
                                                     batch_sample[0])
                image = cv2.imread(filename)
                if image is not None:
                    images.append(image)
                    measurements.append(batch_sample[1])
                else:
                    print("File " + filename + " is missing.")

            X_data = np.array(images)
            y_data = np.array(measurements)
            yield sklearn.utils.shuffle(X_data, y_data)


#
# Program entry point.
#
# Usage:
#
# python model.py <data_dir>
#
# where <data_dir> is the directory containing the data to use to train the
# network. The trained model will be written to a file called "model.hd5" in the
# current working directory.
#
# The network architecture used is described at:
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
#
def main(args=None):
    if args is None:
        args = sys.argv[1:]

    print("Loading csv data...")

    # Steering correction angle to use for data collected from the left and
    # right cameras.
    steering_correction = 0.2

    data_dir = args[0]

    # Populate the samples list with tuples of the form, (<image_filename>,
    # <steering_angle>).
    samples = []
    with open(path.join(data_dir, "driving_log.csv")) as csvfile:
        reader = csv.reader(csvfile)
        first_line = True
        for line in reader:
            if first_line:
                first_line = False
                continue

            measurement = float(line[3])
            samples.append([line[0], measurement])
            samples.append([line[1], measurement + steering_correction])
            samples.append([line[2], measurement - steering_correction])

    # Setup generators for training and validation data.
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(data_dir, train_samples, batch_size=32)
    validation_generator = generator(data_dir,
                                     validation_samples, batch_size=32)

    # Train the network.

    # Value to use for dropout layers
    dropout_rate = 0.5

    print("Training...")
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)),
                         input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(dropout_rate))
    model.add(Dense(50))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    model.compile(loss="mse", optimizer="adam")
    history = model.fit_generator(train_generator,
                                  samples_per_epoch=len(train_samples),
                                  validation_data=validation_generator,
                                  nb_val_samples=len(validation_samples),
                                  nb_epoch=3)

    model.save("model.h5")

    # Plot the training and validation loss
    # This code taken from the classroom example
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig("training_validation_loss_plot.jpg")

    print("All done.")

if __name__ == "__main__":
    main()
