import sys
import csv
import cv2
import numpy as np
from os import path
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from sklearn.model_selection import train_test_split
import sklearn


def csv_log_to_image_filename(data_dir, csv_filename):
    image_file = csv_filename.strip()
    image_file_parts = image_file.split("/")
    rel_image_file = \
        image_file_parts[len(image_file_parts)-2:len(image_file_parts)]
    return path.join(data_dir, "/".join(rel_image_file))


def generator(data_dir, samples, batch_size=32):
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


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    print("Loading csv data...")

    steering_correction = 0.2

    data_dir = args[0]
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

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    train_generator = generator(data_dir, train_samples, batch_size=32)
    validation_generator = generator(data_dir,
                                     validation_samples, batch_size=32)

    print("Training...")
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)),
                         input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss="mse", optimizer="adam")
    model.fit_generator(train_generator,
                        samples_per_epoch=len(train_samples),
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_samples),
                        nb_epoch=3)

    model.save("model.hd5")
    print("All done.")

if __name__ == "__main__":
    main()
