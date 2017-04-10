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
    steering_correction = 0.2
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                measurement = float(batch_sample[3])

                center_file = csv_log_to_image_filename(data_dir,
                                                        batch_sample[0])
                image = cv2.imread(center_file)
                if image is not None:
                    images.append(image)
                    measurements.append(measurement)
                else:
                    print("File " + center_file + " is missing.")

                left_file = csv_log_to_image_filename(data_dir,
                                                      batch_sample[1])
                left_image = cv2.imread(left_file)
                if left_image is not None:
                    images.append(left_image)
                    measurements.append(measurement + steering_correction)
                else:
                    print("File " + left_file + " is missing.")

                right_file = csv_log_to_image_filename(data_dir,
                                                       batch_sample[2])
                right_image = cv2.imread(right_file)
                if right_image is not None:
                    images.append(right_image)
                    measurements.append(measurement - steering_correction)
                else:
                    print("File " + right_file + " is missing.")

            X_data = np.array(images)
            y_data = np.array(measurements)
            yield sklearn.utils.shuffle(X_data, y_data)


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    print("Loading csv data...")

    data_dir = args[0]
    lines = []
    with open(path.join(data_dir, "driving_log.csv")) as csvfile:
        reader = csv.reader(csvfile)
        first_line = True
        for line in reader:
            if first_line:
                first_line = False
                continue
            lines.append(line)

    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

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
