import sys
import csv
import cv2
import numpy as np
from os import path
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    print("Loading data...")

    steering_correction = 0.2

    images = []
    measurements = []

    with open(path.join(args[0], "driving_log.csv")) as csvfile:
        reader = csv.reader(csvfile)
        first_line = True
        for line in reader:
            if first_line:
                first_line = False
                continue

            image = cv2.imread(path.join(args[0], line[0].strip()))
            measurement = float(line[3])

            images.append(image)
            measurements.append(measurement)

            left_image = cv2.imread(path.join(args[0], line[1].strip()))
            images.append(left_image)
            measurements.append(measurement + steering_correction)

            right_image = cv2.imread(path.join(args[0], line[2].strip()))
            images.append(right_image)
            measurements.append(measurement - steering_correction)

    print("Augmenting dataset...")
    augmented_images = []
    augmented_measurements = []
    for image, measurement in zip(images, measurements):
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement * -1.0)

    images.extend(augmented_images)
    measurements.extend(augmented_measurements)

    print("Converting data to numpy arrays...")
    X_train = np.array(images)
    y_train = np.array(measurements)

    print("Training...")
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)),
                         input_shape=X_train[0].shape))
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
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

    model.save("model.hd5")
    print("All done.")

if __name__ == "__main__":
    main()
