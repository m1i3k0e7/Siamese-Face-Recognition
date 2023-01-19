from model import siamese_network
from data import create_train_and_test
import argparse
import tensorflow
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from os import listdir
import random
import cv2
import numpy as np

def main(config):
    input_shape = (config.input_size, config.input_size, 3)
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    epochs = config.epochs
    data_path = config.data_path
    checkpoint_path = config.checkpoint_path

    model = siamese_network(input_shape)
    model.compile(loss="binary_crossentropy", optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
    model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    x_train, y_train, x_test, y_test = create_train_and_test(data_path, input_shape)
    x_valid = x_test[:len(x_test)//2]
    y_valid = y_test[:len(y_test)//2]
    x_test = x_test[len(x_test)//2:]
    y_test = y_test[len(y_test)//2:]

    history = model.fit([x_train[:, 0], x_train[:, 1]], y_train[:], validation_data=([x_valid[:,0], x_valid[:,1]], y_valid[:]), batch_size=batch_size, epochs=epochs, callbacks=[model_checkpoint_callback])
    model.load_weights(checkpoint_path)

    loss, acc = model.evaluate([x_test[:, 0], x_test[:, 1]], y_test[:], verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate of model')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
    parser.add_argument('--input_size', type=int, default=128, help='size of input image')
    parser.add_argument('--epochs', type=int, default=40, help='number of training epochs')
    parser.add_argument('--data_path', type=str, default='path/to/training/data', help='path to training data')
    parser.add_argument('--checkpoint_path', type=str, default='path/to/save/weights', help='path to save trained weights')

    config = parser.parse_args()
    main(config)