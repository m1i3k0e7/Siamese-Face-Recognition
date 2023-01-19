import tensorflow
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from os import listdir
import random
import cv2
import numpy as np

def activation(input):
    return 1.0 / (1.0 + tensorflow.math.pow(8.0, -input))

def cosine_similarity(vectors):
    (featA, featB) = vectors
    dot_product = tensorflow.reduce_sum(tensorflow.multiply(featA, featB), axis=1, keepdims=True)
    norm_A = tensorflow.sqrt(tensorflow.reduce_sum(tensorflow.square(featA), axis=1, keepdims=True))
    norm_B = tensorflow.sqrt(tensorflow.reduce_sum(tensorflow.square(featB), axis=1, keepdims=True))
    return dot_product / (norm_A * norm_B)

def extractor(input_shape):
    base_model = Xception(include_top=False, input_shape=input_shape, weights=None)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024)(x)
    x = Dense(256)(x)
    outputs = Dense(128)(x)
    model = Model(base_model.input, outputs)
    return model

def siamese_network(input_shape):
    feature_extractor = extractor(input_shape)
    face1 = Input(shape=input_shape)
    face2 = Input(shape=input_shape)
    feature1 = feature_extractor(face1)
    feature2 = feature_extractor(face2)
    similarity = Lambda(cosine_similarity)([feature1, feature2])
    outputs = Dense(1, activation=activation)(similarity)
    siamese = Model(inputs=[face1, face2], outputs=outputs)
    return siamese