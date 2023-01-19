from model import *
from data import *
import tensorflow
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, Lambda
from tensorflow.keras.models import Model
import argparse
import cv2
import math

def convert(y):
  result = -math.log((1-y)/y)/math.log(8)
  if result > 1:
    return 1
  elif result < -1:
    return -1
  return result

def main(config):
    input_shape = (config.input_size, config.input_size, 3)
    img_path1 = config.img_path1
    img_path2 = config.img_path2
    checkpoint_path = config.checkpoint_path

    model = siamese_network(input_shape)
    model.load_weights(checkpoint_path)
    
    img1 = read_image(img_path1, input_shape)
    img2 = read_image(img_path2, input_shape)
    img1 = tensorflow.expand_dims(img1, axis=0)
    img2 = tensorflow.expand_dims(img2, axis=0)

    prediction = model.predict([img1, img2], verbose=0)[0][0]
    print('similarity: ', convert(prediction))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=128, help='size of input image')
    parser.add_argument('--checkpoint_path', type=str, default='', help='path to trained weights')
    parser.add_argument('--img_path1', type=str, default='', help='path to img1')
    parser.add_argument('--img_path2', type=str, default='', help='path to img2')

    config = parser.parse_args()
    main(config)
