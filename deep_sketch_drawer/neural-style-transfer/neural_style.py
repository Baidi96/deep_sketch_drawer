# Copyright (c) 2015-2018 Anish Athalye. Released under GPLv3.

import os

import numpy as np
import scipy.misc

from stylize import stylize
import cv2
import math
from argparse import ArgumentParser

from PIL import Image
import os
from produceHED import produceEdge

# default arguments
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 80
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 1000
VGG_PATH = 'model/imagenet-vgg-verydeep-19.mat'
POOLING = 'max'

SKETCH_DIRECTORY = 'image_data/sketches_for_training/'
EDGE_DIRECTORY = 'image_data/intermediate_edge_images/'
OUTPUT_SKETCH_DIRECTORY = 'output/'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--input',
            dest='input', help='edge images',
            metavar='EDGE_DIRECTORY', default=EDGE_DIRECTORY)
    parser.add_argument('--output',
            dest='output', help='generate images',
            metavar='OUTPUT_SKETCH_DIRECTORY', default=OUTPUT_SKETCH_DIRECTORY)
    return parser
   

def transfer_image(image):
    image[:] = image.mean(axis=-1,keepdims=1)
    for i in range(len(image)):
        for j in range(len(image[i])):
            for k in range(len(image[i][j])):
                if image[i][j][k] < 100:
                    image[i][j][k] = 0
                else:
                    image[i][j][k] = 255
    blured = cv2.blur(image,(3,3)) 
    return blured

def create_sketch(content_img, out_name):
    ooutput = out_name
    if not os.path.isfile(VGG_PATH):
        parser.error("Network does not exist. (Did you forget to download it?)")

    content_image = imread(content_img)
    ostyles = []
    files = os.listdir(SKETCH_DIRECTORY)
    for f in files:
        ostyles.append(SKETCH_DIRECTORY + f)  
    style_images = [imread(style) for style in ostyles]

    width = None
    if width is not None:
        new_shape = (int(math.floor(float(content_image.shape[0]) /
                content_image.shape[1] * width)), width)
        content_image = scipy.misc.imresize(content_image, new_shape)
    target_shape = content_image.shape
    
    for i in range(len(style_images)):
        style_scale = STYLE_SCALE
     #   if options.style_scales is not None:
     #       style_scale = options.style_scales[i]
     #   style_images[i] = scipy.misc.imresize(style_images[i], style_scale *
     #           target_shape[1] / style_images[i].shape[1])
    
    style_blend_weights = None
    if style_blend_weights is None:
        # default is equal weights
        style_blend_weights = [1.0/len(style_images) for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight/total_blend_weight
                               for weight in style_blend_weights]
    
    initial = None

    try:
        imsave(ooutput, np.zeros((500, 500, 3)))
    except:
        raise IOError('Not writable or does not have a valid file extension for an image file')

    for iteration, image in stylize(
        network=VGG_PATH,
        initial=initial,
        initial_noiseblend=1.0,
        content=content_image,
        styles=style_images,
        preserve_colors=None,
        iterations=ITERATIONS,
        content_weight=CONTENT_WEIGHT,
        content_weight_blend=CONTENT_WEIGHT_BLEND,
        style_weight=STYLE_WEIGHT,
        style_layer_weight_exp=STYLE_LAYER_WEIGHT_EXP,
        style_blend_weights=style_blend_weights,
        tv_weight=TV_WEIGHT,
        learning_rate=LEARNING_RATE,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        pooling=POOLING,
        print_iterations=None,
        checkpoint_iterations=None
    ):
        output_file = None
        combined_rgb = image
        if iteration is None:
            output_file = ooutput
        if output_file:
            imsave(output_file, transfer_image(combined_rgb))


def main():
    produceEdge()
    path = EDGE_DIRECTORY
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                #print(os.path.splitext(file)[0]
                im = Image.open(file)
                im.save(os.path.splitext(file)[0] + '.jpg')
    parser = build_parser()
    options = parser.parse_args()
    file_dir = options.input
    for file in os.listdir(file_dir):
        content_image = file_path = os.path.join(file_dir, file)  
        out_path = options.output + file
        create_sketch(content_image, out_path)  


def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

if __name__ == '__main__':
    main()
