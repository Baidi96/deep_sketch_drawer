import scipy.misc
import numpy as np
import random
import argparse
import os
def read(folder, selectedImage, type):
    if type == 'photo':
        #read in the photo
        path = folder + "/" + selectedImage
        image = scipy.misc.imread(path)
        image = scipy.misc.imresize(image, (64, 64, 3))
        img = np.zeros((64, 64, 3))
        img[:, :, :] = image#/255.0
    elif type == 'sketch':
        #read in the sketch, using the first valid sketch
        path = folder + "/" + selectedImage.split(".")[0]
        allSketch = os.listdir(folder)
        for i in range(1,10):
            #find the first valid sketch
            temp = path.split("/")[-1] + "-" + str(i) + ".png"
            if temp in allSketch:
                break
            return None
        path += ("-" + str(i) + ".png")
        image = scipy.misc.imread(path, "L")
        image = scipy.misc.imresize(image, (64, 64, 1))
        img = np.zeros((64, 64, 1))
        #turn 256*256 to 256*256*1
        img[:,:,0]= image/255.0
    return img

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ID", type=str, default="baby", help="specifies a model ID")
    parser.add_argument("--mode", type=str, default="train", help="train|test mode")
    parser.add_argument("--image", type=str, default=None, help="path for train images")
    parser.add_argument("--sketch", type=str, default=None, help="path for train sketches")
    parser.add_argument("--result", type=str, default=None, help="path for storing results")
    parser.add_argument("--restore", type=str, default=None, help="the iteration number of the checkpoint file i.e: 500")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
    parser.add_argument("--iter", type=int, default=100, help = "number of iterations")
    parser.add_argument("--record", type=int, default=2000, help="saves every % iterations")
    parser.add_argument("--learning_rate", type=float, default=0.001, help = "learning rate to start")
    args = parser.parse_args()
    return args