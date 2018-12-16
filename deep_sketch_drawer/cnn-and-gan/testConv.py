#modified based on https://github.com/senliuy/Keras_HED_with_model/blob/master/test.py
import os
from models.hed import hed
from models.hed_add_layers import convModel
from keras.utils import plot_model
from keras import backend as K
from keras import callbacks
import numpy as np
import glob
from PIL import Image
import cv2


def test(inputFolder, outputFolder, modelFile, image_size = 256, output_size = 256):
    #environment
    print("here")
    test = glob.glob(inputFolder + '/*')
    K.set_image_data_format('channels_last')
    K.image_data_format()
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    model = hed()
    model.load_weights('./' + modelFile)
    count = 0
    countSkip = 0
    for image in test:
        print("extracting edges from" + image)
        name = image.split('/')[-1]
        x_batch = []
        im = cv2.imread(image)
        (h,w) = im.shape[:2]
        im.resize((image_size, image_size, 3))
        im = np.array(im, dtype=np.float32)
        R = im[..., 0].mean()
        G = im[..., 1].mean()
        B = im[..., 2].mean()
        im[..., 0] -= R
        im[..., 1] -= G
        im[..., 2] -= B
        x_batch.append(im)
        x_batch = np.array(x_batch, np.float32)
        prediction = model.predict(x_batch)
        mask = np.zeros_like(im[:,:,0])
        for i in range(len(prediction)):
            mask += np.reshape(prediction[i],(output_size,output_size))
        ret,out_mask = cv2.threshold(mask,np.mean(mask)+1.2*np.std(mask),255,cv2.THRESH_BINARY_INV)
        print(outputFolder + "/"+name)
        cv2.imwrite(outputFolder + "/"+name.split("/")[-1], out_mask)


