#modified based on https://github.com/coreyauger/quickdraw-gan/blob/master/quickdraw.py
import os
import numpy as np
import cv2
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, Flatten, Concatenate
from keras.layers import Activation, Reshape, Conv2DTranspose, UpSampling2D, MaxPooling2D, Lambda
from keras.optimizers import RMSprop
from keras.backend import tf as ktf
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import unified_data_loader

def discriminator_builder(depth=64, p = 0.2):

    # Define inputs
    inputs = Input((28,28,1))
    # Convolutional layers
    conv1 = Conv2D(depth*1, 5, strides=2, padding='same', activation='relu')(inputs)
    conv1 = Dropout(p)(conv1)

    conv2 = Conv2D(depth*2, 5, strides=2, padding='same', activation='relu')(conv1)
    conv2 = Dropout(p)(conv2)

    conv3 = Conv2D(depth*4, 5, strides=2, padding='same', activation='relu')(conv2)
    conv3 = Dropout(p)(conv3)

    conv4 = Conv2D(depth*8, 5, strides=1, padding='same', activation='relu')(conv3)
    conv4 = Flatten()(Dropout(p)(conv4))

    output = Dense(1, activation='sigmoid')(conv4)

    model = Model(inputs=inputs, outputs=output)

    return model


def generator_builder(depth=64,p=0.4):

    # Define inputs
    inputs = Input((28,28,1))
    flat = Flatten()(inputs)

    # First dense layer
    dense1 = Dense(7*7*64)(flat)
    dense1 = BatchNormalization(axis=-1,momentum=0.9)(dense1)
    dense1 = Activation(activation='relu')(dense1)
    dense1 = Reshape((7,7,64))(dense1)
    dense1 = Dropout(p)(dense1)

    # Convolutional layers
    conv1 = UpSampling2D()(dense1)
    conv1 = Conv2DTranspose(int(depth/2), kernel_size=5, padding='same', activation=None,)(conv1)
    conv1 = BatchNormalization(axis=-1,momentum=0.9)(conv1)
    conv1 = Activation(activation='relu')(conv1)

    conv2 = UpSampling2D()(conv1)
    conv2 = Conv2DTranspose(int(depth/4), kernel_size=5, padding='same', activation=None,)(conv2)
    conv2 = BatchNormalization(axis=-1,momentum=0.9)(conv2)
    conv2 = Activation(activation='relu')(conv2)

    #conv3 = UpSampling2D()(conv2)
    conv3 = Conv2DTranspose(int(depth/8), kernel_size=5, padding='same', activation=None,)(conv2)
    conv3 = BatchNormalization(axis=-1,momentum=0.9)(conv3)
    conv3 = Activation(activation='relu')(conv3)

    # Define output layers
    output = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(conv3)

    # Model definition
    model = Model(inputs=inputs, outputs=output)

    return model


def adversarial_builder(generator, discriminator, gan_lr):

    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=gan_lr, clipvalue=1.0, decay=3e-8), metrics=['accuracy'])
    return model


def make_trainable(net, is_trainable):
    net.trainable = is_trainable
    for l in net.layers:
        l.trainable = is_trainable

def train_and_test(epochs=2000,batch=128, discrim_lr=0.0008, gan_lr=0.0004):
    d_loss = []
    a_loss = []
    running_d_loss = 0
    running_d_acc = 0
    running_a_loss = 0
    running_a_acc = 0
    discriminator = discriminator_builder()
    discriminator.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=discrim_lr, clipvalue=1.0, decay=6e-8), metrics=['accuracy'])
    generator = generator_builder()
    model = adversarial_builder(generator, discriminator, gan_lr)
    print("training.....")
    data, conditionImg = unified_data_loader.DataParser().get_data_for_GAN("28aircraftSketches", "28aircraftEdge60")
    for i in range(epochs):
        real_imgs = np.reshape(data[np.random.choice(data[:-10].shape[0],batch,replace=False)],(batch,28,28,1))
        con_imgs = np.reshape(conditionImg[np.random.choice(conditionImg[:-10].shape[0],batch,replace=False)],(batch,28,28,1))
        fake_imgs = generator.predict(con_imgs)
        x = np.concatenate((real_imgs,fake_imgs))
        y = np.ones([2*batch,1])
        y[batch:,:] = 0
        make_trainable(discriminator, True)
        d_loss.append(discriminator.train_on_batch(x,y))
        running_d_loss += d_loss[-1][0]
        running_d_acc += d_loss[-1][1]
        make_trainable(discriminator, False)
        y = np.ones([batch,1])
        a_loss.append(model.train_on_batch(con_imgs,y))
        running_a_loss += a_loss[-1][0]
        running_a_acc += a_loss[-1][1]

        print('Epoch #{}'.format(i+1))
        if (i+1)%500 == 0:
            print('Epoch #{}'.format(i+1))
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, running_d_loss/i, running_d_acc/i)
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, running_a_loss/i, running_a_acc/i)
            print(log_mesg)
            #noise = np.random.uniform(-1.0, 1.0, size=[16, 100])
            con_im = np.reshape(conditionImg[:10,:,:,:],(10,28,28,1))
            gen_imgs = generator.predict(con_im)
            plt.figure(figsize=(5,5))
            for k in range(10):
                plt.subplot(5, 5, k+1)
                plt.imshow(gen_imgs[k, :, :, 0], cmap='gray')
                plt.axis('off')
            for j in range(10):
                plt.subplot(5, 5, k + j +3 - 1)
                plt.imshow(con_im[j, :, :, 0], cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            plt.show()
            plt.savefig('./images/CGANSketch{}.png'.format(i+1))
    return a_loss, d_loss

def ganMain(epochs, batch_size, discrim_lr, gan_lr):
    train_and_test(epochs,batch_size, discrim_lr, gan_lr)