#modified based on https://github.com/lc82111/Keras_HED/blob/master/main_segmentation.py
from __future__ import print_function
import os
from unified_data_loader import DataParser
from models.hed_add_layers import convModel
from models.hed import hed
from keras.utils import plot_model
from keras import backend as K
from keras import callbacks
import numpy as np
import cv2


def generate_minibatches(types, dataParser, train=True):
    while True:
        if train:
            batch_ids = np.random.choice(dataParser.training_ids, dataParser.batch_size_train)
        else:
            batch_ids = np.random.choice(dataParser.validation_ids, dataParser.batch_size_train*2)
        ims, ems, _ = dataParser.get_batch_with_ids(batch_ids, train)
        if types == "hed":
            yield(ims, [ems,ems,ems,ems,ems,ems])
        elif types == "conv":
            yield(ims, [ems])

def convMain(types, cv_lr, batch_size, epochs):
    # params
    model_name = 'conv'
    model_dir     = os.path.join('checkpoints', model_name)
    batch_size_train = 16
    epochs = 10
    K.set_image_data_format('channels_last')
    K.image_data_format()
    if not os.path.isdir(model_dir): os.makedirs(model_dir)

    # prepare data
    dataParser = DataParser(batch_size_train, 256, 256)

    # model
    if types == "hed":
        model = hed()
    elif types == "conv":
        model = convModel()
    plot_model(model, to_file=os.path.join(model_dir, 'model.png'), show_shapes=True)
    for i in range(1):
        model.fit_generator(
                            generate_minibatches(types, dataParser,),
                            steps_per_epoch=dataParser.steps_per_epoch,
                            epochs=1,
                            validation_data=generate_minibatches(types, dataParser, train=False),
                            validation_steps=dataParser.validation_steps)
        model.save_weights(str(i)+".checkpoint.hdf5")
    # pdb.set_trace()
    #print(train_history)
