#modified based on https://github.com/lc82111/Keras_HED/blob/master/src/networks/hed.py
from keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, Add
from keras.layers import Concatenate, Activation
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from keras import optimizers
from keras.losses import binary_crossentropy
from keras.activations import relu

def side_branch(x, factor):
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)

    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)

    return x

def convModel():
    # Input
    img_input = Input(shape=(256,256,3), name='input')

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    b1= side_branch(x, 1)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    b2= side_branch(x, 2)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    b3= side_branch(x, 4)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    b4= side_branch(x, 8)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    b5= side_branch(x, 16)

    # fuse
    fuse = Concatenate(axis=-1)([b1, b2, b3, b4, b5])
    fuse = Conv2D(1, (1,1), padding='same', use_bias=False, activation=None)(fuse)
    # outputs
    o1    = Activation('sigmoid', name='o1')(b1)
    o2    = Activation('sigmoid', name='o2')(b2)
    o3    = Activation('sigmoid', name='o3')(b3)
    o4    = Activation('sigmoid', name='o4')(b4)
    o5    = Activation('sigmoid', name='o5')(b5)
    ofuse = Activation('sigmoid', name='ofuse')(fuse)
    model = Model(inputs=[img_input], outputs=[o1, o2, o3, o4, o5, ofuse])
    model.load_weights("./checkpoints/HEDSeg/checkpoint.212-0.11.hdf5")
    #freeze all layers produced by HED network
    for layer in model.layers:
        layer.trainable = False
    added = Add(name = "add")([o1, o2, o3, o4, o5, ofuse])
    threshold = Activation(lambda x: relu(x, threshold = 4), name = "threshold")(added)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block6_pool')(threshold)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block6_conv1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block6_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block7_pool')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block7_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block7_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block8_pool')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block8_conv1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block8_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block9_pool')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block9_conv1')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block9_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block10_pool')(x)
    x = Conv2DTranspose(16, kernel_size=2*2, strides=2, padding='same', activation=None)(x)
    x = Conv2DTranspose(32, kernel_size=2*2, strides=2, padding='same', activation=None)(x)
    x = Conv2DTranspose(64, kernel_size=2*2, strides=2, padding='same', activation=None)(x)
    x = Conv2DTranspose(32, kernel_size=2*2, strides=2, padding='same', activation=None)(x)
    o6 = Conv2DTranspose(3, kernel_size=2*2, strides=2, padding='same', activation="sigmoid", name = "o6")(x)
    model = Model(inputs=model.input, outputs = [o6])
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
    model.compile(loss={'o6': cross_entropy_balanced,
                        },
                  metrics={'o6': cross_entropy_balanced},
                  optimizer=adam)

    return model


def cross_entropy_balanced(y_true, y_pred):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)

def ofuse_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x