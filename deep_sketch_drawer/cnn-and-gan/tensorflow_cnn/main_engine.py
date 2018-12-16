import util
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
from PIL import Image
import os
import scipy.misc
import shutil

def conv(input, filter_num, kernel=[5, 5], act = tf.nn.relu, pad_type = 'same', stride_num = 1, ts_name = None):
    #conv layer
    return tf.layers.conv2d(input, filters=filter_num, kernel_size=kernel, strides = [stride_num, stride_num],
                             padding=pad_type, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32),
                            activation = act, name = ts_name)

def max_pool(X,pool_size, strides = 2, ts_name = None):
    #pooling layer
    return tf.layers.max_pooling2d(X, pool_size, strides, name = ts_name)

def create_placeHolders(batch_size):
    return tf.placeholder(tf.float32, [batch_size, 64, 64, 3], name="color_image"),\
            tf.placeholder(tf.float32, [batch_size, 64, 64, 1], name = "sketch_image")

def build_model(input_im):
    """

    :param input_im: input tensor
    :return: output tensor
    """
    temp = conv(input_im, 8, [2,2])
    temp = conv(temp, 8, [2,2])
    temp = max_pool(temp, 3)
    temp = conv(temp, 32, [3,3])
    temp = max_pool(temp, 3)
    temp = conv(temp, 16, [3,3])
    temp = conv(temp, 8, [1, 1])
    temp = tf.image.resize_images(temp, [64,64])
    temp =  conv(temp, 1, ts_name="result", act=None)
    #return temp
    return tf.layers.flatten(temp)

def loss_cal(prediction, sketch):
    #loss function: mean squared error
    loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(output = prediction, target = tf.layers.flatten(sketch), from_logits = True))
    return loss, prediction

def main(arg):
    #hyperparameters and system argument parsing
    ID, learn_rate, iter = arg.ID, arg.learning_rate, arg.iter
    batch_size, record_iter = arg.batch_size, arg.record
    mode = arg.mode
    #check if there is a restore argument
    if arg.restore:
        restore_file = ID + "_" + arg.restore + ".ckpt"
    else:
        restore_file = None
    image_path = arg.image#os.getcwd() + "/" + arg.image
    sketch_path = arg.sketch#os.getcwd() + "/" + arg.sketch
    allPhoto = os.listdir(image_path)
    if mode == "train":
        print('constructing the model')
        sess = tf.Session()
        start_iter = 0
        if not arg.restore:
            #create a new graph
            input_im, sketch_im =  create_placeHolders(batch_size) # 64*64*3
            createSketch = build_model(input_im)
            loss, pred_x = loss_cal(createSketch, sketch_im)
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = learn_rate
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 2000, 0.96, staircase=True)
            opter = tf.train.AdamOptimizer(learning_rate).minimize(loss, name = "opt_to_res")
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            #messingly restore a training at iteration iter
            start_iter = restore_file.split(".")[0].split("_")[-1]
            print("continue training from " + start_iter)
            saver = tf.train.import_meta_graph("model_save/"+ID+"/"+restore_file + ".meta")
            init = tf.global_variables_initializer()
            sess.run(init)
            saver.restore(sess, "model_save/" + ID + "/" + restore_file)
            graph = tf.get_default_graph()
            input_im = graph.get_tensor_by_name("color_image:0")
            sketch_im = graph.get_tensor_by_name("sketch_image:0")
            loss = graph.get_tensor_by_name("mean_squared_error/value:0")
            opter = graph.get_operation_by_name("opt_to_res")
        #create a folder for saving models
        if not os.path.exists("model_save/" + ID):
            os.makedirs("model_save/" + ID)
        for i in range(iter):
            #randomly select photos from the directory
            selectedImage = np.random.choice(allPhoto, batch_size)
            image_batch = np.zeros((batch_size, 64, 64, 3), dtype=float)
            sketch_batch = np.zeros((batch_size, 64, 64, 1), dtype=float)
            for j in range(batch_size):
                #read in the images
                image_z = util.read(image_path, selectedImage[j], 'photo')
                sketch_im_z = util.read(sketch_path, selectedImage[j], 'sketch')
                while sketch_im_z is None:
                    #in case all sketches are invalid for a photo
                    select = np.random.choice(allPhoto)
                    image_z = util.read(image_path, select, 'photo')
                    sketch_im_z = util.read(sketch_path, select, 'sketch')
                sketch_batch[j, :, :, :] = sketch_im_z
                image_batch[j, :, :, :] = image_z
            #train
            _, losses= sess.run([opter, loss],
                                      feed_dict={input_im: image_batch,  sketch_im: sketch_batch})
            print("iter: ", int(start_iter) + i,  " loss: ", losses)
            if i%record_iter == 0:
                save_path = "model_save/" + ID + "/"+ID + "_" + str(int(start_iter) + i) + ".ckpt"
                print("Save to path:", save_path)
                saver.save(sess=sess, save_path=save_path)

    elif mode == "test":
        #the restore flag must be included in argument
        #messingly testing
        assert(restore_file)
        start_iter = 0
        #input_im, sketch_im = create_placeHolders(batch_size)  # 64*64*3
        #createSketch = build_model(input_im)
        #loss, pred_x = loss_cal(createSketch, sketch_im)
        sess = tf.Session()
        saver = tf.train.import_meta_graph("model_save/" + ID + "/" + restore_file + ".meta")
        #saver = tf.train.Saver()
        #tf.reset_default_graph()
        saver.restore(sess, "model_save/" + ID + "/" + restore_file)
        graph = tf.get_default_graph()
        #get input tensor and prediction tensor
        input_im = graph.get_tensor_by_name("color_image:0")
        createSketch = graph.get_tensor_by_name("result/BiasAdd:0")
        if os.path.exists('result'):
            shutil.rmtree('result')
        os.makedirs('result')
        image_path = arg.image
        allPhoto = os.listdir(image_path)
        #randomly select 32 images from the testPhoto directory
        selectedImage = np.random.choice(allPhoto, batch_size)
        image_batch = np.zeros((batch_size, 64, 64, 3), dtype=float)
        for j in range(batch_size):
            image_z = util.read(image_path, selectedImage[j], 'photo')
            image_batch[j, :, :, :] = image_z
        sketchy = sess.run(createSketch, feed_dict={input_im: image_batch})
        sketchy = tf.image.resize_images(sketchy, [64, 64])
        finalSketch= sess.run(sketchy)
        for i in range(batch_size):
            #resize (256,256,1) to (256, 256)
            im = np.reshape(finalSketch[i, :, :, :], (64, 64))
            mask = im < 0.5
            im = im * mask
            #local use
            result = arg.result
            scipy.misc.imsave(result +"/"+ selectedImage[i], im)
        save_path = "model_save/" + ID + "/" + ID + "_" + str(int(start_iter) + iter) + ".ckpt"
        saver.save(sess=sess, save_path=save_path)
        print("test sketches generated")
sys_args = util.parse_args()
main(sys_args)
#main2()