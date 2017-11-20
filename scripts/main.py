


import tensorflow as tf
import numpy as np

import input_manager as input_manager
import cnnmodel as model


model_dir = "../model_output/wganset"
tb_dir = "../model_output/wganset/tensorboard"
total_steps =200000
check_interval = 100
batch_size = 64 
datam = input_manager.InputManager("../data/wgan_set.pb2" , batch_size, 1  )

cnn = model.cnn_model( batch_size )

iterator = datam.iterator()

input_tensors = iterator.get_next()

global_step = tf.get_variable(
        name="global_step" ,
        shape = []  ,
        dtype = tf.int64 ,
        initializer = tf.zeros_initializer() ,
        trainable = False ,
        collections = [ tf.GraphKeys.GLOBAL_VARIABLES , tf.GraphKeys.GLOBAL_STEP]
)
# banda1 y etiquetas 
cnn.build( input_tensors[0] , input_tensors[2] , global_step  )

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

saver = tf.train.Saver( )
hooks = [
            tf.train.CheckpointSaverHook(
                checkpoint_dir = model_dir ,
                save_steps = check_interval ,
                saver = saver 
            )
]

with tf.train.SingularMonitoredSession( hooks = hooks , checkpoint_dir = model_dir , config = config ) as sess:

    sess.run( iterator.initializer )
    start_step = sess.run( global_step )
    label = sess.run ( input_tensors[2] )
    print(label)
    try:

        cnn.train( start_step , total_steps , sess, tb_dir )

    except tf.errors.OutOfRangeError:

        print("data set agotado")

        
