import tensorflow as tf 

class cnn_model():


    def __init__(self , mb_size ):


        self.dropout = 0.2
        self.mb_size = mb_size
        
        return 
    def build(self ,inputs , labels   , global_step ):
        # build cnn
        #inputs = self.build_bands( inputs[0] , inputs[1] )
        print( inputs.shape ) 
        inputs = tf.reshape( inputs , [self.mb_size , 75 , 75 , 1  ] )
        labels = tf.reshape( labels , [ self.mb_size , 1 ] ) 
        # input [-1 , 75 , 75 , 2 ]
        conv1 = tf.layers.conv2d(
            inputs = inputs ,
            filters = 8 ,
            kernel_size = [5,5] ,
            strides = [5,5] ,
            padding = 'same'
        )
    
        conv1 = tf.layers.batch_normalization(conv1)
        conv1 = tf.layers.dropout(conv1 , self.dropout )
        # conv1 [-1 , 15,15,8]
        conv2 = tf.layers.conv2d(
            inputs = conv1 ,
            filters = 32 ,
            kernel_size = [5,5] , 
            strides = [5,5] ,
            padding = 'same'
        )

        conv2 = tf.layers.batch_normalization(conv2 )
        conv2 = tf.layers.dropout( conv2 , self.dropout )
        # conv2 [-1 ,  3 , 3 , 32 ]

        fc_input = tf.reshape( conv2 , [-1 , 3*3*32] ) 
        logits = tf.layers.dense(
            inputs = fc_input ,
            units = 1 ,
            activation  = tf.nn.relu 
        )

        out = tf.sigmoid( logits ) 

        self.loss = tf.losses.log_loss(
            predictions = logits + 1e-8  , labels = labels 
        )

        
        self.train_op = tf.train.AdamOptimizer(
            learning_rate = 1e-6
        ).minimize( self.loss , global_step = global_step   )

        tf.summary.scalar('log_loss' , self.loss )

        self.build_merge()
    

    def train(self , start_step ,epochs , sess,  tb_dir  ):

        self.build_writer(tb_dir , sess )
        
        for it in xrange(start_step , epochs):

            loss , _  = sess.run( [self.loss , self.train_op] )
            print("step {}".format(it ) )
            print("loss {}".format( loss ))
            if it % 100 == 0 :

                summary = sess.run( self.merged_op )
                self.writer.add_summary( summary , it ) 

        return

    def build_merge(self ):

        self.merged_op = tf.summary.merge_all()

    def build_writer(self , tb_dir , sess) :

        self.writer = tf.summary.FileWriter( tb_dir , sess.graph )

    def predict(self):


        return

    def new_loss(self , logits , targets ) :

        return -tf.reduce_sum( targets * tf.log( logits + 1e-10))
    
        
