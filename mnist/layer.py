class Layer():
    def __repr__(self):
        return self
    
    def __init__(self, name, x):
        self.name = name
        self.x = x

    def create_variable_on_cpu(self, name, shape, initializer, dtype):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
        return var

    def add_layer(self, **kwargs):
        # sort out in and out dimensions
        if len(kwargs) == 1:
            w_shape = kwargs['kernel']
            b_shape = w_shape[3]
        elif len(kwargs) == 2:
            w_shape = kwargs['in_dim']
            b_shape = kwargs['out_dim']

        # create the Tensor
        with tf.variable_scope(self.name):
            self.w = self.create_variable_on_cpu(name='weights', 
                                                 shape=w_shape,
                                                 initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(0.1)), 
                                                 dtype=tf.float32)
            self.b = self.create_variable_on_cpu(name='bias',
                                                 shape=b_shape,
                                                 initializer = tf.constant_initializer(0.0),
                                                 dtype=tf.float32)
            return self

    def conv(self, kernel, stride=1):
        self.add_layer(kernel=kernel)
        self.layer = tf.nn.conv2d(x, self.w, strides=[1, stride, stride, 1], padding='SAME') + self.b
        return self
        
    def linear(self, in_dim, out_dim):
        self.add_layer(in_dim=in_dim, out_dim=out_dim)
        self.layer = tf.matmul(x, self.w) + self.b
        return self
        
    def activation(self, activation='relu'):
        # an alternative would be to take a function handle as arg
        if activation == 'relu':
            self.layer = tf.nn.relu(self.layer)
        elif activation == 'sigmoid':
            self.layer = tf.nn.sigmoid(self.layer)
        elif activation == 'softmax':
            self.layer = tf.nn.softmax(self.layer)
        elif activation == 'linear':
            self.layer = self.layer        
        return self
        
    def dropout(self, keep_prob=0.75):
        self.layer = tf.nn.dropout(self.layer, keep_prob)
        return self        







