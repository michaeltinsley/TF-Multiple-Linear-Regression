import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import mean_squared_error


class LR:
    
    def __init__(self, X, Y, epochs=100, learning_rate=0.001):
        self.X = X
        self.Y = Y
        
        self.epochs = epochs
        
        self.learning_rate = learning_rate
    
        self.create_network()
        
        
    def create_network(self):
        """
        Creates Tensorflow variables.
        """
        
        self.x = tf.placeholder(tf.float32, shape=[None, self.X.shape[1]])
        self.y = tf.placeholder(tf.float32, shape=[None, self.Y.shape[1]])
        
        self.W = tf.Variable(tf.zeros( shape = [self.X.shape[1], self.Y.shape[1]]))
        self.b = tf.Variable(tf.zeros( shape=[ self.Y.shape[1]]))
        
        self.output = tf.matmul( self.x, self.W) + self.b
        
        self.loss = tf.reduce_sum(tf.square(self.output - self.y))
        self.optimiser = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss) # Optimiser selection and learning rate
        
        
    def train(self):
        """
        Trains the created MLP model.
        """
        
        init = tf.global_variables_initializer()
        hist = []
        with tf.Session() as sess:
        
            sess.run(init)
            
            for epoch in tqdm(range(self.epochs)):
                sess.run([self.optimiser, self.loss], feed_dict={self.x : self.X, 
                                                                 self.y : self.Y})
                weight, bias = sess.run([self.W, self.b])
                hist += [mean_squared_error(np.dot(self.X, weight) + bias, self.Y)]
            self.W, self.b =  sess.run([self.W, self.b])
            
        self.history = hist
        
    def predict(self, test_X):
        """
        Returns a set of predicted outputs.
        """
        output = tf.matmul( test_X.astype(np.float32), self.W) + self.b
    
        with tf.Session() as sess:
            prediction = output.eval()
        
        return prediction
        
