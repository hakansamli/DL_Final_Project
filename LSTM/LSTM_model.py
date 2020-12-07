import tensorflow as tf

class model_LSTM(tf.keras.Model):
    def __init__(self, batch_size, window_size, num_unique_diffs, is_stacked=False):
        '''
        A model that uses one LSTM to predict stock market values
        - param batch_size: a batch size for training and testing
        - param window_size: a window size for stock market values. 
                             Hence the differences will have its window size as window_size-1. 
                             (For example, there are 4 differences between consecutive pairs of 5 values)
        - param num_unique_diffs: the number of uniqe difference values between consecutive stock market values
        - return: nothing
        '''
        super(model_LSTM, self).__init__()
        
        # Define hyperparameters
        self.batch_size = batch_size
        self.window_size = window_size
        self.num_unique_diffs = num_unique_diffs
        self.embedding_size = 64

        self.is_stacked = is_stacked
        
        # Define an optimizer
        self.optimizer = tf.keras.optimizers.Adam(0.1)

        # Define the embedding
        self.embeddings = tf.Variable(tf.random.normal([self.num_unique_diffs, self.embedding_size], stddev=0.1))
        
        # Define neural networks

        # Return sequences only when it is a stacked LSTM
        self.lstm1 = tf.keras.layers.LSTM(64, dropout=0.2, return_sequences=self.is_stacked)
        
        # Add more LSTM(s) to create a stacked_LSTM model
        if self.is_stacked == True:
            self.lstm2 = tf.keras.layers.LSTM(32, dropout=0.2)
        
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        #self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(self.num_unique_diffs)
        
    def call(self, input_diffs):
        '''
        - param input_diffs: a batch of windows of stock market value differences, with shape [batch_size, window_size-1]
        - return: probability distributions
        '''
        
        embedded = tf.nn.embedding_lookup(self.embeddings, input_diffs)
        
        output = self.lstm1(embedded)
        
        # When the model is a stacked LSTM
        if self.is_stacked == True:
            output = self.lstm2(output)

        output = self.dense1(output)
        #output = self.dropout(output)
        output = self.dense2(output)
        
        return tf.nn.softmax(output)

    def loss_function(self, true_diffs, prbs):
        '''
        - param true_diffs: a sequence of true differences. A tensor of shape [batch_size, ].
        - param prbs: probabilities of unique difference values for each sequence in a batch. A tensor of shape [batch_size, num_unique_diffs].
        - return: loss
        '''
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(true_diffs, prbs))
