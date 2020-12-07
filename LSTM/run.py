from preprocess import *
from LSTM_model import model_LSTM
import tensorflow as tf
import numpy as np

def train(model, train_inputs, train_labels):
    '''
    Trains for one epoch

    - param model: either a single LSTM or a stacked LSTM for stock market prediction
    - param train_inputs: a batch of ids of stock market value differences. A tensor of shape [number of windows, window_size-1]
    - param train_labels: a batch of ids of true differences. A tensor of shape [number of windows, ]
    - return: nothing
    '''
    
    for i in range(int(train_inputs.shape[0] / model.batch_size)):
        with tf.GradientTape() as tape:
            
            prbs = model.call(train_inputs[(i * model.batch_size):((i+1) * model.batch_size), :])
            loss = model.loss_function(train_labels[(i * model.batch_size):((i+1) * model.batch_size)], prbs)
                    
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    pass

def test(model, test_inputs, test_labels):
    '''
    Tests for one epoch

    - param model: either a single LSTM or a stacked LSTM for stock market prediction
    - param test_inputs: a batch of ids of stock market value differences. A tensor of shape [number of windows, window_size-1]
    - param test_labels: a batch of ids of true differences. A tensor of shape [number of windows, ]
    - return: perplexity
    '''
    
    loss = []
    
    for i in range(int(test_inputs.shape[0] / model.batch_size)):
        
        prbs = model.call(test_inputs[(i * model.batch_size):((i+1) * model.batch_size), :])
        loss.append(model.loss_function(test_labels[(i * model.batch_size):((i+1) * model.batch_size)], prbs))

    return tf.exp(tf.reduce_mean(loss))

def predict_values(model, input_values, input_diffs, dict_diffs):
    '''
    Creates a list of predicted stock market values

    - param model: a single or stacked LSTM
    - param input_values: a tensor of input stock market values with shape [number of windows, window_size]
    - param input_diffs: a tensor of differences between consecutive stock market values, with shape [number of windows,  window_size-1]
    - param dict_diffs: a dictionary that maps unique difference values to their ids
    - return: a tensor of predicted stock market value for each window 
    '''
    predicted_values = []
    
    # Create a probability distribution for each window
    prbs = model.call(input_diffs)
    
    # For each window
    for i in range(input_values.shape[0]):
        
        # Normalize
        p = np.array(prbs[i])
        p /= p.sum()
        
        # Randomly choose an id out of the probability distribution
        predicted_id = np.random.choice(len(dict_diffs), p=p)
        
        # Record the value of the predicted difference given a predicted id
        predicted_diff = list(dict_diffs.keys())[predicted_id]
        
        # Add the predicted difference to the last element in the window, and record the value
        predicted_values.append(int(input_values[i, model.window_size-1]) + predicted_diff)
    
    # Convert the list into a tensor whose data type is float32
    return tf.convert_to_tensor(predicted_values, dtype=tf.float32)

def main():
    '''
    Main function that runs training and testing
    '''
    
    # File path for data
    file_path = '/gpfs/main/home/cpark53/cs1470/DL-Final_project/lstm/^GSPC.csv'

    # Set window_size
    window_size = 20
        
    # Load data with get_data in preprocess.py
    input_values, true_values, input_diffs, true_diffs, dict_diffs = get_data(file_path, window_size)
    
    # Round 1. Divide the dataset into 70% training data and 30% testing data
    train_inputs = input_diffs[0:int(0.7*len(input_diffs))]
    train_labels = true_diffs[0:int(0.7*len(true_diffs))]

    test_inputs = input_diffs[int(0.7*len(input_diffs)):]
    test_labels = true_diffs[int(0.7*len(true_diffs)):]
    
    # Single LSTM
    single_lstm = model_LSTM(30, window_size, len(dict_diffs))

    print("-----------------------------------------------------------")
    print("About to train a single LSTM model")
    print("{}% of data as train and {}% of data as test".format(70, 30))
    print("\n")

    train(single_lstm, train_inputs, train_labels)

    perplexity = test(single_lstm, test_inputs, test_labels)
    print("Perplexity is {}.".format(perplexity))

    predicted_values = predict_values(single_lstm, input_values, input_diffs, dict_diffs)
    mse = tf.keras.losses.MSE(true_values, predicted_values)
    print("The MSE is {}.".format(mse))
    print("\n")
    
    print("End of the single LSTM")
    
    # Stacked LSTM
    stacked_lstm = model_LSTM(30, window_size, len(dict_diffs), True)

    print("-----------------------------------------------------------")
    print("About to train a stacked LSTM model")
    print("{}% of data as train and {}% of data as test".format(70, 30))
    print("\n")

    train(stacked_lstm, train_inputs, train_labels)
    perplexity = test(stacked_lstm, test_inputs, test_labels)
    print("Perplexity is {}.".format(perplexity))
    
    predicted_values = predict_values(stacked_lstm, input_values, input_diffs, dict_diffs)
    mse = tf.keras.losses.MSE(true_values, predicted_values)
    print("The MSE is {}.".format(mse))
    print("\n")
    
    print("End of the stacked LSTM")
    
    # Round 2. Run the entire dataset with multiple epochs
       
    # Single LSTM
    single_lstm = model_LSTM(30, window_size, len(dict_diffs))

    print("-----------------------------------------------------------")
    print("About to train a single LSTM model")
    print("With the entire data running {} epochs".format(50))
    print("\n")

    # Run 50 epochs of training and testing
    for i in range(50):
        train(single_lstm, input_diffs, true_diffs)
        perplexity = test(single_lstm, input_diffs, true_diffs)
        
        if (i+1)%10 == 0:
            print("Perplexity after {} epochs is {}.".format(i+1, perplexity))
    
    print("\n")
    predicted_values = predict_values(single_lstm, input_values, input_diffs, dict_diffs)
    mse = tf.keras.losses.MSE(true_values, predicted_values)
    print("The MSE is {}.".format(mse))
    print("\n")

    print("End of the single LSTM")

    # Stacked LSTM
    stacked_lstm = model_LSTM(30, window_size, len(dict_diffs), True)

    print("-----------------------------------------------------------")
    print("About to train a stacked LSTM model")
    print("With the entire data running {} epochs".format(50))
    print("\n")

    # Run 50 epochs of training and testing
    for i in range(50):
        train(stacked_lstm, input_diffs, true_diffs)
        perplexity = test(stacked_lstm, input_diffs, true_diffs)
        
        if (i+1)%10 == 0:
            print("Perplexity after {} epochs is {}.".format(i+1, perplexity))
    
    print("\n")
    predicted_values = predict_values(stacked_lstm, input_values, input_diffs, dict_diffs)
    mse = tf.keras.losses.MSE(true_values, predicted_values)
    print("The MSE is {}.".format(mse))
    print("\n")

    print("End of the stacked LSTM")
    print("-----------------------------------------------------------")

if __name__ == '__main__':
    main()
