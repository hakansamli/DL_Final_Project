import pandas as pd
import tensorflow as tf
import numpy as np

def get_data(file_path, window_size):
    '''
    Reads in a csv file and returns a tuple of the input values, the actual future values, 
    the ids of differences between consecutive stock market values, the actual ids of future differences, 
    and a dictionary containing unique difference values and their ids.
    
    - param file_path: location of the file, datatype string
    - param window_size: the size of a window
    - return: a tuple of 
              1) a tensor of input stock market values with shape [number of windows, window_size] 
              2) a tensor of true stock market values with shape [number of windows, ]
              3) a tensor of differences between consecutive stock market values, with shape [number of windows,  window_size-1]
              4) a tensor of actual differences between consecutive stock market values, with shape [number of windows, ]
              5) a dictionary that maps unique difference values to their ids
    '''
    
    # Read in stock market values
    values = pd.read_csv(file_path)

    # Convert the values into a numpy array whose data type is float32
    values = np.array(values['Close'].values, dtype='float32')

    # Calculate differences between each pair of consecutive stock market values
    differences = values[1:] - values[:-1]

    # Create a dictionary that maps each unique value of difference to a unique id
    unique_diffs = set(differences)
    dict_diffs = {diff: i for i, diff in enumerate(list(unique_diffs))}

    # Convert the values of differences into their ids using the dictionary
    id_diffs = []
    for diff in differences:
        id_diffs.append(list(dict_diffs.keys()).index(diff))
    
    # Create empty lists
    input_values = []
    true_values = []

    input_diffs = []
    true_diffs = []
    
    # Transform stock market values and their difference ids into a batch of windows
    # Assume 'values' is one dimensional and therefore has length len(values)
    for i in range(len(values) - window_size):
        input_values.append(values[i: (i + window_size)])
        true_values.append(values[i+window_size])

        input_diffs.append(id_diffs[i: (i + window_size - 1)])
        true_diffs.append(id_diffs[i+window_size-1])

    # Convert to tensors
    input_values = tf.convert_to_tensor(input_values)
    true_values = tf.convert_to_tensor(true_values)
    
    input_diffs = tf.convert_to_tensor(input_diffs)
    true_diffs = tf.convert_to_tensor(true_diffs)

    return input_values, true_values, input_diffs, true_diffs, dict_diffs
