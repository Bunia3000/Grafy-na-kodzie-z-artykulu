import os
import tensorflow as tf
import numpy as np

from helpers import datafile_structure

def load_dataset(dirname, knot, net, dtype, Nbeads, pers_len, label):
    """Function to load the dataset to use

    Args:
        dirname (str): Storage directory for set knot
        knot (str): knot name
        net (str): net type
        dtype (str): type of the chosen data
        Nbeads (int): Number of beads of the inputs
        pers_len(int): persistence length
        label (str): Label to use during training

    Raises:
        Exception: Exception if data type is not available

    Returns:
        tf.data.Dataset: loaded dataset
    """

    header, fname, select_cols = datafile_structure(dtype, knot, Nbeads, pers_len)

    n_col = len(select_cols)
    type_list = [tf.float32]*n_col

    # Loading the dataset file
    dataset = tf.data.experimental.CsvDataset(os.path.join(dirname,fname), type_list, header=header, field_delim=" ", select_cols=select_cols)

    # Reshape the incoming data
    dataset = dataset.batch(Nbeads)

    # # Padding
    # dataset = dataset.map(pad_sequences)

    dataset = dataset.map(lambda *x: tf.reshape(tf.stack(x, axis=1), (Nbeads, n_col)))

    if dtype == "XYZ":
        dataset = dataset.map(lambda x: x - tf.math.reduce_mean(x, axis=0))

    if "CNN" in net:
        dataset = dataset.map(lambda x: tf.reshape(x, (Nbeads, n_col, 1)))
        
    # Add Kymoknot labels if loading for a localisation problem
    if "localise" in net:
        label_dataset = tf.data.experimental.CsvDataset(dirname + f"KYMOKNOT/BU__KN_{knot}.dat.cleaned", [tf.float32], header=True, field_delim=" ", select_cols = [2])
        label_dataset = label_dataset.batch(Nbeads)
        label_dataset = label_dataset.map(lambda x: tf.reshape(x, (Nbeads, 1)))

        dataset = tf.data.Dataset.zip((dataset, label_dataset))

    elif "FOR" in net:
        dataset = dataset.map(lambda x: (tf.reshape(x, (Nbeads * n_col,)), label))

    else:
        # Create labelled classification database
        dataset = dataset.map(lambda x: (x, label))


    return dataset


def split_train_test_validation(dataset, train_size, test_size, val_size):
    """Generate splitted dataset

    Args:
        dataset (tf.data.Dataset): Total dataset
        train_size (int): size of the train dataset
        test_size (int): size of the test dataset
        val_size (int): size of the validation dataset

    Returns:
        tf.data.Dataset: train dataset
        tf.data.Dataset: test dataset
        tf.data.Dataset: validation dataset
    """

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    val_dataset = val_dataset.take(val_size)
    test_dataset = test_dataset.take(test_size)

    return train_dataset, test_dataset, val_dataset


def pad_sequences(x):
    """Padding sequences to 1000

    Args:
        x (tf.Tensor): input sequence

    Returns:
        tf.Tensor: padded sequence
    """

    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=1000, value=-100)
    return x
