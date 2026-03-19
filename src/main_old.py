import os, csv

# Decomment the line if you want to train without the GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Comment to visualise debug and info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
import keras_tuner as kt
from keras import backend as K
import shap 

import numpy as np
import matplotlib.pyplot as plt

from helpers_old import get_knots, get_params, generate_model
from loaders_old import load_dataset, split_train_test_validation
from models_old import build_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")), tf.config.list_physical_devices("GPU"))

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

def main():
    # Loading the chosen datasets
    datasets = []
    for i, knot in enumerate(knots):
        datasets.append(load_dataset(os.path.join(master_knots_dir,knot,f"N{Nbeads}",f"lp{pers_len}"), knot, net, dtype, Nbeads, pers_len, i))

    # Creating dataset to sample from during training
    dataset = tf.data.Dataset.sample_from_datasets(datasets, seed=42)

    # Defining dataset size as a function of the number of knots
    ninputs = len(knots) * len_db

    # Splitting in train test and validation according to chosen sizes
    train_dataset, test_dataset, val_dataset = split_train_test_validation(
        dataset, int(ninputs * (0.9)), int(ninputs * (0.075)), int(ninputs * (0.025))
    )

    # Starting training or testing according to user preferences
    if mode == "train" or mode == "tune":
        # Shuffling the training dataset
        if net != "randFOR":
            train_dataset = train_dataset.shuffle(buffer_size=10000, seed=42)

        # Defining layer shape depending on the data type
        if dtype == "XYZ":
            in_layer = (Nbeads, 3)
        elif "2DSIGWRITHE" in dtype:
            in_layer = (Nbeads, Nbeads)
        else:
            in_layer = (Nbeads, 1)

        if "CNN" in net:
            in_layer = (in_layer[0], in_layer[1], 1)
            
        model = generate_model(net, in_layer, knots, norm)

        if mode=="train":
            train(model, train_dataset, val_dataset, bs)

        elif mode == "tune":

            if net != "randFOR":

                # HyperBand algorithm from keras tuner
                tuner1 = kt.Hyperband(
                    build_model(in_layer, len(knots), "relu", norm),
                    objective='val_accuracy',
                    max_epochs=10,
                    factor=2,
                    directory='KT_HB',
                    project_name=f"{dtype}_{prob}_Adj_{adj}_Norm_{norm}_Net_{net}_Nbeads_{Nbeads}_BatchSize_{bs}_LenDB_{len_db}_PersLen{pers_len}"
                )
                
                # Bayesian Optimisation algorithm from keras tuner
                tuner2 = kt.BayesianOptimization(
                        build_model(in_layer, len(knots), "relu", norm),
                        objective='val_accuracy',
                        max_trials = 50, 
                        directory='KT_BO', 
                        project_name=f"{dtype}_{prob}_Adj_{adj}_Norm_{norm}_Net_{net}_Nbeads_{Nbeads}_BatchSize_{bs}_LenDB_{len_db}_PersLen{pers_len}"
                )

                tuner = tune(tuner1, train_dataset, val_dataset, bs)
        
                best_model = tuner.get_best_models(num_models=1)[0]
                best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
                print("Best model: ", best_model)
                print("Best hyperparameters: ", best_hps)
                
    elif mode == "test":
        test(test_dataset, bs)

    elif mode == "importance":
        plot_importance(test_dataset)

def train(model, train_dataset, val_dataset, bs):
    """Training function

    Args:
        model (tf.keras.Model): Model to use during training
        train_dataset (tf.data.Dataset): Training dataset
        val_dataset (tf.data.Dataset): Validation dataset
        bs (int): batch size
    """

    # Setting up the datasets used during the training
    if net != "randFOR":
        train_dataset = train_dataset.repeat()

    train_dataset = train_dataset.batch(bs)
    val_dataset = val_dataset.batch(bs)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    # Early Stopping Callback
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10, restore_best_weights=True, min_delta=0.001)

    # Finding best NN model weights and saving them during training process
    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, save_weights_only=False, monitor="val_loss", mode="min", save_best_only=True
    )

    # Creating the callbacks list
    cb_list = [es, mc]

    # Fitting the model
    history = model.fit(train_dataset, steps_per_epoch=250, epochs=epochs, verbose=1, callbacks=cb_list, validation_data=val_dataset)

    # Saving training history
    with open(os.path.join(checkpoint_filepath,"training_history.csv"), "w") as f:
        w = csv.DictWriter(f, history.history.keys())
        w.writeheader()
        w.writerow(history.history)

def plot_importance(test_dataset):
    """Function to plot importance in results

    Args:
        test_dataset (tf.data.Dataset): Test dataset
    """    

    # Loading the model
    model = tf.keras.models.load_model(checkpoint_filepath)

    # Creating test labels and input datasets
    label_dataset = test_dataset.map(lambda x, y: y)
    test_dataset = test_dataset.map(lambda x, y: x)
    
    el_num = (np.random.rand(50) * len_db * 0.075 * len(knots)).astype(int)

    input_sequences = []
    input_labels = []
    for i, (x_element, y_element) in enumerate(zip(test_dataset, label_dataset)):
        if np.any(el_num == i):
            input_sequences.append(x_element.numpy().reshape(1,-1))
            input_labels.append(y_element)

    input_sequences = np.array(input_sequences)

    explainer = shap.DeepExplainer(model, input_sequences.transpose(0,2,1))

    for j, (input_sequence, input_label) in enumerate(zip(input_sequences, input_labels)):
        
        plt.plot(explainer.shap_values(input_sequence.transpose(1,0)[None,:,:])[input_label].flatten())

        plt.savefig(os.path.join(checkpoint_filepath,"layer_analysis",f"activation_gradient_label_{input_label}_test_{el_num[j]}.pdf"))
        plt.cla()

        for i in range(len(model.layers)):
            intermediate_model=tf.keras.models.Model(inputs=model.input,outputs=model.layers[i].output) #Intermediate model between Input Layer and Output Layer which we are concerned about

            activations = intermediate_model(input_sequence)

            plt.plot(activations.numpy().flatten())
            plt.xlabel("Neuron index")
            plt.savefig(os.path.join(checkpoint_filepath,"layer_analysis",f"activation_layer_{i}_label_{input_label}_test_{el_num[j]}.pdf"))
            plt.cla()

            try:
                # WEIGHT-BASED METHOD
                weights_per_neuron = np.abs(model.weights[i*2]).sum(axis=0)
                
                plt.plot(weights_per_neuron)
                plt.xlabel("Neuron index")
                plt.savefig(os.path.join(checkpoint_filepath,"layer_analysis",f"weights_layer_{i}.pdf"))
                plt.cla()
            except Exception:
                continue

def test(test_dataset, bs):
    """Testing function for the models created

    Args:
        test_dataset (tf.data.Dataset): Test dataset
    """

    # Loading the model
    model = tf.keras.models.load_model(checkpoint_filepath)

    # Defining the batch size for the model
    test_dataset = test_dataset.batch(bs)

    print("Evaluating the accuracy on the test dataset:")
    model.evaluate(test_dataset)

    # Creating test labels and input datasets
    test_labels = test_dataset.map(lambda x, y: y)
    test_dataset = test_dataset.map(lambda x, y: x)

    # Use the network to predict values
    predictions = np.argmax(model.predict(test_dataset), axis=-1)

    # Getting the true labels
    labels = np.empty(0)
    for arr in test_labels.as_numpy_iterator():
        labels = np.hstack([labels, arr.flatten()])
        
    # Getting the confusion matrix and saving
    cf_matrix = tf.math.confusion_matrix(labels, predictions)
    print(cf_matrix)
    np.savetxt(os.path.join(checkpoint_filepath,"conf_m.txt"), cf_matrix.numpy(), fmt="%i")

def tune(tuner, train_dataset, val_dataset, bs):
    """Hyperparameter tuning function

    Args:
        tuner (keras_tuner.Hyperband): Optimisation algorithm
        train_dataset (tf.data.Dataset): Training dataset
        val_dataset (tf.data.Dataset): Validation dataset
        bs (int): batch size
    """

    # Setting up the datasets used during the training
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(bs)
    val_dataset = val_dataset.batch(bs)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    # Early Stopping Callback
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10, restore_best_weights=True, min_delta=0.001)

    # Finding best NN model weights and saving them during training process
    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, save_weights_only=False, monitor="val_loss", mode="min", save_best_only=True
    )

    # Creating the callbacks list
    cb_list = [es, mc]

    print("Tuning...")
    tuner.search(train_dataset, steps_per_epoch=250, epochs=epochs, verbose = 1, callbacks = cb_list, validation_data=val_dataset)

    return tuner

if __name__ == "__main__":
    args = get_params()
    prob = args.problem
    dtype = args.datatype
    adj = args.adjacent
    norm = args.normalised
    net = args.network
    epochs = args.epochs
    knots = get_knots(prob)
    mode = args.mode
    Nbeads = int(args.nbeads)
    bs = int(args.b_size)
    len_db = int(args.len_db)
    master_knots_dir = args.master_knots_dir
    pers_len = args.pers_len

    # Setting up loading/training directory
    checkpoint_filepath = f"NN_model_best_{dtype}_{prob}_Adj_{adj}_Norm_{norm}_Net_{net}_Nbeads_{Nbeads}_BatchSize_{bs}_LenDB_{len_db}_PersLen{pers_len}"

    main()
