from tensorflow.keras.layers import (
    Input,
    Masking,
    LSTM,
    Dense,
    Bidirectional,
    BatchNormalization,
    TimeDistributed,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Reshape,
    Permute,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)




def adapt_input_for_rnn(input_layer, input_shape):
    """
    Adapt input to standard RNN shape: (timesteps, features).

    Supported:
    - 1knot_sta: (255, 1) -> unchanged
    - 1knot_xyz: (256, 3) -> unchanged
    - 3loops_sta: (3, 255, 1) -> (765, 1)
    - 3loops_xyz: (3, 256, 3) -> (768, 3)
    """
    if len(input_shape) == 2:
        return input_layer

    if len(input_shape) == 3:
        n_loops, n_steps, n_features = input_shape
        return Reshape((n_loops * n_steps, n_features))(input_layer)

    raise ValueError(
        f"Unsupported input_shape for RNN: {input_shape}. "
        "Expected rank 2 or 3 without batch dimension."
    )


def adapt_input_for_cnn(input_layer, input_shape):
    """
    Adapt input to standard Conv2D shape: (height, width, channels).

    Supported:
    - 1knot_sts: (255, 255, 1) -> unchanged
    - 3loops_sts: (3, 255, 255, 1) -> (255, 255, 3)
    """
    if len(input_shape) == 3:
        return input_layer

    if len(input_shape) == 4:
        n_loops, height, width, channels = input_shape

        if channels != 1:
            raise ValueError(
                f"Unsupported channel dimension for 3loops CNN input: {input_shape}"
            )

        x = Reshape((n_loops, height, width))(input_layer)
        x = Permute((2, 3, 1))(x)
        return x

    raise ValueError(
        f"Unsupported input_shape for CNN: {input_shape}. "
        "Expected rank 3 or 4 without batch dimension."
    )





def setup_RNN(input_shape, output_shape, hidden_activation, opt, norm):
    model = tf.keras.models.Sequential()

    input_layer = Input(shape=input_shape)
    x = adapt_input_for_rnn(input_layer, input_shape)

    if norm:
        x = BatchNormalization()(x)

    lstm_layer1 = LSTM(
        100,
        activation=hidden_activation,
        return_sequences=True,
        recurrent_dropout=0,
    )(x)

    bidirectional_layer = Bidirectional(
        LSTM(
            100,
            activation=hidden_activation,
            return_sequences=True,
            recurrent_dropout=0,
        )
    )(lstm_layer1)

    lstm_layer2 = LSTM(
        100,
        activation=hidden_activation,
        return_sequences=True,
        recurrent_dropout=0,
    )(bidirectional_layer)

    lstm_layer3 = LSTM(100, activation=hidden_activation)(lstm_layer2)

    output_layer = Dense(output_shape, activation="softmax")(lstm_layer3)

    model = Model(inputs=input_layer, outputs=output_layer)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    print("Generated RNN model:")
    print(model.summary())

    return model


def setup_RNN2(input_shape, output_shape, hidden_activation, opt, norm):
    # mask_value = -100

    model = tf.keras.models.Sequential()

    input_layer = Input(shape=input_shape)
    # input_layer = Masking(mask_value=mask_value)(input_layer)

    if norm:
        bn_layer = BatchNormalization()(input_layer)

        lstm_layer1 = LSTM(100, activation=hidden_activation, return_sequences=True, recurrent_dropout=0)(bn_layer)

    else:
        lstm_layer1 = LSTM(100, activation=hidden_activation, return_sequences=True, recurrent_dropout=0)(input_layer)

    bidirectional_layer = Bidirectional(LSTM(100, activation=hidden_activation, return_sequences=True, recurrent_dropout=0))(lstm_layer1)

    lstm_layer2 = LSTM(100, activation=hidden_activation, return_sequences=True, recurrent_dropout=0)(bidirectional_layer)

    lstm_layer2a = LSTM(100, activation=hidden_activation, return_sequences=True, recurrent_dropout=0)(lstm_layer2)

    # add final LSTM layer with output only from last memory cell (a la Vandans et al.)
    lstm_layer3 = LSTM(100, activation=hidden_activation)(lstm_layer2a)

    output_layer = Dense(output_shape, activation="softmax")(lstm_layer3)

    model = Model(inputs=input_layer, outputs=output_layer)

    # loss function compares y_pred to y_true: in this case sparse categoricalcrossentropy
    # used for labels that are integers (CategoricalCrossEntropy used for one-hot encoding)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=opt,  # adaptive moment estimation gradient descent
        # loss="MSE", MSE NOT VALID FOR LABELS THAT DONT REPRESENT KNOT
        loss=loss_fn,
        metrics=["accuracy"],
    )

    print("Generated RNN model:")
    print(model.summary())

    return model

def setup_RNN2b(input_shape, output_shape, hidden_activation, opt, norm):
    # mask_value = -100

    model = tf.keras.models.Sequential()

    input_layer = Input(shape=input_shape)
    # input_layer = Masking(mask_value=mask_value)(input_layer)

    if norm:
        bn_layer = BatchNormalization()(input_layer)

        lstm_layer1 = LSTM(100, activation=hidden_activation, return_sequences=True, recurrent_dropout=0)(bn_layer)

    else:
        lstm_layer1 = LSTM(100, activation=hidden_activation, return_sequences=True, recurrent_dropout=0)(input_layer)

    bidirectional_layer = Bidirectional(LSTM(100, activation=hidden_activation, return_sequences=True, recurrent_dropout=0))(lstm_layer1)

    bidirectional_layera = Bidirectional(LSTM(100, activation=hidden_activation, return_sequences=True, recurrent_dropout=0))(bidirectional_layer)

    lstm_layer2 = LSTM(100, activation=hidden_activation, return_sequences=True, recurrent_dropout=0)(bidirectional_layera)

    lstm_layer2a = LSTM(100, activation=hidden_activation, return_sequences=True, recurrent_dropout=0)(lstm_layer2)

    # add final LSTM layer with output only from last memory cell (a la Vandans et al.)
    lstm_layer3 = LSTM(100, activation=hidden_activation)(lstm_layer2a)

    output_layer = Dense(output_shape, activation="softmax")(lstm_layer3)

    model = Model(inputs=input_layer, outputs=output_layer)

    # loss function compares y_pred to y_true: in this case sparse categoricalcrossentropy
    # used for labels that are integers (CategoricalCrossEntropy used for one-hot encoding)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=opt,  # adaptive moment estimation gradient descent
        # loss="MSE", MSE NOT VALID FOR LABELS THAT DONT REPRESENT KNOT
        loss=loss_fn,
        metrics=["accuracy"],
    )

    print("Generated RNN model:")
    print(model.summary())

    return model

def setup_NN(input_shape, output_shape, hidden_activation, opt, norm):
    input_layer = Input(shape=input_shape)
    # input_layer = Masking(mask_value=mask_value)(input_layer)
   
    flatten_layer = Flatten()(input_layer)

    if norm:
        bn_layer = BatchNormalization()(flatten_layer)

        dense_layer1 = Dense(320, activation=hidden_activation)(bn_layer)

    else:
        dense_layer1 = Dense(320, activation=hidden_activation)(flatten_layer)

    # add hidden layers to NN
    dense_layer2 = Dense(320, activation=hidden_activation)(dense_layer1)

    dense_layer3 = Dense(320, activation=hidden_activation)(dense_layer2)

    dense_layer4 = Dense(320, activation=hidden_activation)(dense_layer3)

    # add output layer
    output_layer = Dense(output_shape, activation="softmax")(dense_layer4)

    model = Model(inputs=input_layer, outputs=output_layer)

    # loss function compares y_pred to y_true: in this case sparse categoricalcrossentropy
    # used for labels that are integers (CategoricalCrossEntropy used for one-hot encoding)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=opt,  # adaptive moment estimation gradient descent
        # loss="MSE", MSE NOT VALID FOR LABELS THAT DONT REPRESENT KNOT
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # save and print NN details
    print("Generated NN model:")
    print(model.summary())
    return model

def setup_NN2(input_shape, output_shape, hidden_activation, opt, norm):
    input_layer = Input(shape=input_shape)
    # input_layer = Masking(mask_value=mask_value)(input_layer)
   
    flatten_layer = Flatten()(input_layer)

    if norm:
        bn_layer = BatchNormalization()(flatten_layer)

        dense_layer1 = Dense(320, activation=hidden_activation)(bn_layer)

    else:
        dense_layer1 = Dense(320, activation=hidden_activation)(flatten_layer)

    # add hidden layers to NN
    dense_layer2 = Dense(320, activation=hidden_activation)(dense_layer1)

    dense_layer3 = Dense(320, activation=hidden_activation)(dense_layer2)

    dense_layer4 = Dense(320, activation=hidden_activation)(dense_layer3)
    dense_layer4a = Dense(320, activation=hidden_activation)(dense_layer4)
    dense_layer4b = Dense(320, activation=hidden_activation)(dense_layer4a)

    # add output layer
    output_layer = Dense(output_shape, activation="softmax")(dense_layer4b)

    model = Model(inputs=input_layer, outputs=output_layer)

    # loss function compares y_pred to y_true: in this case sparse categoricalcrossentropy
    # used for labels that are integers (CategoricalCrossEntropy used for one-hot encoding)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=opt,  # adaptive moment estimation gradient descent
        # loss="MSE", MSE NOT VALID FOR LABELS THAT DONT REPRESENT KNOT
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # save and print NN details
    print("Generated NN model:")
    print(model.summary())
    return model


def setup_CNN_STS(input_shape, output_shape, hidden_activation, opt, norm):
    """
    CNN designed for STS inputs.

    Supports:
    - 1knot_sts: (255, 255, 1)
    - 3loops_sts: (3, 255, 255, 1) -> adapted to (255, 255, 3)
    """
    input_layer = Input(shape=input_shape)
    x = adapt_input_for_cnn(input_layer, input_shape)

    # opcjonalna normalizacja batchowa na wejściu
    if norm:
        x = BatchNormalization()(x)

    # Block 1
    x = Conv2D(
        16,
        (3, 3),
        padding="same",
        activation=hidden_activation,
        kernel_regularizer=l2(1e-4),
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.10)(x)

    # Block 2
    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation=hidden_activation,
        kernel_regularizer=l2(1e-4),
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.15)(x)

    # Block 3
    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        activation=hidden_activation,
        kernel_regularizer=l2(1e-4),
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.20)(x)

    # Block 4
    x = Conv2D(
        128,
        (3, 3),
        padding="same",
        activation=hidden_activation,
        kernel_regularizer=l2(1e-4),
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # zamiast Flatten -> dużo lżejszy model
    x = GlobalAveragePooling2D()(x)

    x = Dense(
        64,
        activation=hidden_activation,
        kernel_regularizer=l2(1e-4),
    )(x)
    x = Dropout(0.30)(x)

    output_layer = Dense(output_shape, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    print("Generated CNN_STS model:")
    print(model.summary())

    return model


def localise_setup_RNN(input_shape, output_shape, hidden_activation, opt, norm):
    # mask_value = -100

    model = tf.keras.models.Sequential()

    input_layer = Input(shape=input_shape)
    # input_layer = Masking(mask_value=mask_value)(input_layer)

    if norm:
        bn_layer = BatchNormalization()(input_layer)

        lstm_layer1 = LSTM(100, activation=hidden_activation, return_sequences=True, recurrent_dropout=0)(bn_layer)

    else:
        lstm_layer1 = LSTM(100, activation=hidden_activation, return_sequences=True, recurrent_dropout=0)(input_layer)

    bidirectional_layer = Bidirectional(LSTM(100, activation=hidden_activation, return_sequences=True, recurrent_dropout=0))(lstm_layer1)

    lstm_layer2 = LSTM(100, activation=hidden_activation, return_sequences=True, recurrent_dropout=0)(bidirectional_layer)

    # add final LSTM layer
    lstm_layer3 = LSTM(100, activation=hidden_activation, return_sequences=True)(lstm_layer2)

    dense_layer = Dense(1, activation="sigmoid")(lstm_layer3)

    output_layer = TimeDistributed()(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    # loss function compares y_pred to y_true: in this case sparse categoricalcrossentropy
    # used for labels that are integers (CategoricalCrossEntropy used for one-hot encoding)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    model.compile(
        optimizer=opt,  # adaptive moment estimation gradient descent
        # loss="MSE", MSE NOT VALID FOR LABELS THAT DONT REPRESENT KNOT
        loss=loss_fn,
        metrics=["accuracy"],
    )

    print("Generated RNN model:")
    print(model.summary())

    return model


def localise_setup_NN(input_shape, output_shape, hidden_activation, opt, norm):
    input_layer = Input(shape=input_shape)
    # input_layer = Masking(mask_value=mask_value)(input_layer)

    if norm:
        bn_layer = BatchNormalization()(input_layer)

        dense_layer1 = Dense(320, activation=hidden_activation)(bn_layer)

    else:
        dense_layer1 = Dense(320, activation=hidden_activation)(input_layer)

    # add hidden layers to NN
    dense_layer2 = Dense(320, activation=hidden_activation)(dense_layer1)

    dense_layer3 = Dense(320, activation=hidden_activation)(dense_layer2)

    dense_layer4 = Dense(320, activation=hidden_activation)(dense_layer3)

    # add output layer
    output_layer = Dense(1, activation="sigmoid")(dense_layer4)

    model = Model(inputs=input_layer, outputs=output_layer)

    loss_fn = tf.keras.losses.BinaryCrossentropy()

    model.compile(
        optimizer=opt,  # adaptive moment estimation gradient descent
        # loss="MSE", MSE NOT VALID FOR LABELS THAT DONT REPRESENT KNOT
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # save and print NN details
    print("Generated NN model:")
    print(model.summary())
    return model


def setup_CNN(input_shape, output_shape, hidden_activation, opt, norm):
    input_layer = Input(shape=input_shape)
    x = adapt_input_for_cnn(input_layer, input_shape)

    if norm:
        x = BatchNormalization()(x)

    conv_layer1 = Conv2D(32, (3, 3), activation=hidden_activation)(x)
    max_pool_layer1 = MaxPooling2D((2, 2))(conv_layer1)
    conv_layer2 = Conv2D(32, (3, 3), activation=hidden_activation)(max_pool_layer1)

    flatten_layer = Flatten()(conv_layer2)
    dense_layer = Dense(80, activation=hidden_activation)(flatten_layer)

    output_layer = Dense(output_shape, activation="softmax")(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    print("Generated CNN model:")
    print(model.summary())

    return model

def build_model(hp, input_shape, output_shape, hidden_activation, norm):
    model = tf.keras.models.Sequential()
    
    # input layer
    model.add(tf.keras.layers.Flatten(input_shape=(input_shape,)))

    if norm:
        model.add(BatchNormalization())

    # hidden layers
    for i in range(hp.Int('layers', 2, 6)):
        model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i), 32, 512, step=32),
                                    activation=hidden_activation))
   
    # output layer
    model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    opt = tf.keras.optimizers.Adam(learning_rate=hp.Choice("lr", values = [.1,.01,.001,.0001,.00001]))
    
    model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['accuracy'])

    return model