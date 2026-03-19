from typing import List

try:
    import tensorflow_decision_forests as tfdf
except ImportError:
    tfdf = None

from models import *


SUPPORTED_DTYPES = {
    "1knot_xyz",
    "1knot_sta",
    "1knot_sts",
    "3loops_xyz",
    "3loops_sta",
    "3loops_sts",
}

SUPPORTED_NETS = {
    "FFNN",
    "FFNN2",
    "CNN",
    "RNN",
    "RNN2",
    "RNN2b",
    "localiseFFNN",
    "localiseRNN",
    "randFOR",
    "CNN_STS",
}


def validate_dtype(dtype: str) -> str:
    """
    Validate new theta-curve dtype.
    """
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported dtype: {dtype}. "
            f"Supported dtypes: {sorted(SUPPORTED_DTYPES)}"
        )
    return dtype


def validate_net(net: str) -> str:
    """
    Validate network name.
    """
    if net not in SUPPORTED_NETS:
        raise ValueError(
            f"Unsupported network: {net}. "
            f"Supported networks: {sorted(SUPPORTED_NETS)}"
        )
    return net


def generate_model(net: str, in_layer: tuple, knots: List[str], norm: bool):
    """
    Generate model according to chosen network type.

    Args:
        net: network type
        in_layer: input shape
        knots: list of class names
        norm: whether to use batch normalization where supported

    Returns:
        Compiled TensorFlow / Keras model
    """
    validate_net(net)

    n_classes = len(knots)

    if n_classes < 2 and "localise" not in net:
        raise ValueError(
            f"Classification model requires at least 2 classes, got {n_classes}"
        )

    if net == "FFNN":
        model = setup_NN(
            in_layer,
            n_classes,
            "relu",
            tf.keras.optimizers.Adam(learning_rate=0.001),
            norm,
        )

    elif net == "FFNN2":
        model = setup_NN2(
            in_layer,
            n_classes,
            "relu",
            tf.keras.optimizers.Adam(learning_rate=0.001),
            norm,
        )

    elif net == "CNN":
        model = setup_CNN(
            in_layer,
            n_classes,
            "relu",
            tf.keras.optimizers.Adam(learning_rate=0.001),
            norm,
        )

    elif net == "CNN_STS":
        model = setup_CNN_STS(
            in_layer,
            n_classes,
            "relu",
            tf.keras.optimizers.Adam(learning_rate=0.001),
            norm,
        )

    elif net == "RNN":
        model = setup_RNN(
            in_layer,
            n_classes,
            "tanh",
            tf.keras.optimizers.Adam(learning_rate=0.00001),
            norm,
        )

    elif net == "RNN2":
        model = setup_RNN2(
            in_layer,
            n_classes,
            "tanh",
            tf.keras.optimizers.Adam(learning_rate=0.00001),
            norm,
        )

    elif net == "RNN2b":
        model = setup_RNN2b(
            in_layer,
            n_classes,
            "tanh",
            tf.keras.optimizers.Adam(learning_rate=0.00001),
            norm,
        )

    elif net == "localiseFFNN":
        model = localise_setup_NN(
            in_layer,
            n_classes,
            "relu",
            tf.keras.optimizers.Adam(learning_rate=0.001),
            norm,
        )

    elif net == "localiseRNN":
        model = localise_setup_RNN(
            in_layer,
            n_classes,
            "tanh",
            tf.keras.optimizers.Adam(learning_rate=0.00001),
            norm,
        )

    elif net == "randFOR":
        if tfdf is None:
            raise ImportError(
                "tensorflow_decision_forests is not installed in this environment. "
                "On Windows use another network type or run this project in WSL/Linux."
            )
        model = tfdf.keras.RandomForestModel()

    else:
        raise ValueError(f"Network not available: {net}")

    return model