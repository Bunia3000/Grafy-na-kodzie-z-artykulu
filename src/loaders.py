import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_sample_id(sample_id):
    """
    Parse sample ID in the format:
    "4_1/0007" -> ("4_1", "0007")
    """
    try:
        class_name, file_stem = sample_id.split("/")
    except ValueError as exc:
        raise ValueError(f"Invalid sample_id format: {sample_id}") from exc

    return class_name, file_stem


def load_1knot_xyz(path):
    """
    Load one 1knot XYZ sample.

    Expected file format:
    index x y z

    Returns:
        np.ndarray with shape (256, 3)
    """
    data = np.loadtxt(path)

    if data.shape != (256, 4):
        raise ValueError(
            f"Invalid shape in {path}, expected (256, 4), got {data.shape}"
        )

    indices = data[:, 0].astype(int)
    expected_indices = np.arange(256)
    if not np.array_equal(indices, expected_indices):
        raise ValueError(f"Invalid indices in {path}, expected 0..255")

    coords = data[:, 1:4]

    return coords.astype(np.float32)


def load_3loops_xyz(path):
    """
    Load one 3loops XYZ sample.

    Expected file format:
    loop_id point_id x y z

    Returns:
        np.ndarray with shape (3, 256, 3)
    """
    data = np.loadtxt(path)

    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError(
            f"Invalid format in {path}, expected 5 columns, got shape {data.shape}"
        )

    result = np.zeros((3, 256, 3), dtype=np.float32)

    for loop_id in range(3):
        loop_data = data[data[:, 0] == loop_id]

        if len(loop_data) != 256:
            raise ValueError(
                f"Loop {loop_id} in {path} has {len(loop_data)} points instead of 256"
            )

        point_ids = loop_data[:, 1].astype(int)
        expected_point_ids = np.arange(256)
        if not np.array_equal(point_ids, expected_point_ids):
            raise ValueError(
                f"Invalid point_id sequence for loop {loop_id} in {path}, expected 0..255"
            )

        coords = loop_data[:, 2:5]
        result[loop_id] = coords

    return result


def load_1knot_sta(path):
    """
    Load one 1knot StA sample.

    Expected file format:
    index value

    Returns:
        np.ndarray with shape (255, 1)
    """
    data = np.loadtxt(path)

    if data.shape != (255, 2):
        raise ValueError(
            f"Invalid shape in {path}, expected (255, 2), got {data.shape}"
        )

    indices = data[:, 0].astype(int)
    expected_indices = np.arange(255)
    if not np.array_equal(indices, expected_indices):
        raise ValueError(f"Invalid indices in {path}, expected 0..254")

    values = data[:, 1].reshape(255, 1)

    return values.astype(np.float32)


def load_3loops_sta(path):
    """
    Load one 3loops StA sample.

    Expected file format:
    loop_id index value

    Returns:
        np.ndarray with shape (3, 255, 1)
    """
    data = np.loadtxt(path)

    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(
            f"Invalid format in {path}, expected 3 columns, got shape {data.shape}"
        )

    result = np.zeros((3, 255, 1), dtype=np.float32)

    for loop_id in range(3):
        loop_data = data[data[:, 0] == loop_id]

        if len(loop_data) != 255:
            raise ValueError(
                f"Loop {loop_id} in {path} has {len(loop_data)} values instead of 255"
            )

        indices = loop_data[:, 1].astype(int)
        expected_indices = np.arange(255)
        if not np.array_equal(indices, expected_indices):
            raise ValueError(
                f"Invalid indices for loop {loop_id} in {path}, expected 0..254"
            )

        values = loop_data[:, 2].reshape(255, 1)
        result[loop_id] = values

    return result


def load_1knot_sts(path):
    """
    Load one 1knot StS sample.

    Expected file format:
    .npy array with raw shape (255, 255)

    Returns:
        np.ndarray with shape (255, 255, 1)
    """
    data = np.load(path)

    if data.shape != (255, 255):
        raise ValueError(
            f"Invalid shape in {path}, expected (255, 255), got {data.shape}"
        )

    data = data[..., np.newaxis]

    return data.astype(np.float32)


def load_3loops_sts(path):
    """
    Load one 3loops StS sample.

    Expected file format:
    .npy array with raw shape (3, 255, 255)

    Returns:
        np.ndarray with shape (3, 255, 255, 1)
    """
    data = np.load(path)

    if data.shape != (3, 255, 255):
        raise ValueError(
            f"Invalid shape in {path}, expected (3, 255, 255), got {data.shape}"
        )

    data = data[..., np.newaxis]

    return data.astype(np.float32)


def get_dtype_config(dtype):
    """
    Return configuration for a given dtype.
    """
    configs = {
        "1knot_xyz": {
            "subdir": "1knot/xyz",
            "extension": ".xyz",
            "expected_shape": (256, 3),
            "loader_fn": load_1knot_xyz,
        },
        "1knot_sta": {
            "subdir": "1knot/sta",
            "extension": ".txt",
            "expected_shape": (255, 1),
            "loader_fn": load_1knot_sta,
        },
        "1knot_sts": {
            "subdir": "1knot/sts",
            "extension": ".npy",
            "expected_shape": (255, 255, 1),
            "loader_fn": load_1knot_sts,
        },
        "3loops_xyz": {
            "subdir": "3loops/xyz",
            "extension": ".xyz",
            "expected_shape": (3, 256, 3),
            "loader_fn": load_3loops_xyz,
        },
        "3loops_sta": {
            "subdir": "3loops/sta",
            "extension": ".txt",
            "expected_shape": (3, 255, 1),
            "loader_fn": load_3loops_sta,
        },
        "3loops_sts": {
            "subdir": "3loops/sts",
            "extension": ".npy",
            "expected_shape": (3, 255, 255, 1),
            "loader_fn": load_3loops_sts,
        },
    }

    if dtype not in configs:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return configs[dtype]


def get_expected_shape(dtype):
    """
    Return expected array shape for a given dtype.
    """
    return get_dtype_config(dtype)["expected_shape"]


def build_sample_path(data_root, dtype, sample_id):
    """
    Build full path to a sample file from:
    - data_root
    - dtype
    - sample_id ("class/file_stem")
    """
    config = get_dtype_config(dtype)
    class_name, file_stem = parse_sample_id(sample_id)

    path = (
        Path(data_root)
        / config["subdir"]
        / class_name
        / f"{file_stem}{config['extension']}"
    )

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return path


def validate_array_shape(array, expected_shape, sample_id):
    """
    Validate array shape.
    """
    if array.shape != expected_shape:
        raise ValueError(
            f"Invalid shape for sample {sample_id}: "
            f"expected {expected_shape}, got {array.shape}"
        )


def validate_array_values(array, sample_id):
    """
    Validate that array contains no NaN or inf.
    """
    if np.isnan(array).any():
        raise ValueError(f"NaN values found in sample {sample_id}")

    if np.isinf(array).any():
        raise ValueError(f"Inf values found in sample {sample_id}")


def load_split_json(split_file):
    """
    Load split JSON file.
    """
    with open(split_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def get_split_sample_ids(split_data, split_name):
    """
    Get list of sample IDs for a given split:
    train / val / test
    """
    if split_name not in split_data:
        raise ValueError(f"Split '{split_name}' not found in JSON")

    return split_data[split_name]


def load_single_sample(data_root, dtype, sample_id):
    """
    Load one sample of any supported dtype.
    """
    config = get_dtype_config(dtype)
    path = build_sample_path(data_root, dtype, sample_id)

    array = config["loader_fn"](path)

    validate_array_shape(array, config["expected_shape"], sample_id)
    validate_array_values(array, sample_id)

    def normalize(x):
        mean = np.mean(x)
        std = np.std(x)

        if std < 1e-8:
            return x  # unikamy dzielenia przez 0

        return (x - mean) / std

    array = normalize(array)

    return array


def load_dataset_from_split(data_root, dtype, split_file, split_name, class_to_label):
    """
    Load full dataset split and return:
    - tf.data.Dataset
    - list of sample_ids

    Args:
        data_root: root directory, e.g. .../data/processed
        dtype: one of the 6 supported dtypes
        split_file: path to JSON file with train/val/test split
        split_name: "train", "val", or "test"
        class_to_label: dict mapping class names to integer labels

    Returns:
        dataset: tf.data.Dataset yielding (x, y)
        sample_ids: list of sample IDs used in the split
    """
    split_data = load_split_json(split_file)
    sample_ids = get_split_sample_ids(split_data, split_name)

    xs = []
    ys = []

    for sample_id in sample_ids:
        class_name, _ = parse_sample_id(sample_id)

        if class_name not in class_to_label:
            raise ValueError(
                f"Class '{class_name}' from sample '{sample_id}' "
                f"is missing in class_to_label"
            )

        x = load_single_sample(data_root, dtype, sample_id)
        y = class_to_label[class_name]

        xs.append(x)
        ys.append(y)

    if not xs:
        raise ValueError(
            f"No samples found for split '{split_name}' in split file '{split_file}'"
        )

    xs = np.stack(xs).astype(np.float32)
    ys = np.array(ys, dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((xs, ys))

    return dataset, sample_ids
