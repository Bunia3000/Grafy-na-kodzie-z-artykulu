import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from helpers import generate_model
from loaders import (
    get_expected_shape,
    load_dataset_from_split,
    load_split_json,
)

# Mniej komunikatów z TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/test models on theta-curve datasets."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to processed data root, e.g. C:/Users/Adam/.vscode/grafy/data/processed",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        required=True,
        choices=[
            "1knot_xyz",
            "1knot_sta",
            "1knot_sts",
            "3loops_xyz",
            "3loops_sta",
            "3loops_sts",
        ],
        help="Data type to use.",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        required=True,
        help="Path to JSON split file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "test"],
        help="Run mode.",
    )
    parser.add_argument(
        "--net",
        type=str,
        default="FFNN",
        help="Model type, e.g. FFNN, CNN, RNN.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Enable batch normalization inside the model, if supported.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory to save model and results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_class_names_from_split(split_file):
    """
    Extract class names from split JSON.

    Supported logic:
    1. If JSON contains top-level key 'classes', use it.
    2. Otherwise infer classes from sample IDs in train/val/test.
    """
    split_data = load_split_json(split_file)

    if "classes" in split_data:
        class_names = sorted(split_data["classes"])
        return class_names

    sample_ids = []
    for split_name in ["train", "val", "test"]:
        if split_name in split_data:
            sample_ids.extend(split_data[split_name])

    class_names = sorted({sample_id.split("/")[0] for sample_id in sample_ids})
    return class_names


def build_class_to_label(class_names):
    """
    Example:
    ['0_1', '3_1'] -> {'0_1': 0, '3_1': 1}
    """
    return {class_name: idx for idx, class_name in enumerate(class_names)}


def prepare_datasets(data_root, dtype, split_file, class_to_label, batch_size, seed):
    """
    Load train/val/test datasets and prepare them for training/testing.
    """
    train_dataset, train_ids = load_dataset_from_split(
        data_root=data_root,
        dtype=dtype,
        split_file=split_file,
        split_name="train",
        class_to_label=class_to_label,
    )
    val_dataset, val_ids = load_dataset_from_split(
        data_root=data_root,
        dtype=dtype,
        split_file=split_file,
        split_name="val",
        class_to_label=class_to_label,
    )
    test_dataset, test_ids = load_dataset_from_split(
        data_root=data_root,
        dtype=dtype,
        split_file=split_file,
        split_name="test",
        class_to_label=class_to_label,
    )

    train_dataset = train_dataset.shuffle(buffer_size=len(train_ids), seed=seed)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_ids,
        val_ids,
        test_ids,
    )


def save_training_history(history, out_path):
    """
    Save Keras history as CSV with one row per epoch.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    history_dict = history.history
    keys = list(history_dict.keys())
    n_epochs = len(history_dict[keys[0]])

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)

        for epoch_idx in range(n_epochs):
            row = [epoch_idx + 1] + [history_dict[key][epoch_idx] for key in keys]
            writer.writerow(row)


def save_confusion_matrix(conf_matrix, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, conf_matrix.astype(int), fmt="%d")


def compute_classification_metrics(conf_matrix, class_names):
    """
    Compute per-class precision / recall / f1 and macro averages
    from confusion matrix.
    """
    conf_matrix = np.asarray(conf_matrix, dtype=np.float64)
    n_classes = conf_matrix.shape[0]

    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for i in range(n_classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[class_names[i]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(conf_matrix[i, :].sum()),
        }

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_metrics = {
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }

    return per_class, macro_metrics


def save_summary(summary_dict, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False)


def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs,
    checkpoint_dir,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / "best_model.keras"
    history_path = checkpoint_dir / "training_history.csv"

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        restore_best_weights=True,
        min_delta=0.001,
        verbose=1,
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    start_time = time.time()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint],
    )

    elapsed_time = time.time() - start_time

    save_training_history(history, history_path)

    return history, elapsed_time


def test_model(model, test_dataset, checkpoint_dir, class_names):
    checkpoint_dir = Path(checkpoint_dir)
    model_path = checkpoint_dir / "best_model.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)

    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)

    y_true = []
    y_pred = []

    for x_batch, y_batch in test_dataset:
        preds = model.predict(x_batch, verbose=0)
        preds = np.argmax(preds, axis=1)

        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(preds.tolist())

    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)

    conf_matrix = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=len(class_names),
    ).numpy()

    per_class_metrics, macro_metrics = compute_classification_metrics(
        conf_matrix,
        class_names,
    )

    save_confusion_matrix(conf_matrix, checkpoint_dir / "confusion_matrix.txt")

    test_summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_metrics": macro_metrics,
        "per_class_metrics": per_class_metrics,
    }

    save_summary(test_summary, checkpoint_dir / "test_summary.json")

    return test_summary, conf_matrix


def main():
    args = parse_args()
    set_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    class_names = get_class_names_from_split(args.split_file)
    class_to_label = build_class_to_label(class_names)

    input_shape = get_expected_shape(args.dtype)

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_ids,
        val_ids,
        test_ids,
    ) = prepare_datasets(
        data_root=args.data_root,
        dtype=args.dtype,
        split_file=args.split_file,
        class_to_label=class_to_label,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    metadata = {
        "data_root": args.data_root,
        "dtype": args.dtype,
        "split_file": args.split_file,
        "mode": args.mode,
        "net": args.net,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "norm": args.norm,
        "seed": args.seed,
        "input_shape": input_shape,
        "class_names": class_names,
        "class_to_label": class_to_label,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
    }
    save_summary(metadata, checkpoint_dir / "run_metadata.json")

    model = generate_model(
        net=args.net,
        in_layer=input_shape,
        knots=class_names,
        norm=args.norm,
    )

    if args.mode == "train":
        history, elapsed_time = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=args.epochs,
            checkpoint_dir=checkpoint_dir,
        )

        train_summary = {
            "elapsed_time_sec": elapsed_time,
            "best_val_loss": float(np.min(history.history["val_loss"])),
            "best_val_accuracy": float(np.max(history.history["val_accuracy"])),
            "last_train_loss": float(history.history["loss"][-1]),
            "last_train_accuracy": float(history.history["accuracy"][-1]),
        }
        save_summary(train_summary, checkpoint_dir / "train_summary.json")

        print("Training finished.")
        print(f"Results saved to: {checkpoint_dir}")

    elif args.mode == "test":
        test_summary, conf_matrix = test_model(
            model=model,
            test_dataset=test_dataset,
            checkpoint_dir=checkpoint_dir,
            class_names=class_names,
        )

        print("Test finished.")
        print("Confusion matrix:")
        print(conf_matrix)
        print(f"Results saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()