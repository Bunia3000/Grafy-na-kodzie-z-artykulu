import json
import random
from pathlib import Path


REFERENCE_DATA_DIR = Path(r"C:\Users\Adam\.vscode\grafy\data\processed\1knot\xyz")
OUTPUT_DIR = Path(r"C:\Users\Adam\.vscode\grafy\splits")

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def list_sample_ids(data_dir, class_names):
    """
    Return dict:
    {
        "0_1": ["0_1/0001", "0_1/0002", ...],
        "3_1": ["3_1/0001", "3_1/0002", ...],
    }
    """
    samples_by_class = {}

    for class_name in class_names:
        class_dir = data_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        files = sorted(class_dir.glob("*.xyz"))

        if not files:
            raise ValueError(f"No .xyz files found in class directory: {class_dir}")

        sample_ids = [f"{class_name}/{file.stem}" for file in files]
        samples_by_class[class_name] = sample_ids

    return samples_by_class


def downsample_classes(samples_by_class, seed):
    """
    Downsample all classes to the size of the smallest class.
    """
    rng = random.Random(seed)

    class_sizes = {class_name: len(samples) for class_name, samples in samples_by_class.items()}
    min_count = min(class_sizes.values())

    balanced = {}
    for class_name, samples in samples_by_class.items():
        samples_copy = samples.copy()
        rng.shuffle(samples_copy)
        selected = sorted(samples_copy[:min_count])
        balanced[class_name] = selected

    return balanced, min_count


def split_one_class(sample_ids, train_ratio, val_ratio, test_ratio):
    """
    Split one class list into train / val / test.
    Assumes sample_ids are already shuffled if needed.
    """
    n = len(sample_ids)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_ids = sample_ids[:n_train]
    val_ids = sample_ids[n_train:n_train + n_val]
    test_ids = sample_ids[n_train + n_val:]

    if len(train_ids) + len(val_ids) + len(test_ids) != n:
        raise ValueError("Split sizes do not sum up correctly.")

    return train_ids, val_ids, test_ids


def split_samples(balanced_samples_by_class, seed, train_ratio, val_ratio, test_ratio):
    """
    Split balanced sample IDs class-by-class and then merge.
    """
    rng = random.Random(seed)

    all_train = []
    all_val = []
    all_test = []

    for class_name, sample_ids in balanced_samples_by_class.items():
        sample_ids_copy = sample_ids.copy()
        rng.shuffle(sample_ids_copy)

        train_ids, val_ids, test_ids = split_one_class(
            sample_ids_copy,
            train_ratio,
            val_ratio,
            test_ratio,
        )

        all_train.extend(train_ids)
        all_val.extend(val_ids)
        all_test.extend(test_ids)

    rng.shuffle(all_train)
    rng.shuffle(all_val)
    rng.shuffle(all_test)

    return all_train, all_val, all_test


def validate_split_uniqueness(train_ids, val_ids, test_ids):
    """
    Ensure no overlaps between train / val / test.
    """
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    if train_set & val_set:
        raise ValueError("Overlap detected between train and val splits.")
    if train_set & test_set:
        raise ValueError("Overlap detected between train and test splits.")
    if val_set & test_set:
        raise ValueError("Overlap detected between val and test splits.")


def build_split_dict(class_names, seed, balanced_count_per_class, train_ids, val_ids, test_ids):
    return {
        "classes": sorted(class_names),
        "seed": seed,
        "balanced_count_per_class": balanced_count_per_class,
        "split_ratio": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def save_split_json(split_dict, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2, ensure_ascii=False)


def make_one_split(experiment_name, class_names, seed=SEED):
    """
    Create one split JSON for the given experiment.
    """
    samples_by_class = list_sample_ids(REFERENCE_DATA_DIR, class_names)

    balanced_samples_by_class, min_count = downsample_classes(samples_by_class, seed)

    train_ids, val_ids, test_ids = split_samples(
        balanced_samples_by_class,
        seed=seed,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )

    validate_split_uniqueness(train_ids, val_ids, test_ids)

    split_dict = build_split_dict(
        class_names=class_names,
        seed=seed,
        balanced_count_per_class=min_count,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )

    output_path = OUTPUT_DIR / f"{experiment_name}_seed{seed}.json"
    save_split_json(split_dict, output_path)

    print(f"Saved split: {output_path}")
    print(f"Classes: {class_names}")
    print(f"Balanced count per class: {min_count}")
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")
    print("-" * 60)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("01v31", ["0_1", "3_1"]),
        ("31v41", ["3_1", "4_1"]),
        ("01v31v41", ["0_1", "3_1", "4_1"]),
    ]

    for experiment_name, class_names in experiments:
        make_one_split(experiment_name, class_names, seed=SEED)

    print("All splits generated successfully.")


if __name__ == "__main__":
    main()