from pathlib import Path
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_ROOT = Path(r"C:\Users\Adam\.vscode\grafy\data\processed")
SPLITS_DIR = Path(r"C:\Users\Adam\.vscode\grafy\splits")
OUTPUT_DIR = Path(r"C:\Users\Adam\.vscode\grafy\data\simple_benchmark")

EXPERIMENTS = [
    {
        "name": "01v31",
        "classes": ["0_1", "3_1"],
        "split_file": SPLITS_DIR / "01v31_seed42.json",
    },
    {
        "name": "31v41",
        "classes": ["3_1", "4_1"],
        "split_file": SPLITS_DIR / "31v41_seed42.json",
    },
]

DATASETS = [
    ("1knot", "sta", ".txt"),
    ("1knot", "sts", ".npy"),
    ("3loops", "sta", ".txt"),
    ("3loops", "sts", ".npy"),
]

FEATURE_MODES = [
    "raw",
    "clipped_1_99",
]

RANDOM_STATE = 42


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_sta(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    return data[:, -1].astype(np.float32)


def load_sts(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def load_sample(repr_name: str, feature_name: str, sample_id: str) -> np.ndarray:
    class_name, stem = sample_id.split("/")
    extension = ".txt" if feature_name == "sta" else ".npy"
    path = DATA_ROOT / repr_name / feature_name / class_name / f"{stem}{extension}"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    if feature_name == "sta":
        return load_sta(path)

    return load_sts(path)


def clip_per_sample(x: np.ndarray, q_low: float = 0.01, q_high: float = 0.99) -> np.ndarray:
    flat = x.reshape(-1)
    low = np.quantile(flat, q_low)
    high = np.quantile(flat, q_high)
    return np.clip(x, low, high)


def preprocess_sample(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return x

    if mode == "clipped_1_99":
        return clip_per_sample(x, 0.01, 0.99)

    raise ValueError(f"Unsupported feature mode: {mode}")


def extract_generic_features(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1).astype(np.float64)

    mean_val = np.mean(flat)
    std_val = np.std(flat)
    abs_flat = np.abs(flat)

    features = [
        mean_val,
        std_val,
        np.median(flat),
        np.mean(abs_flat),
        np.max(abs_flat),
        np.quantile(flat, 0.01),
        np.quantile(flat, 0.05),
        np.quantile(flat, 0.25),
        np.quantile(flat, 0.50),
        np.quantile(flat, 0.75),
        np.quantile(flat, 0.95),
        np.quantile(flat, 0.99),
        np.linalg.norm(flat),
        np.mean(flat > 0),
        np.mean(flat < 0),
        np.mean(flat == 0),
    ]

    return np.array(features, dtype=np.float32)


def extract_sta_features(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1).astype(np.float64)

    diff_1 = np.diff(flat)
    abs_diff_1 = np.abs(diff_1)

    generic = extract_generic_features(x)

    extra = np.array(
        [
            np.mean(diff_1),
            np.std(diff_1),
            np.mean(abs_diff_1),
            np.max(abs_diff_1),
            np.quantile(diff_1, 0.01) if len(diff_1) > 0 else 0.0,
            np.quantile(diff_1, 0.99) if len(diff_1) > 0 else 0.0,
        ],
        dtype=np.float32,
    )

    return np.concatenate([generic, extra])


def extract_sts_features(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        mats = [x]
    elif x.ndim == 3:
        mats = [x[i] for i in range(x.shape[0])]
    else:
        raise ValueError(f"Unsupported STS ndim: {x.ndim}")

    generic = extract_generic_features(x)

    per_mat_features = []
    for mat in mats:
        mat = mat.astype(np.float64)
        diag = np.diag(mat)
        offdiag_mask = ~np.eye(mat.shape[0], dtype=bool)
        offdiag = mat[offdiag_mask]

        per_mat_features.extend(
            [
                np.mean(diag),
                np.std(diag),
                np.mean(np.abs(diag)),
                np.mean(offdiag),
                np.std(offdiag),
                np.mean(np.abs(offdiag)),
            ]
        )

    return np.concatenate([generic, np.array(per_mat_features, dtype=np.float32)])


def extract_features(x: np.ndarray, feature_name: str) -> np.ndarray:
    if feature_name == "sta":
        return extract_sta_features(x)

    if feature_name == "sts":
        return extract_sts_features(x)

    raise ValueError(f"Unsupported feature name: {feature_name}")


def build_xy(
    split_data: dict,
    split_name: str,
    class_to_label: dict,
    repr_name: str,
    feature_name: str,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []

    for sample_id in split_data[split_name]:
        class_name, _ = sample_id.split("/")
        x = load_sample(repr_name, feature_name, sample_id)
        x = preprocess_sample(x, feature_mode)
        feats = extract_features(x, feature_name)

        xs.append(feats)
        ys.append(class_to_label[class_name])

    return np.stack(xs), np.array(ys, dtype=np.int32)


def evaluate_model(model, x_train, y_train, x_test, y_test) -> dict:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm,
    }


def get_split_path(experiment_name: str, dataset_key: str, default_split_path: Path) -> dict:
    filtered_split_path = SPLITS_DIR / f"{experiment_name}_{dataset_key}_filtered_seed42.json"

    result = {
        "raw": default_split_path,
        "filtered": None,
    }

    if filtered_split_path.exists():
        result["filtered"] = filtered_split_path

    return result


def run_one_benchmark(
    split_path: Path,
    class_names: list[str],
    repr_name: str,
    feature_name: str,
    feature_mode: str,
) -> dict:
    split_data = load_json(split_path)
    class_to_label = {class_name: idx for idx, class_name in enumerate(class_names)}

    x_train, y_train = build_xy(
        split_data=split_data,
        split_name="train",
        class_to_label=class_to_label,
        repr_name=repr_name,
        feature_name=feature_name,
        feature_mode=feature_mode,
    )
    x_test, y_test = build_xy(
        split_data=split_data,
        split_name="test",
        class_to_label=class_to_label,
        repr_name=repr_name,
        feature_name=feature_name,
        feature_mode=feature_mode,
    )

    logistic_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logistic_result = evaluate_model(
        logistic_model,
        x_train,
        y_train,
        x_test,
        y_test,
    )
    forest_result = evaluate_model(
        random_forest_model,
        x_train,
        y_train,
        x_test,
        y_test,
    )

    return {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(x_train.shape[1]),
        "logistic_regression": logistic_result,
        "random_forest": forest_result,
    }


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    full_report = {}

    for experiment in EXPERIMENTS:
        exp_name = experiment["name"]
        class_names = experiment["classes"]
        split_file = experiment["split_file"]

        full_report[exp_name] = {
            "classes": class_names,
            "datasets": {},
        }

        print(f"Running experiment: {exp_name}")

        for repr_name, feature_name, _ext in DATASETS:
            dataset_key = f"{repr_name}_{feature_name}"
            full_report[exp_name]["datasets"][dataset_key] = {}

            split_variants = get_split_path(exp_name, dataset_key, split_file)

            for split_variant_name, split_variant_path in split_variants.items():
                if split_variant_path is None:
                    continue

                full_report[exp_name]["datasets"][dataset_key][split_variant_name] = {}

                for feature_mode in FEATURE_MODES:
                    print(
                        f"  {dataset_key} | split={split_variant_name} | "
                        f"features={feature_mode}"
                    )

                    result = run_one_benchmark(
                        split_path=split_variant_path,
                        class_names=class_names,
                        repr_name=repr_name,
                        feature_name=feature_name,
                        feature_mode=feature_mode,
                    )

                    full_report[exp_name]["datasets"][dataset_key][split_variant_name][
                        feature_mode
                    ] = result

    save_json(full_report, OUTPUT_DIR / "benchmark_report.json")
    print(f"Done. Results saved to: {OUTPUT_DIR / 'benchmark_report.json'}")


if __name__ == "__main__":
    main()