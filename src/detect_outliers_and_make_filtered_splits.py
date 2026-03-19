#analizuje STA i STS,
#działa dla par:
#0_1 vs 3_1
#3_1 vs 4_1
#wykrywa outliery per próbka,
#zapisuje raport,
#tworzy nowe splity bez outlierów w train.

from pathlib import Path
import json

import numpy as np


DATA_ROOT = Path(r"C:\Users\Adam\.vscode\grafy\data\processed")
SPLITS_DIR = Path(r"C:\Users\Adam\.vscode\grafy\splits")
OUTPUT_DIR = Path(r"C:\Users\Adam\.vscode\grafy\data\outlier_diagnostics")

# Jakie eksperymenty analizować
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

# Jakie reprezentacje/cechy analizować
DATASETS = [
    ("1knot", "sta", ".txt"),
    ("1knot", "sts", ".npy"),
    ("3loops", "sta", ".txt"),
    ("3loops", "sts", ".npy"),
]

# Na jakich splitach wykrywać outliery
ANALYZE_SPLITS = ["train"]

# Z których splitów usuwać outliery
FILTER_SPLITS = ["train"]

# Próg robust-zscore (im mniejszy, tym agresywniejszy)
ROBUST_Z_THRESHOLD = 4.0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_sta(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    return data[:, -1].astype(np.float32)


def load_sts(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def load_sample(repr_name: str, feature_name: str, sample_id: str) -> np.ndarray:
    class_name, stem = sample_id.split("/")
    ext = ".txt" if feature_name == "sta" else ".npy"
    path = DATA_ROOT / repr_name / feature_name / class_name / f"{stem}{ext}"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    if feature_name == "sta":
        return load_sta(path)
    return load_sts(path)


def sample_features(x: np.ndarray) -> dict:
    flat = x.reshape(-1).astype(np.float64)

    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "mean_abs": float(np.mean(np.abs(flat))),
        "max_abs": float(np.max(np.abs(flat))),
        "l2_norm": float(np.linalg.norm(flat)),
        "q01": float(np.quantile(flat, 0.01)),
        "q99": float(np.quantile(flat, 0.99)),
    }


def robust_zscore(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad < 1e-12:
        return np.zeros_like(values, dtype=np.float64)

    return 0.6745 * (values - median) / mad


def find_outliers_for_group(rows: list[dict], threshold: float) -> set[str]:
    """
    Outlier detection inside one class.
    Używamy log1p dla cech dodatnich, żeby ogromne piki nie dominowały aż tak brutalnie.
    """
    if not rows:
        return set()

    max_abs = np.array([np.log1p(r["features"]["max_abs"]) for r in rows], dtype=np.float64)
    l2_norm = np.array([np.log1p(r["features"]["l2_norm"]) for r in rows], dtype=np.float64)
    std_vals = np.array([np.log1p(r["features"]["std"]) for r in rows], dtype=np.float64)

    z_max = np.abs(robust_zscore(max_abs))
    z_l2 = np.abs(robust_zscore(l2_norm))
    z_std = np.abs(robust_zscore(std_vals))

    flagged = set()
    for row, a, b, c in zip(rows, z_max, z_l2, z_std):
        if max(a, b, c) > threshold:
            flagged.add(row["sample_id"])

    return flagged


def load_split(split_file: Path) -> dict:
    with split_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def analyze_experiment(experiment: dict) -> None:
    exp_name = experiment["name"]
    class_names = experiment["classes"]
    split_file = experiment["split_file"]

    split_data = load_split(split_file)

    report = {
        "experiment": exp_name,
        "classes": class_names,
        "split_file": str(split_file),
        "robust_z_threshold": ROBUST_Z_THRESHOLD,
        "datasets": {},
    }

    # Nowe splity będziemy budować na bazie oryginalnych
    filtered_split = json.loads(json.dumps(split_data))

    for repr_name, feature_name, _ext in DATASETS:
        key = f"{repr_name}_{feature_name}"
        report["datasets"][key] = {
            "classes": {},
            "flagged_outliers": {},
        }

        all_flagged = set()

        for class_name in class_names:
            rows = []

            for split_name in ANALYZE_SPLITS:
                for sample_id in split_data.get(split_name, []):
                    cls, _ = sample_id.split("/")
                    if cls != class_name:
                        continue

                    x = load_sample(repr_name, feature_name, sample_id)
                    feats = sample_features(x)

                    rows.append(
                        {
                            "sample_id": sample_id,
                            "split": split_name,
                            "features": feats,
                        }
                    )

            flagged = find_outliers_for_group(rows, ROBUST_Z_THRESHOLD)
            all_flagged |= flagged

            report["datasets"][key]["classes"][class_name] = {
                "n_analyzed": len(rows),
                "n_flagged": len(flagged),
                "flagged_sample_ids": sorted(flagged),
            }

        report["datasets"][key]["flagged_outliers"] = sorted(all_flagged)

        # tworzymy split odfiltrowany tylko dla FILTER_SPLITS
        filtered_name = f"{exp_name}_{key}_filtered_seed42.json"
        filtered_path = SPLITS_DIR / filtered_name

        filtered_local = json.loads(json.dumps(split_data))
        for split_name in FILTER_SPLITS:
            filtered_local[split_name] = [
                sid for sid in filtered_local.get(split_name, [])
                if sid not in all_flagged
            ]

        # metadata
        filtered_local["outlier_filter"] = {
            "based_on": key,
            "threshold": ROBUST_Z_THRESHOLD,
            "removed_from_splits": FILTER_SPLITS,
            "removed_sample_ids": sorted(all_flagged),
        }

        save_json(filtered_local, filtered_path)
        report["datasets"][key]["filtered_split_file"] = str(filtered_path)

    save_json(report, OUTPUT_DIR / f"{exp_name}_outlier_report.json")


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    for experiment in EXPERIMENTS:
        print(f"Analyzing experiment: {experiment['name']}")
        analyze_experiment(experiment)

    print(f"Done. Reports saved to: {OUTPUT_DIR}")
    print(f"Filtered splits saved to: {SPLITS_DIR}")


if __name__ == "__main__":
    main()