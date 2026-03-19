#bierze dane z processed
#porównuje klasy 0_1 i 3_1
#liczy statystyki dla sta i sts
#sprawdza proste cechy rozróżniające klasy
#szuka podejrzanie podobnych próbek
#robi wykresy przykładowych próbek
#robi PCA na prostych cechach

from pathlib import Path
import json
import hashlib

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATA_ROOT = Path(r"C:\Users\Adam\.vscode\grafy\data\processed")
OUTPUT_DIR = Path(r"C:\Users\Adam\.vscode\grafy\data\diagnostics")
CLASSES = ["0_1", "3_1"]

# ile próbek brać do szybkiej diagnostyki / wykresów
N_SAMPLES_STATS = 200
N_SAMPLES_PLOTS = 3
N_SAMPLES_PCA = 150


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_file_list(repr_name: str, feature_name: str, class_name: str, extension: str) -> list[Path]:
    folder = DATA_ROOT / repr_name / feature_name / class_name
    files = sorted(folder.glob(f"*{extension}"))
    return files


def sample_files(files: list[Path], n: int) -> list[Path]:
    if len(files) <= n:
        return files
    return files[:n]


def load_sta(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    values = data[:, -1]
    return values.astype(np.float32)


def load_sts(path: Path) -> np.ndarray:
    data = np.load(path).astype(np.float32)
    return data


def robust_stats(x: np.ndarray) -> dict:
    flat = x.reshape(-1)
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "median": float(np.median(flat)),
        "q01": float(np.quantile(flat, 0.01)),
        "q05": float(np.quantile(flat, 0.05)),
        "q25": float(np.quantile(flat, 0.25)),
        "q75": float(np.quantile(flat, 0.75)),
        "q95": float(np.quantile(flat, 0.95)),
        "q99": float(np.quantile(flat, 0.99)),
        "mean_abs": float(np.mean(np.abs(flat))),
        "max_abs": float(np.max(np.abs(flat))),
        "l2_norm": float(np.linalg.norm(flat)),
    }


def aggregate_stats(arrays: list[np.ndarray]) -> dict:
    per_sample = [robust_stats(a) for a in arrays]
    keys = per_sample[0].keys()
    summary = {}
    for key in keys:
        vals = np.array([d[key] for d in per_sample], dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return summary


def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_exact_duplicates(files: list[Path]) -> dict:
    hashes = {}
    duplicates = {}
    for path in files:
        h = file_hash(path)
        if h in hashes:
            duplicates.setdefault(h, [hashes[h]]).append(path.name)
        else:
            hashes[h] = path.name
    return duplicates


def extract_simple_features_sta(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1)
    return np.array([
        np.mean(flat),
        np.std(flat),
        np.mean(np.abs(flat)),
        np.max(np.abs(flat)),
        np.quantile(flat, 0.01),
        np.quantile(flat, 0.05),
        np.quantile(flat, 0.25),
        np.quantile(flat, 0.50),
        np.quantile(flat, 0.75),
        np.quantile(flat, 0.95),
        np.quantile(flat, 0.99),
        np.linalg.norm(flat),
    ], dtype=np.float32)


def extract_simple_features_sts(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1)
    return np.array([
        np.mean(flat),
        np.std(flat),
        np.mean(np.abs(flat)),
        np.max(np.abs(flat)),
        np.quantile(flat, 0.01),
        np.quantile(flat, 0.05),
        np.quantile(flat, 0.25),
        np.quantile(flat, 0.50),
        np.quantile(flat, 0.75),
        np.quantile(flat, 0.95),
        np.quantile(flat, 0.99),
        np.linalg.norm(flat),
    ], dtype=np.float32)


def plot_sta_examples(repr_name: str, class_name: str, files: list[Path], out_dir: Path) -> None:
    ensure_dir(out_dir)
    for i, path in enumerate(sample_files(files, N_SAMPLES_PLOTS), start=1):
        x = load_sta(path)
        plt.figure(figsize=(10, 3))
        plt.plot(x, linewidth=1)
        plt.title(f"{repr_name} / STA / {class_name} / {path.stem}")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(out_dir / f"{repr_name}_sta_{class_name}_{i}_{path.stem}.png", dpi=150)
        plt.close()


def plot_sts_examples(repr_name: str, class_name: str, files: list[Path], out_dir: Path) -> None:
    ensure_dir(out_dir)
    for i, path in enumerate(sample_files(files, N_SAMPLES_PLOTS), start=1):
        x = load_sts(path)

        # 1knot: (255,255), 3loops: (3,255,255)
        if x.ndim == 2:
            mats = [x]
        elif x.ndim == 3:
            mats = [x[j] for j in range(min(3, x.shape[0]))]
        else:
            continue

        for j, mat in enumerate(mats, start=1):
            plt.figure(figsize=(5, 4))
            plt.imshow(mat, cmap="coolwarm", aspect="auto")
            plt.colorbar()
            plt.title(f"{repr_name} / STS / {class_name} / {path.stem} / part {j}")
            plt.tight_layout()
            plt.savefig(out_dir / f"{repr_name}_sts_{class_name}_{i}_{path.stem}_part{j}.png", dpi=150)
            plt.close()


def run_pca(features: np.ndarray, labels: np.ndarray, label_names: list[str], title: str, out_path: Path) -> None:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(x_scaled)

    plt.figure(figsize=(6, 5))
    for idx, label_name in enumerate(label_names):
        mask = labels == idx
        plt.scatter(
            x_pca[mask, 0],
            x_pca[mask, 1],
            s=18,
            alpha=0.7,
            label=label_name,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_feature_matrix_sta(repr_name: str, class_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []

    for class_idx, class_name in enumerate(class_names):
        files = get_file_list(repr_name, "sta", class_name, ".txt")
        for path in sample_files(files, N_SAMPLES_PCA):
            x = load_sta(path)
            xs.append(extract_simple_features_sta(x))
            ys.append(class_idx)

    return np.stack(xs), np.array(ys, dtype=np.int32)


def build_feature_matrix_sts(repr_name: str, class_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []

    for class_idx, class_name in enumerate(class_names):
        files = get_file_list(repr_name, "sts", class_name, ".npy")
        for path in sample_files(files, N_SAMPLES_PCA):
            x = load_sts(path)
            xs.append(extract_simple_features_sts(x))
            ys.append(class_idx)

    return np.stack(xs), np.array(ys, dtype=np.int32)


def diagnose_one_feature(repr_name: str, feature_name: str, class_names: list[str]) -> dict:
    result = {
        "representation": repr_name,
        "feature": feature_name,
        "classes": {},
    }

    for class_name in class_names:
        extension = ".txt" if feature_name == "sta" else ".npy"
        files = get_file_list(repr_name, feature_name, class_name, extension)
        chosen = sample_files(files, N_SAMPLES_STATS)

        if feature_name == "sta":
            arrays = [load_sta(p) for p in chosen]
        else:
            arrays = [load_sts(p) for p in chosen]

        result["classes"][class_name] = {
            "n_files_total": len(files),
            "n_files_used": len(chosen),
            "aggregated_stats": aggregate_stats(arrays),
            "exact_duplicates": find_exact_duplicates(chosen),
        }

    return result


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    all_results = {}

    # 1) statystyki i duplikaty
    for repr_name in ["1knot", "3loops"]:
        for feature_name in ["sta", "sts"]:
            key = f"{repr_name}_{feature_name}"
            print(f"Diagnosing: {key}")
            result = diagnose_one_feature(repr_name, feature_name, CLASSES)
            all_results[key] = result

    save_json(all_results, OUTPUT_DIR / "diagnostics_summary.json")

    # 2) przykładowe wykresy
    plots_dir = OUTPUT_DIR / "plots"
    for repr_name in ["1knot", "3loops"]:
        for class_name in CLASSES:
            sta_files = get_file_list(repr_name, "sta", class_name, ".txt")
            sts_files = get_file_list(repr_name, "sts", class_name, ".npy")
            plot_sta_examples(repr_name, class_name, sta_files, plots_dir / "sta")
            plot_sts_examples(repr_name, class_name, sts_files, plots_dir / "sts")

    # 3) PCA na prostych cechach
    for repr_name in ["1knot", "3loops"]:
        x_sta, y_sta = build_feature_matrix_sta(repr_name, CLASSES)
        run_pca(
            x_sta,
            y_sta,
            CLASSES,
            title=f"PCA simple features - {repr_name} STA",
            out_path=OUTPUT_DIR / "pca" / f"pca_{repr_name}_sta.png",
        )

        x_sts, y_sts = build_feature_matrix_sts(repr_name, CLASSES)
        run_pca(
            x_sts,
            y_sts,
            CLASSES,
            title=f"PCA simple features - {repr_name} STS",
            out_path=OUTPUT_DIR / "pca" / f"pca_{repr_name}_sts.png",
        )

    print(f"Done. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()