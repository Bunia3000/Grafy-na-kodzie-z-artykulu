import numpy as np
from pathlib import Path

DATA_ROOT = Path(r"C:\Users\Adam\.vscode\grafy\data\processed")

def inspect_sta(dtype="1knot", class_name="0_1", n_samples=50):
    folder = DATA_ROOT / dtype / "sta" / class_name
    files = sorted(folder.glob("*.txt"))[:n_samples]

    values = []

    for f in files:
        data = np.loadtxt(f)
        values.append(data[:, -1])  # bierzemy kolumnę value

    values = np.concatenate(values)

    print(f"\nSTA {dtype} {class_name}")
    print(f"min: {values.min()}")
    print(f"max: {values.max()}")
    print(f"mean: {values.mean()}")
    print(f"std: {values.std()}")


def inspect_sts(dtype="1knot", class_name="0_1", n_samples=50):
    folder = DATA_ROOT / dtype / "sts" / class_name
    files = sorted(folder.glob("*.npy"))[:n_samples]

    values = []

    for f in files:
        data = np.load(f)
        values.append(data.flatten())

    values = np.concatenate(values)

    print(f"\nSTS {dtype} {class_name}")
    print(f"min: {values.min()}")
    print(f"max: {values.max()}")
    print(f"mean: {values.mean()}")
    print(f"std: {values.std()}")


if __name__ == "__main__":
    for dtype in ["1knot", "3loops"]:
        for cls in ["0_1", "3_1"]:
            inspect_sta(dtype, cls)
            inspect_sts(dtype, cls)