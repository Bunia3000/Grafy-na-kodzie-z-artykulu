from pathlib import Path
from collections import defaultdict
import numpy as np


PROJECT_ROOT = Path(r"C:\Users\Adam\.vscode\grafy")
INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "3loops" / "xyz_resampled_256_canonical"
STS_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "3loops" / "sts"
STA_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "3loops" / "sta"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "compute_3loops_sts_sta_report.txt"

EXPECTED_LOOPS = [0, 1, 2]
EXPECTED_POINTS_PER_LOOP = 256
EXPECTED_SEGMENTS_PER_LOOP = EXPECTED_POINTS_PER_LOOP - 1  # 255
CYCLIC_EXCLUDE_DISTANCE = 2
EPS = 1e-12
STS_DTYPE = np.float32


def read_3loops_xyz(file_path: Path) -> dict[int, np.ndarray]:
    """
    Wczytuje plik 3loops xyz w formacie:
    loop_id point_id x y z

    Zwraca:
    {
        loop_id: np.ndarray shape=(256, 3), dtype=float64
    }
    """
    loops_raw = defaultdict(list)

    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{file_path} | linia {line_no}: oczekiwano 5 kolumn, dostano {len(parts)}"
                )

            try:
                loop_id = int(parts[0])
                point_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
            except ValueError as exc:
                raise ValueError(
                    f"{file_path} | linia {line_no}: błąd parsowania: {exc}"
                ) from exc

            loops_raw[loop_id].append((point_id, x, y, z))

    loop_ids = sorted(loops_raw.keys())
    if loop_ids != EXPECTED_LOOPS:
        raise ValueError(
            f"{file_path} | niepoprawne loop_id: {loop_ids}, oczekiwano {EXPECTED_LOOPS}"
        )

    loops = {}
    expected_ids = list(range(EXPECTED_POINTS_PER_LOOP))

    for loop_id in EXPECTED_LOOPS:
        records = loops_raw[loop_id]

        if len(records) != EXPECTED_POINTS_PER_LOOP:
            raise ValueError(
                f"{file_path} | loop {loop_id}: liczba punktów = {len(records)}, "
                f"oczekiwano {EXPECTED_POINTS_PER_LOOP}"
            )

        point_ids = [r[0] for r in records]
        if point_ids != expected_ids:
            raise ValueError(
                f"{file_path} | loop {loop_id}: niepoprawne point_id"
            )

        loops[loop_id] = np.array(
            [[x, y, z] for _, x, y, z in records],
            dtype=np.float64,
        )

    return loops


def build_segments(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Buduje segmenty z kolejnych par punktów:
    segment i = (p_i, p_{i+1})

    Zwraca:
    - centers: shape (255, 3)
    - tangents_unit: shape (255, 3)
    """
    p0 = points[:-1]
    p1 = points[1:]

    tangents = p1 - p0
    lengths = np.linalg.norm(tangents, axis=1)

    if np.any(lengths <= EPS):
        bad = np.where(lengths <= EPS)[0]
        raise ValueError(f"Znaleziono segmenty o zerowej długości: {bad[:10]}")

    tangents_unit = tangents / lengths[:, None]
    centers = 0.5 * (p0 + p1)

    if centers.shape != (EXPECTED_SEGMENTS_PER_LOOP, 3):
        raise ValueError(f"Niepoprawny shape centers: {centers.shape}")

    if tangents_unit.shape != (EXPECTED_SEGMENTS_PER_LOOP, 3):
        raise ValueError(f"Niepoprawny shape tangents: {tangents_unit.shape}")

    return centers, tangents_unit


def build_cyclic_exclusion_mask(n_segments: int, exclude_distance: int) -> np.ndarray:
    """
    Tworzy maskę par segmentów do wycięcia przy odległości cyklicznej <= exclude_distance.

    d(i,j) = min(|i-j|, n-|i-j|)
    """
    idx = np.arange(n_segments)
    diff = np.abs(idx[:, None] - idx[None, :])
    cyc = np.minimum(diff, n_segments - diff)
    mask_exclude = cyc <= exclude_distance
    return mask_exclude


def compute_sts(centers: np.ndarray, tangents_unit: np.ndarray) -> np.ndarray:
    """
    Liczy macierz StS:
        StS(i,j) = ((t_i x t_j) · (r_i - r_j)) / |r_i - r_j|^3

    - wycinamy pary o odległości cyklicznej <= 2
    - w tych miejscach wpisujemy 0
    """
    n = centers.shape[0]
    if n != EXPECTED_SEGMENTS_PER_LOOP:
        raise ValueError(f"Oczekiwano {EXPECTED_SEGMENTS_PER_LOOP} segmentów, dostano {n}")

    r_diff = centers[:, None, :] - centers[None, :, :]         # (n, n, 3)
    cross_t = np.cross(
        tangents_unit[:, None, :],
        tangents_unit[None, :, :],
        axis=2,
    )                                                          # (n, n, 3)

    numer = np.einsum("ijk,ijk->ij", cross_t, r_diff)          # (n, n)
    r_norm = np.linalg.norm(r_diff, axis=2)                    # (n, n)
    denom = r_norm ** 3

    sts = np.zeros((n, n), dtype=np.float64)

    valid = denom > EPS
    sts[valid] = numer[valid] / denom[valid]

    mask_exclude = build_cyclic_exclusion_mask(n, CYCLIC_EXCLUDE_DISTANCE)
    sts[mask_exclude] = 0.0

    return sts.astype(STS_DTYPE, copy=False)


def compute_sta_from_sts(sts: np.ndarray) -> np.ndarray:
    """
    Liczy StA jako sumę po j:
        StA(i) = sum_j StS(i,j)

    Zwraca wektor shape = (255,), dtype=float32
    """
    sta = np.sum(sts, axis=1, dtype=np.float64)
    return sta.astype(STS_DTYPE, copy=False)


def write_sts_npy(out_path: Path, sts_tensor: np.ndarray) -> None:
    """
    Zapisuje tensor StS dla 3 pętli jako jeden plik .npy
    shape = (3, 255, 255)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, sts_tensor)


def write_sta_txt(out_path: Path, sta_tensor: np.ndarray) -> None:
    """
    Zapisuje StA dla 3 pętli jako jeden plik tekstowy:
    loop_id index value
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for loop_id in range(sta_tensor.shape[0]):
            for idx, value in enumerate(sta_tensor[loop_id]):
                f.write(f"{loop_id} {idx} {float(value):.12f}\n")


def main() -> None:
    STS_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    STA_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(INPUT_ROOT.glob("*/*.xyz"))

    processed = 0
    failed = 0
    class_counts = defaultdict(int)
    failures = []

    sts_global_min = np.inf
    sts_global_max = -np.inf
    sta_global_min = np.inf
    sta_global_max = -np.inf

    for in_file in all_files:
        rel_path = in_file.relative_to(INPUT_ROOT)
        out_sts_file = (STS_OUTPUT_ROOT / rel_path).with_suffix(".npy")
        out_sta_file = (STA_OUTPUT_ROOT / rel_path).with_suffix(".txt")

        try:
            loops = read_3loops_xyz(in_file)

            sts_list = []
            sta_list = []

            for loop_id in EXPECTED_LOOPS:
                points = loops[loop_id]
                centers, tangents_unit = build_segments(points)

                sts = compute_sts(centers, tangents_unit)
                sta = compute_sta_from_sts(sts)

                if sts.shape != (EXPECTED_SEGMENTS_PER_LOOP, EXPECTED_SEGMENTS_PER_LOOP):
                    raise ValueError(
                        f"{in_file} | loop {loop_id}: niepoprawny shape StS: {sts.shape}"
                    )

                if sta.shape != (EXPECTED_SEGMENTS_PER_LOOP,):
                    raise ValueError(
                        f"{in_file} | loop {loop_id}: niepoprawny shape StA: {sta.shape}"
                    )

                sts_list.append(sts)
                sta_list.append(sta)

            sts_tensor = np.stack(sts_list, axis=0).astype(STS_DTYPE, copy=False)  # (3,255,255)
            sta_tensor = np.stack(sta_list, axis=0).astype(STS_DTYPE, copy=False)  # (3,255)

            if sts_tensor.shape != (3, EXPECTED_SEGMENTS_PER_LOOP, EXPECTED_SEGMENTS_PER_LOOP):
                raise ValueError(f"{in_file} | niepoprawny shape końcowy StS: {sts_tensor.shape}")

            if sta_tensor.shape != (3, EXPECTED_SEGMENTS_PER_LOOP):
                raise ValueError(f"{in_file} | niepoprawny shape końcowy StA: {sta_tensor.shape}")

            write_sts_npy(out_sts_file, sts_tensor)
            write_sta_txt(out_sta_file, sta_tensor)

            processed += 1
            class_counts[in_file.parent.name] += 1

            sts_global_min = min(sts_global_min, float(np.min(sts_tensor)))
            sts_global_max = max(sts_global_max, float(np.max(sts_tensor)))
            sta_global_min = min(sta_global_min, float(np.min(sta_tensor)))
            sta_global_max = max(sta_global_max, float(np.max(sta_tensor)))

        except Exception as exc:
            failed += 1
            failures.append((in_file, str(exc)))

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("Raport liczenia 3loops -> StS -> StA\n")
        f.write("====================================\n\n")
        f.write(f"Wejście: {INPUT_ROOT}\n")
        f.write(f"Wyjście StS: {STS_OUTPUT_ROOT}\n")
        f.write(f"Wyjście StA: {STA_OUTPUT_ROOT}\n\n")

        f.write(f"Przetworzono poprawnie: {processed}\n")
        f.write(f"Błędy: {failed}\n\n")

        if processed > 0:
            f.write("Statystyki globalne:\n")
            f.write(f"  StS min = {sts_global_min:.12f}\n")
            f.write(f"  StS max = {sts_global_max:.12f}\n")
            f.write(f"  StA min = {sta_global_min:.12f}\n")
            f.write(f"  StA max = {sta_global_max:.12f}\n\n")

        f.write(f"Liczba pętli na próbkę: {len(EXPECTED_LOOPS)}\n")
        f.write(f"Liczba punktów na pętlę: {EXPECTED_POINTS_PER_LOOP}\n")
        f.write(f"Liczba segmentów na pętlę: {EXPECTED_SEGMENTS_PER_LOOP}\n")
        f.write(f"Wycięcie lokalnych par: odległość cykliczna <= {CYCLIC_EXCLUDE_DISTANCE}\n")
        f.write(f"Typ danych StS/StA: {STS_DTYPE.__name__}\n\n")

        f.write("Liczba zapisanych plików per klasa:\n")
        for class_name in sorted(class_counts):
            f.write(f"  {class_name}: {class_counts[class_name]}\n")

        if failures:
            f.write("\nBłędne pliki:\n")
            for file_path, err in failures:
                f.write(f"{file_path}\n")
                f.write(f"  {err}\n")

    print(f"Przetworzono poprawnie: {processed}")
    print(f"Błędy: {failed}")
    print(f"StS zapisano do: {STS_OUTPUT_ROOT}")
    print(f"StA zapisano do: {STA_OUTPUT_ROOT}")
    print(f"Raport zapisano do: {REPORT_PATH}")


if __name__ == "__main__":
    main()