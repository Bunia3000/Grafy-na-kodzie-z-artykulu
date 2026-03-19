from pathlib import Path
from collections import defaultdict
import numpy as np


PROJECT_ROOT = Path(r"C:\Users\Adam\.vscode\grafy")
INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "1knot" / "xyz_resampled_256_canonical"
STS_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "1knot" / "sts"
STA_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "1knot" / "sta"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "compute_1knot_sts_sta_report.txt"

EXPECTED_POINTS = 256
EXPECTED_SEGMENTS = EXPECTED_POINTS - 1  # zgodnie z naszym pipeline'em: 255
CYCLIC_EXCLUDE_DISTANCE = 2
EPS = 1e-12
STS_DTYPE = np.float32


def read_1knot_xyz(file_path: Path) -> np.ndarray:
    """
    Wczytuje plik 1knot xyz w formacie:
    index x y z

    Zwraca tablicę shape = (256, 3), dtype=float64
    """
    rows = []

    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) != 4:
                raise ValueError(
                    f"{file_path} | linia {line_no}: oczekiwano 4 kolumn, dostano {len(parts)}"
                )

            try:
                point_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
            except ValueError as exc:
                raise ValueError(
                    f"{file_path} | linia {line_no}: błąd parsowania: {exc}"
                ) from exc

            rows.append((point_id, x, y, z))

    if len(rows) != EXPECTED_POINTS:
        raise ValueError(
            f"{file_path} | liczba punktów = {len(rows)}, oczekiwano {EXPECTED_POINTS}"
        )

    point_ids = [row[0] for row in rows]
    expected_ids = list(range(EXPECTED_POINTS))
    if point_ids != expected_ids:
        raise ValueError(
            f"{file_path} | niepoprawne index: oczekiwano 0..{EXPECTED_POINTS - 1}"
        )

    points = np.array([[x, y, z] for _, x, y, z in rows], dtype=np.float64)
    return points


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

    if centers.shape != (EXPECTED_SEGMENTS, 3):
        raise ValueError(f"Niepoprawny shape centers: {centers.shape}")

    if tangents_unit.shape != (EXPECTED_SEGMENTS, 3):
        raise ValueError(f"Niepoprawny shape tangents: {tangents_unit.shape}")

    return centers, tangents_unit


def build_cyclic_exclusion_mask(n_segments: int, exclude_distance: int) -> np.ndarray:
    """
    Tworzy maskę par segmentów do wycięcia przy odległości cyklicznej <= exclude_distance.

    d(i,j) = min(|i-j|, n-|i-j|)

    Zwraca:
    mask_exclude shape = (n_segments, n_segments), dtype=bool
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

    Zgodnie z ustaleniami:
    - wycinamy pary o odległości cyklicznej <= 2
    - w tych miejscach wpisujemy 0
    - pełna macierz jest zapisywana
    """
    n = centers.shape[0]
    if n != EXPECTED_SEGMENTS:
        raise ValueError(f"Oczekiwano {EXPECTED_SEGMENTS} segmentów, dostano {n}")

    # różnice położeń segmentów: r_i - r_j
    r_diff = centers[:, None, :] - centers[None, :, :]         # (n, n, 3)

    # iloczyny wektorowe tangentów: t_i x t_j
    cross_t = np.cross(
        tangents_unit[:, None, :],
        tangents_unit[None, :, :],
        axis=2,
    )                                                          # (n, n, 3)

    # licznik
    numer = np.einsum("ijk,ijk->ij", cross_t, r_diff)          # (n, n)

    # mianownik
    r_norm = np.linalg.norm(r_diff, axis=2)                    # (n, n)
    denom = r_norm ** 3

    sts = np.zeros((n, n), dtype=np.float64)

    # liczymy tylko tam, gdzie mianownik nie jest zerowy
    valid = denom > EPS
    sts[valid] = numer[valid] / denom[valid]

    # wycięcie lokalnych par segmentów po odległości cyklicznej
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


def write_sts_npy(out_path: Path, sts: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, sts)


def write_sta_txt(out_path: Path, sta: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, value in enumerate(sta):
            f.write(f"{idx} {float(value):.12f}\n")


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
            points = read_1knot_xyz(in_file)
            centers, tangents_unit = build_segments(points)

            sts = compute_sts(centers, tangents_unit)
            sta = compute_sta_from_sts(sts)

            if sts.shape != (EXPECTED_SEGMENTS, EXPECTED_SEGMENTS):
                raise ValueError(
                    f"{in_file} | niepoprawny shape StS: {sts.shape}"
                )

            if sta.shape != (EXPECTED_SEGMENTS,):
                raise ValueError(
                    f"{in_file} | niepoprawny shape StA: {sta.shape}"
                )

            write_sts_npy(out_sts_file, sts)
            write_sta_txt(out_sta_file, sta)

            processed += 1
            class_counts[in_file.parent.name] += 1

            sts_global_min = min(sts_global_min, float(np.min(sts)))
            sts_global_max = max(sts_global_max, float(np.max(sts)))
            sta_global_min = min(sta_global_min, float(np.min(sta)))
            sta_global_max = max(sta_global_max, float(np.max(sta)))

        except Exception as exc:
            failed += 1
            failures.append((in_file, str(exc)))

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("Raport liczenia 1knot -> StS -> StA\n")
        f.write("===================================\n\n")
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

        f.write(f"Liczba punktów wejściowych: {EXPECTED_POINTS}\n")
        f.write(f"Liczba segmentów: {EXPECTED_SEGMENTS}\n")
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