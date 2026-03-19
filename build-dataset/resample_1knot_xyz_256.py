from pathlib import Path
from collections import defaultdict
import math


PROJECT_ROOT = Path(r"C:\Users\Adam\.vscode\grafy")
INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "1knot" / "xyz"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "1knot" / "xyz_resampled_256"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "resample_1knot_xyz_256_report.txt"

INPUT_POINTS = 299
OUTPUT_POINTS = 256
EPS = 1e-12


def euclidean_distance(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
) -> float:
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2 +
        (p1[2] - p2[2]) ** 2
    )


def read_1knot_xyz(file_path: Path) -> list[tuple[float, float, float]]:
    """
    Wczytuje plik 1knot xyz w formacie:
    index x y z

    Zwraca listę punktów:
    [(x, y, z), ...]
    """
    points = []

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

            points.append((point_id, x, y, z))

    if len(points) != INPUT_POINTS:
        raise ValueError(
            f"{file_path} | liczba punktów = {len(points)}, oczekiwano {INPUT_POINTS}"
        )

    point_ids = [p[0] for p in points]
    expected_ids = list(range(INPUT_POINTS))
    if point_ids != expected_ids:
        raise ValueError(f"{file_path} | niepoprawne index: oczekiwano 0..{INPUT_POINTS - 1}")

    return [(x, y, z) for _, x, y, z in points]


def cumulative_arc_lengths(points: list[tuple[float, float, float]]) -> list[float]:
    """
    Zwraca długości skumulowane:
    s[0] = 0
    s[i] = długość łuku od punktu 0 do punktu i
    """
    s = [0.0]
    for i in range(1, len(points)):
        d = euclidean_distance(points[i - 1], points[i])
        s.append(s[-1] + d)
    return s


def interpolate_point(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    """
    Interpolacja liniowa pomiędzy p1 i p2.
    t w zakresie [0, 1].
    """
    return (
        p1[0] + t * (p2[0] - p1[0]),
        p1[1] + t * (p2[1] - p1[1]),
        p1[2] + t * (p2[2] - p1[2]),
    )


def resample_polyline(
    points: list[tuple[float, float, float]],
    n_out: int,
) -> list[tuple[float, float, float]]:
    """
    Resampling łamanej do n_out punktów równomiernie po długości łuku.
    """
    if len(points) < 2:
        raise ValueError("Za mało punktów do resamplingu")

    s = cumulative_arc_lengths(points)
    total_length = s[-1]

    if total_length <= EPS:
        raise ValueError("Cała krzywa ma zerową długość")

    targets = [
        total_length * i / (n_out - 1)
        for i in range(n_out)
    ]

    resampled = []
    seg_idx = 0

    for target in targets:
        while seg_idx < len(s) - 2 and s[seg_idx + 1] < target:
            seg_idx += 1

        s_left = s[seg_idx]
        s_right = s[seg_idx + 1]
        p_left = points[seg_idx]
        p_right = points[seg_idx + 1]

        if abs(s_right - s_left) <= EPS:
            # segment znikomo mały; bierzemy lewy punkt
            resampled.append(p_left)
            continue

        t = (target - s_left) / (s_right - s_left)
        t = max(0.0, min(1.0, t))
        resampled.append(interpolate_point(p_left, p_right, t))

    if len(resampled) != n_out:
        raise ValueError(
            f"Błąd resamplingu: dostano {len(resampled)} punktów, oczekiwano {n_out}"
        )

    return resampled


def polyline_length(points: list[tuple[float, float, float]]) -> float:
    return sum(
        euclidean_distance(points[i], points[i + 1])
        for i in range(len(points) - 1)
    )


def write_1knot_xyz(
    out_path: Path,
    points: list[tuple[float, float, float]],
) -> None:
    """
    Zapisuje krzywą w formacie:
    index x y z
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, (x, y, z) in enumerate(points):
            f.write(f"{idx} {x:.12f} {y:.12f} {z:.12f}\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(INPUT_ROOT.glob("*/*.xyz"))

    processed = 0
    failed = 0
    class_counts = defaultdict(int)
    failures = []

    input_lengths = []
    output_lengths = []

    for in_file in all_files:
        rel_path = in_file.relative_to(INPUT_ROOT)
        out_file = OUTPUT_ROOT / rel_path

        try:
            points = read_1knot_xyz(in_file)

            input_len = polyline_length(points)
            resampled = resample_polyline(points, OUTPUT_POINTS)
            output_len = polyline_length(resampled)

            input_lengths.append(input_len)
            output_lengths.append(output_len)

            write_1knot_xyz(out_file, resampled)

            processed += 1
            class_counts[in_file.parent.name] += 1

        except Exception as exc:
            failed += 1
            failures.append((in_file, str(exc)))

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("Raport resamplingu 1knot xyz do 256 punktów\n")
        f.write("===========================================\n\n")
        f.write(f"Wejście: {INPUT_ROOT}\n")
        f.write(f"Wyjście: {OUTPUT_ROOT}\n\n")
        f.write(f"Przetworzono poprawnie: {processed}\n")
        f.write(f"Błędy: {failed}\n\n")

        if input_lengths:
            f.write("Długość łuku przed resamplingiem:\n")
            f.write(f"  min = {min(input_lengths):.12f}\n")
            f.write(f"  max = {max(input_lengths):.12f}\n\n")

        if output_lengths:
            f.write("Długość łuku po resamplingu:\n")
            f.write(f"  min = {min(output_lengths):.12f}\n")
            f.write(f"  max = {max(output_lengths):.12f}\n\n")

        f.write(f"Liczba punktów wejściowych: {INPUT_POINTS}\n")
        f.write(f"Liczba punktów wyjściowych: {OUTPUT_POINTS}\n\n")

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
    print(f"Dane zapisano do: {OUTPUT_ROOT}")
    print(f"Raport zapisano do: {REPORT_PATH}")


if __name__ == "__main__":
    main()