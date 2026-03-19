from pathlib import Path
from collections import defaultdict
import math


PROJECT_ROOT = Path(r"C:\Users\Adam\.vscode\grafy")
INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "3loops" / "xyz"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "3loops" / "xyz_resampled_256"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "resample_3loops_xyz_256_report.txt"

INPUT_LOOPS = [0, 1, 2]
INPUT_POINTS_PER_LOOP = 201
OUTPUT_POINTS_PER_LOOP = 256
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


def read_3loops_xyz(
    file_path: Path,
) -> dict[int, list[tuple[float, float, float]]]:
    """
    Wczytuje plik 3loops xyz w formacie:
    loop_id point_id x y z

    Zwraca:
    {
        loop_id: [(x, y, z), ...]
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
    if loop_ids != INPUT_LOOPS:
        raise ValueError(
            f"{file_path} | niepoprawne loop_id: {loop_ids}, oczekiwano {INPUT_LOOPS}"
        )

    loops = {}
    expected_ids = list(range(INPUT_POINTS_PER_LOOP))

    for loop_id in INPUT_LOOPS:
        records = loops_raw[loop_id]

        if len(records) != INPUT_POINTS_PER_LOOP:
            raise ValueError(
                f"{file_path} | loop {loop_id}: liczba punktów = {len(records)}, "
                f"oczekiwano {INPUT_POINTS_PER_LOOP}"
            )

        point_ids = [r[0] for r in records]
        if point_ids != expected_ids:
            raise ValueError(
                f"{file_path} | loop {loop_id}: niepoprawne point_id"
            )

        loops[loop_id] = [(x, y, z) for _, x, y, z in records]

    return loops


def cumulative_arc_lengths(points: list[tuple[float, float, float]]) -> list[float]:
    """
    Zwraca długości skumulowane dla łamanej:
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
    Interpolacja liniowa między p1 i p2.
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


def write_3loops_xyz(
    out_path: Path,
    loops: dict[int, list[tuple[float, float, float]]],
) -> None:
    """
    Zapisuje 3loops xyz w formacie:
    loop_id point_id x y z
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for loop_id in INPUT_LOOPS:
            for point_id, (x, y, z) in enumerate(loops[loop_id]):
                f.write(f"{loop_id} {point_id} {x:.12f} {y:.12f} {z:.12f}\n")


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
            loops = read_3loops_xyz(in_file)

            resampled_loops = {}
            for loop_id in INPUT_LOOPS:
                input_len = polyline_length(loops[loop_id])
                output_loop = resample_polyline(loops[loop_id], OUTPUT_POINTS_PER_LOOP)
                output_len = polyline_length(output_loop)

                input_lengths.append(input_len)
                output_lengths.append(output_len)
                resampled_loops[loop_id] = output_loop

            write_3loops_xyz(out_file, resampled_loops)

            processed += 1
            class_counts[in_file.parent.name] += 1

        except Exception as exc:
            failed += 1
            failures.append((in_file, str(exc)))

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("Raport resamplingu 3loops xyz do 256 punktów na pętlę\n")
        f.write("=====================================================\n\n")
        f.write(f"Wejście: {INPUT_ROOT}\n")
        f.write(f"Wyjście: {OUTPUT_ROOT}\n\n")
        f.write(f"Przetworzono poprawnie: {processed}\n")
        f.write(f"Błędy: {failed}\n\n")

        if input_lengths:
            f.write("Długość łuku pętli przed resamplingiem:\n")
            f.write(f"  min = {min(input_lengths):.12f}\n")
            f.write(f"  max = {max(input_lengths):.12f}\n\n")

        if output_lengths:
            f.write("Długość łuku pętli po resamplingu:\n")
            f.write(f"  min = {min(output_lengths):.12f}\n")
            f.write(f"  max = {max(output_lengths):.12f}\n\n")

        f.write(f"Liczba punktów wejściowych na pętlę: {INPUT_POINTS_PER_LOOP}\n")
        f.write(f"Liczba punktów wyjściowych na pętlę: {OUTPUT_POINTS_PER_LOOP}\n\n")

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