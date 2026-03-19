from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(r"C:\Users\Adam\.vscode\grafy")
INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "3loops" / "xyz_resampled_256"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "3loops" / "xyz_resampled_256_canonical"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "canonicalize_3loops_xyz_256_report.txt"

EXPECTED_LOOPS = [0, 1, 2]
EXPECTED_POINTS_PER_LOOP = 256
EXPECTED_UNIQUE_POINTS_PER_LOOP = 255
ROUND_DECIMALS = 12
TOL = 1e-9


def points_equal(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    tol: float = TOL,
) -> bool:
    return (
        abs(p1[0] - p2[0]) <= tol and
        abs(p1[1] - p2[1]) <= tol and
        abs(p1[2] - p2[2]) <= tol
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

        points = [(x, y, z) for _, x, y, z in records]

        # Pętla ma być jawnie zamknięta: pierwszy punkt = ostatni punkt
        if not points_equal(points[0], points[-1]):
            raise ValueError(
                f"{file_path} | loop {loop_id}: pierwszy i ostatni punkt nie zamykają pętli"
            )

        loops[loop_id] = points

    return loops


def rounded_point(
    p: tuple[float, float, float],
    decimals: int = ROUND_DECIMALS,
) -> tuple[float, float, float]:
    return (
        round(p[0], decimals),
        round(p[1], decimals),
        round(p[2], decimals),
    )


def lexicographic_key_for_curve(points: list[tuple[float, float, float]]) -> tuple:
    return tuple(rounded_point(p) for p in points)


def rotate_curve(
    points: list[tuple[float, float, float]],
    start_idx: int,
) -> list[tuple[float, float, float]]:
    return points[start_idx:] + points[:start_idx]


def find_min_point_indices(points: list[tuple[float, float, float]]) -> list[int]:
    rounded_points = [rounded_point(p) for p in points]
    min_point = min(rounded_points)
    return [i for i, p in enumerate(rounded_points) if p == min_point]


def canonicalize_closed_loop_unique(
    closed_points: list[tuple[float, float, float]],
) -> tuple[list[tuple[float, float, float]], int, str]:
    """
    Kanonizacja pętli zapisanej jako 256 punktów z duplikatem końca:
    [p0, p1, ..., p254, p0]

    Na czas kanonizacji pracujemy na 255 unikalnych punktach:
    [p0, p1, ..., p254]

    Potem dopinamy z powrotem pierwszy punkt na koniec.
    """
    if len(closed_points) != EXPECTED_POINTS_PER_LOOP:
        raise ValueError(
            f"Pętla ma {len(closed_points)} punktów, oczekiwano {EXPECTED_POINTS_PER_LOOP}"
        )

    if not points_equal(closed_points[0], closed_points[-1]):
        raise ValueError("Pętla nie jest jawnie zamknięta")

    unique_points = closed_points[:-1]

    if len(unique_points) != EXPECTED_UNIQUE_POINTS_PER_LOOP:
        raise ValueError(
            f"Niepoprawna liczba unikalnych punktów: {len(unique_points)}"
        )

    candidate_indices = find_min_point_indices(unique_points)

    best_curve = None
    best_start_idx = None
    best_direction = None
    best_key = None

    reversed_points = list(reversed(unique_points))

    for start_idx in candidate_indices:
        # forward
        forward_curve = rotate_curve(unique_points, start_idx)
        forward_key = lexicographic_key_for_curve(forward_curve)

        if best_key is None or forward_key < best_key:
            best_curve = forward_curve
            best_start_idx = start_idx
            best_direction = "forward"
            best_key = forward_key

        # reverse
        reverse_start_idx = len(unique_points) - 1 - start_idx
        reverse_curve = rotate_curve(reversed_points, reverse_start_idx)
        reverse_key = lexicographic_key_for_curve(reverse_curve)

        if reverse_key < best_key:
            best_curve = reverse_curve
            best_start_idx = start_idx
            best_direction = "reverse"
            best_key = reverse_key

    # Dopinamy punkt zamykający
    best_closed_curve = best_curve + [best_curve[0]]

    if len(best_closed_curve) != EXPECTED_POINTS_PER_LOOP:
        raise ValueError(
            f"Po kanonizacji pętla ma {len(best_closed_curve)} punktów, "
            f"oczekiwano {EXPECTED_POINTS_PER_LOOP}"
        )

    return best_closed_curve, best_start_idx, best_direction


def write_3loops_xyz(
    out_path: Path,
    loops: list[list[tuple[float, float, float]]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for loop_id, points in enumerate(loops):
            for point_id, (x, y, z) in enumerate(points):
                f.write(f"{loop_id} {point_id} {x:.12f} {y:.12f} {z:.12f}\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(INPUT_ROOT.glob("*/*.xyz"))

    processed = 0
    failed = 0
    class_counts = defaultdict(int)
    failures = []

    forward_count = 0
    reverse_count = 0
    start_index_counts = defaultdict(int)
    permutation_counts = defaultdict(int)

    for in_file in all_files:
        rel_path = in_file.relative_to(INPUT_ROOT)
        out_file = OUTPUT_ROOT / rel_path

        try:
            loops = read_3loops_xyz(in_file)

            canonical_loops_with_meta = []

            for original_loop_id in EXPECTED_LOOPS:
                canonical_points, start_idx, direction = canonicalize_closed_loop_unique(
                    loops[original_loop_id]
                )

                canonical_loops_with_meta.append(
                    (original_loop_id, canonical_points)
                )

                start_index_counts[start_idx] += 1
                if direction == "forward":
                    forward_count += 1
                elif direction == "reverse":
                    reverse_count += 1
                else:
                    raise ValueError(f"Nieznany kierunek: {direction}")

            # sortowanie pętli do kolejności kanonicznej
            canonical_loops_with_meta.sort(
                key=lambda item: lexicographic_key_for_curve(item[1][:-1])
            )

            permutation = tuple(item[0] for item in canonical_loops_with_meta)
            permutation_counts[permutation] += 1

            canonical_loops = [item[1] for item in canonical_loops_with_meta]

            write_3loops_xyz(out_file, canonical_loops)

            processed += 1
            class_counts[in_file.parent.name] += 1

        except Exception as exc:
            failed += 1
            failures.append((in_file, str(exc)))

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("Raport kanonizacji 3loops xyz_resampled_256\n")
        f.write("===========================================\n\n")
        f.write(f"Wejście: {INPUT_ROOT}\n")
        f.write(f"Wyjście: {OUTPUT_ROOT}\n\n")
        f.write(f"Przetworzono poprawnie: {processed}\n")
        f.write(f"Błędy: {failed}\n\n")

        f.write("Wybrany kierunek dla pętli:\n")
        f.write(f"  forward: {forward_count}\n")
        f.write(f"  reverse: {reverse_count}\n\n")

        f.write("Najczęściej wybierane indeksy startowe:\n")
        for idx, count in sorted(start_index_counts.items(), key=lambda x: (-x[1], x[0]))[:20]:
            f.write(f"  {idx}: {count}\n")

        f.write("\nNajczęstsze permutacje kolejności pętli po kanonizacji:\n")
        for perm, count in sorted(permutation_counts.items(), key=lambda x: (-x[1], x[0]))[:20]:
            f.write(f"  {perm}: {count}\n")

        f.write("\nLiczba zapisanych plików per klasa:\n")
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