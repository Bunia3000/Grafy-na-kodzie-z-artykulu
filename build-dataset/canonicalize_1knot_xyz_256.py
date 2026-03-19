from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(r"C:\Users\Adam\.vscode\grafy")
INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "1knot" / "xyz_resampled_256"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "1knot" / "xyz_resampled_256_canonical"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "canonicalize_1knot_xyz_256_report.txt"

EXPECTED_POINTS = 256
ROUND_DECIMALS = 12


def read_1knot_xyz(file_path: Path) -> list[tuple[float, float, float]]:
    """
    Wczytuje plik 1knot xyz w formacie:
    index x y z

    Zwraca listę punktów:
    [(x, y, z), ...]
    """
    points_raw = []

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

            points_raw.append((point_id, x, y, z))

    if len(points_raw) != EXPECTED_POINTS:
        raise ValueError(
            f"{file_path} | liczba punktów = {len(points_raw)}, oczekiwano {EXPECTED_POINTS}"
        )

    point_ids = [p[0] for p in points_raw]
    expected_ids = list(range(EXPECTED_POINTS))
    if point_ids != expected_ids:
        raise ValueError(
            f"{file_path} | niepoprawne index: oczekiwano 0..{EXPECTED_POINTS - 1}"
        )

    return [(x, y, z) for _, x, y, z in points_raw]


def rounded_point(p: tuple[float, float, float], decimals: int = ROUND_DECIMALS) -> tuple[float, float, float]:
    return (
        round(p[0], decimals),
        round(p[1], decimals),
        round(p[2], decimals),
    )


def lexicographic_key_for_curve(points: list[tuple[float, float, float]]) -> tuple:
    """
    Zamienia całą krzywą na krotkę leksykograficzną po zaokrągleniu punktów.
    """
    return tuple(rounded_point(p) for p in points)


def rotate_curve(points: list[tuple[float, float, float]], start_idx: int) -> list[tuple[float, float, float]]:
    """
    Cyklicznie przesuwa krzywą tak, by punkt start_idx był pierwszy.
    """
    return points[start_idx:] + points[:start_idx]


def find_min_point_indices(points: list[tuple[float, float, float]]) -> list[int]:
    """
    Zwraca wszystkie indeksy punktów minimalnych leksykograficznie
    po zaokrągleniu.
    """
    rounded_points = [rounded_point(p) for p in points]
    min_point = min(rounded_points)
    return [i for i, p in enumerate(rounded_points) if p == min_point]


def canonicalize_curve(points: list[tuple[float, float, float]]) -> tuple[list[tuple[float, float, float]], int, str]:
    """
    Kanonizacja zamkniętej krzywej:
    1. wybór punktu startowego = leksykograficznie najmniejszy punkt
    2. wybór kierunku = leksykograficznie mniejsza z dwóch orientacji

    Zwraca:
    - punkty po kanonizacji
    - wybrany indeks startowy (w oryginalnej krzywej)
    - wybrany kierunek: 'forward' albo 'reverse'
    """
    if len(points) != EXPECTED_POINTS:
        raise ValueError(f"Krzywa ma {len(points)} punktów, oczekiwano {EXPECTED_POINTS}")

    candidate_indices = find_min_point_indices(points)

    best_curve = None
    best_start_idx = None
    best_direction = None
    best_key = None

    reversed_points = list(reversed(points))

    for start_idx in candidate_indices:
        # wariant forward
        forward_curve = rotate_curve(points, start_idx)
        forward_key = lexicographic_key_for_curve(forward_curve)

        if best_key is None or forward_key < best_key:
            best_curve = forward_curve
            best_start_idx = start_idx
            best_direction = "forward"
            best_key = forward_key

        # wariant reverse:
        # punkt o indeksie start_idx w oryginalnej krzywej
        # ma odpowiadający indeks w odwróconej: n-1-start_idx
        reverse_start_idx = len(points) - 1 - start_idx
        reverse_curve = rotate_curve(reversed_points, reverse_start_idx)
        reverse_key = lexicographic_key_for_curve(reverse_curve)

        if reverse_key < best_key:
            best_curve = reverse_curve
            best_start_idx = start_idx
            best_direction = "reverse"
            best_key = reverse_key

    return best_curve, best_start_idx, best_direction


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

    forward_count = 0
    reverse_count = 0
    start_index_counts = defaultdict(int)

    for in_file in all_files:
        rel_path = in_file.relative_to(INPUT_ROOT)
        out_file = OUTPUT_ROOT / rel_path

        try:
            points = read_1knot_xyz(in_file)
            canonical_points, start_idx, direction = canonicalize_curve(points)

            write_1knot_xyz(out_file, canonical_points)

            processed += 1
            class_counts[in_file.parent.name] += 1
            start_index_counts[start_idx] += 1

            if direction == "forward":
                forward_count += 1
            elif direction == "reverse":
                reverse_count += 1
            else:
                raise ValueError(f"Nieznany kierunek: {direction}")

        except Exception as exc:
            failed += 1
            failures.append((in_file, str(exc)))

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("Raport kanonizacji 1knot xyz_resampled_256\n")
        f.write("==========================================\n\n")
        f.write(f"Wejście: {INPUT_ROOT}\n")
        f.write(f"Wyjście: {OUTPUT_ROOT}\n\n")
        f.write(f"Przetworzono poprawnie: {processed}\n")
        f.write(f"Błędy: {failed}\n\n")

        f.write("Wybrany kierunek:\n")
        f.write(f"  forward: {forward_count}\n")
        f.write(f"  reverse: {reverse_count}\n\n")

        f.write("Liczba zapisanych plików per klasa:\n")
        for class_name in sorted(class_counts):
            f.write(f"  {class_name}: {class_counts[class_name]}\n")

        f.write("\nNajczęściej wybierane indeksy startowe:\n")
        for idx, count in sorted(start_index_counts.items(), key=lambda x: (-x[1], x[0]))[:20]:
            f.write(f"  {idx}: {count}\n")

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