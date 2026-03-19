from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(r"C:\Users\Adam\.vscode\grafy")
INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "normalized_graphs"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "1knot" / "xyz"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "build_1knot_xyz_report.txt"

EXPECTED_EDGES = [0, 1, 2]
EXPECTED_POINTS_PER_EDGE = 101


def read_xyz_file(file_path: Path) -> dict[int, list[tuple[int, float, float, float]]]:
    """
    Wczytuje plik .xyz w formacie:
    edge_id point_id x y z

    Zwraca:
    {
        edge_id: [(point_id, x, y, z), ...]
    }
    """
    edges = defaultdict(list)

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
                edge_id = int(parts[0])
                point_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
            except ValueError as exc:
                raise ValueError(
                    f"{file_path} | linia {line_no}: błąd parsowania: {exc}"
                ) from exc

            edges[edge_id].append((point_id, x, y, z))

    return dict(edges)


def validate_graph_structure(
    edges: dict[int, list[tuple[int, float, float, float]]],
    file_path: Path,
) -> None:
    """
    Sprawdza podstawową strukturę grafu:
    - edge_id = 0,1,2
    - 101 punktów na krawędź
    - point_id = 0..100
    """
    edge_ids = sorted(edges.keys())
    if edge_ids != EXPECTED_EDGES:
        raise ValueError(f"{file_path} | niepoprawne edge_id: {edge_ids}")

    expected_point_ids = list(range(EXPECTED_POINTS_PER_EDGE))

    for edge_id in EXPECTED_EDGES:
        records = edges[edge_id]

        if len(records) != EXPECTED_POINTS_PER_EDGE:
            raise ValueError(
                f"{file_path} | edge {edge_id}: liczba punktów = {len(records)}, "
                f"oczekiwano {EXPECTED_POINTS_PER_EDGE}"
            )

        point_ids = [rec[0] for rec in records]
        if point_ids != expected_point_ids:
            raise ValueError(
                f"{file_path} | edge {edge_id}: niepoprawne point_id"
            )


def extract_edge_points(
    edge_records: list[tuple[int, float, float, float]]
) -> list[tuple[float, float, float]]:
    """
    Zamienia:
    [(point_id, x, y, z), ...]
    na:
    [(x, y, z), ...]
    """
    return [(x, y, z) for _, x, y, z in edge_records]


def points_equal(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    tol: float = 1e-9,
) -> bool:
    return (
        abs(p1[0] - p2[0]) <= tol and
        abs(p1[1] - p2[1]) <= tol and
        abs(p1[2] - p2[2]) <= tol
    )


def build_1knot_curve(
    edges: dict[int, list[tuple[int, float, float, float]]],
    file_path: Path,
) -> list[tuple[float, float, float]]:
    """
    Buduje jedną zamkniętą krzywą z grafu theta według ustalonej konwencji:

    - edge 0: bierzemy punkty od 1 do 100
      (czyli bez pierwszego wspólnego wierzchołka, z dolnym wierzchołkiem na końcu)

    - edge 1 reversed: bierzemy punkty od 99 do 0
      (czyli bez dolnego wierzchołka na początku, z górnym wierzchołkiem na końcu)

    - edge 2: bierzemy punkty od 1 do 99
      (czyli bez górnego i bez dolnego wspólnego wierzchołka)

    Dzięki temu dostajemy jedną zamkniętą sekwencję punktów bez dublowania wspólnych wierzchołków.
    """
    edge0 = extract_edge_points(edges[0])
    edge1 = extract_edge_points(edges[1])
    edge2 = extract_edge_points(edges[2])

    # Kontrola wspólnych wierzchołków
    top0, bottom0 = edge0[0], edge0[-1]
    top1, bottom1 = edge1[0], edge1[-1]
    top2, bottom2 = edge2[0], edge2[-1]

    if not (points_equal(top0, top1) and points_equal(top0, top2)):
        raise ValueError(f"{file_path} | krawędzie nie mają wspólnego górnego wierzchołka")

    if not (points_equal(bottom0, bottom1) and points_equal(bottom0, bottom2)):
        raise ValueError(f"{file_path} | krawędzie nie mają wspólnego dolnego wierzchołka")

    # Budowanie 1knot zgodnie z ustaloną konwencją
    part1 = edge0[1:]                  # p1..p100  -> 100 punktów
    part2 = list(reversed(edge1))[1:]  # reversed: bottom..top, bez pierwszego(bottom) -> 100 punktów
    part3 = edge2[1:-1]                # bez top i bez bottom -> 99 punktów

    curve = part1 + part2 + part3

    expected_length = 100 + 100 + 99
    if len(curve) != expected_length:
        raise ValueError(
            f"{file_path} | zła długość 1knot: {len(curve)}, oczekiwano {expected_length}"
        )

    return curve


def write_1knot_xyz(
    out_path: Path,
    curve: list[tuple[float, float, float]],
) -> None:
    """
    Zapisuje 1knot xyz w formacie:
    index x y z
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, (x, y, z) in enumerate(curve):
            f.write(f"{idx} {x:.12f} {y:.12f} {z:.12f}\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(INPUT_ROOT.glob("*/*.xyz"))

    processed = 0
    failed = 0
    class_counts = defaultdict(int)
    failures = []

    output_lengths = []

    for in_file in all_files:
        rel_path = in_file.relative_to(INPUT_ROOT)
        out_file = OUTPUT_ROOT / rel_path

        try:
            edges = read_xyz_file(in_file)
            validate_graph_structure(edges, in_file)

            curve = build_1knot_curve(edges, in_file)
            output_lengths.append(len(curve))

            write_1knot_xyz(out_file, curve)

            processed += 1
            class_counts[in_file.parent.name] += 1

        except Exception as exc:
            failed += 1
            failures.append((in_file, str(exc)))

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("Raport budowy 1knot xyz\n")
        f.write("=======================\n\n")
        f.write(f"Wejście: {INPUT_ROOT}\n")
        f.write(f"Wyjście: {OUTPUT_ROOT}\n\n")
        f.write(f"Przetworzono poprawnie: {processed}\n")
        f.write(f"Błędy: {failed}\n\n")

        if output_lengths:
            f.write("Długość wyjściowej krzywej 1knot:\n")
            f.write(f"  min = {min(output_lengths)}\n")
            f.write(f"  max = {max(output_lengths)}\n\n")

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