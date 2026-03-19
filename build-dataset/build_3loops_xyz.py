from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(r"C:\Users\Adam\.vscode\grafy")
INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "normalized_graphs"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "3loops" / "xyz"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "build_3loops_xyz_report.txt"

EXPECTED_EDGES = [0, 1, 2]
EXPECTED_POINTS_PER_EDGE = 101
EXPECTED_LOOPS = [0, 1, 2]


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


def build_loop(
    edge_a: list[tuple[float, float, float]],
    edge_b: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """
    Buduje pętlę z:
    edge_a + reverse(edge_b), bez dublowania wspólnego dolnego wierzchołka.

    Jeśli edge_a ma punkty:
      top ... bottom
    a reversed(edge_b) ma:
      bottom ... top

    to wynik to:
      edge_a + reversed(edge_b)[1:]

    czyli:
      101 + 100 = 201 punktów
    """
    part1 = edge_a
    part2 = list(reversed(edge_b))[1:]
    loop = part1 + part2

    if len(loop) != 201:
        raise ValueError(f"Zła długość pętli: {len(loop)}, oczekiwano 201")

    return loop


def build_3loops(
    edges: dict[int, list[tuple[int, float, float, float]]],
    file_path: Path,
) -> dict[int, list[tuple[float, float, float]]]:
    """
    Buduje trzy pętle:
    - loop 0 = edge0 + reverse(edge1)
    - loop 1 = edge0 + reverse(edge2)
    - loop 2 = edge1 + reverse(edge2)

    Wszystkie bez dublowania wspólnego dolnego wierzchołka.
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

    loops = {
        0: build_loop(edge0, edge1),  # loop12
        1: build_loop(edge0, edge2),  # loop13
        2: build_loop(edge1, edge2),  # loop23
    }

    return loops


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
        for loop_id in EXPECTED_LOOPS:
            loop_points = loops[loop_id]
            for point_id, (x, y, z) in enumerate(loop_points):
                f.write(f"{loop_id} {point_id} {x:.12f} {y:.12f} {z:.12f}\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(INPUT_ROOT.glob("*/*.xyz"))

    processed = 0
    failed = 0
    class_counts = defaultdict(int)
    failures = []

    loop_lengths = []

    for in_file in all_files:
        rel_path = in_file.relative_to(INPUT_ROOT)
        out_file = OUTPUT_ROOT / rel_path

        try:
            edges = read_xyz_file(in_file)
            validate_graph_structure(edges, in_file)

            loops = build_3loops(edges, in_file)

            for loop_id in EXPECTED_LOOPS:
                loop_lengths.append(len(loops[loop_id]))

            write_3loops_xyz(out_file, loops)

            processed += 1
            class_counts[in_file.parent.name] += 1

        except Exception as exc:
            failed += 1
            failures.append((in_file, str(exc)))

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("Raport budowy 3loops xyz\n")
        f.write("========================\n\n")
        f.write(f"Wejście: {INPUT_ROOT}\n")
        f.write(f"Wyjście: {OUTPUT_ROOT}\n\n")
        f.write(f"Przetworzono poprawnie: {processed}\n")
        f.write(f"Błędy: {failed}\n\n")

        if loop_lengths:
            f.write("Długość wyjściowych pętli:\n")
            f.write(f"  min = {min(loop_lengths)}\n")
            f.write(f"  max = {max(loop_lengths)}\n\n")

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