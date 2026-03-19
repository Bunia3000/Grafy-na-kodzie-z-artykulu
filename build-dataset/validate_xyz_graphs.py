from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(r"C:\Users\Adam\.vscode\grafy")
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "xyz_graphs"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "xyz_validation_report.txt"

EXPECTED_EDGES = 3
EXPECTED_POINTS_PER_EDGE = 101
EXPECTED_POINT_IDS = list(range(EXPECTED_POINTS_PER_EDGE))


def read_xyz_file(file_path: Path):
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
                    f"Niepoprawna liczba kolumn w linii {line_no}: "
                    f"oczekiwano 5, dostano {len(parts)}"
                )

            try:
                edge_id = int(parts[0])
                point_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
            except ValueError as exc:
                raise ValueError(
                    f"Błąd parsowania w linii {line_no}: {exc}"
                ) from exc

            edges[edge_id].append((point_id, x, y, z))

    return dict(edges)


def validate_graph(edges: dict):
    """
    Sprawdza poprawność jednego grafu.
    Zwraca listę błędów. Jeśli pusta lista -> plik poprawny.
    """
    errors = []

    # 1. Czy są dokładnie 3 krawędzie
    edge_ids = sorted(edges.keys())
    if len(edge_ids) != EXPECTED_EDGES:
        errors.append(
            f"Zła liczba krawędzi: {len(edge_ids)} (oczekiwano {EXPECTED_EDGES})"
        )

    if edge_ids != [0, 1, 2]:
        errors.append(f"Niepoprawne edge_id: {edge_ids} (oczekiwano [0, 1, 2])")

    # Jeśli brak którejś krawędzi, dalsze testy mogą być bez sensu, ale próbujemy dalej.
    start_vertices = []
    end_vertices = []

    for edge_id in [0, 1, 2]:
        if edge_id not in edges:
            errors.append(f"Brak krawędzi edge_id={edge_id}")
            continue

        pts = edges[edge_id]

        # 2. Czy liczba punktów na krawędź = 101
        if len(pts) != EXPECTED_POINTS_PER_EDGE:
            errors.append(
                f"Krawędź {edge_id}: zła liczba punktów: {len(pts)} "
                f"(oczekiwano {EXPECTED_POINTS_PER_EDGE})"
            )

        # 3. Czy point_id są od 0 do 100
        point_ids = [p[0] for p in pts]
        if point_ids != EXPECTED_POINT_IDS:
            errors.append(
                f"Krawędź {edge_id}: niepoprawne point_id "
                f"(oczekiwano 0..100, dostano {point_ids[:5]}...{point_ids[-5:] if point_ids else []})"
            )

        # 4. Zbieramy wspólne wierzchołki
        if pts:
            start_vertex = (pts[0][1], pts[0][2], pts[0][3])
            end_vertex = (pts[-1][1], pts[-1][2], pts[-1][3])

            start_vertices.append(start_vertex)
            end_vertices.append(end_vertex)

    # 5. Czy wszystkie początki są takie same
    if start_vertices:
        first_start = start_vertices[0]
        for idx, vertex in enumerate(start_vertices[1:], start=1):
            if vertex != first_start:
                errors.append(
                    f"Niezgodne wierzchołki początkowe między krawędziami: "
                    f"{start_vertices}"
                )
                break

    # 6. Czy wszystkie końce są takie same
    if end_vertices:
        first_end = end_vertices[0]
        for idx, vertex in enumerate(end_vertices[1:], start=1):
            if vertex != first_end:
                errors.append(
                    f"Niezgodne wierzchołki końcowe między krawędziami: "
                    f"{end_vertices}"
                )
                break

    # 7. Czy początek i koniec grafu nie są identyczne
    # (nie musi to być błąd topologiczny zawsze, ale tutaj raczej oczekujemy 2 różnych wierzchołków)
    if start_vertices and end_vertices:
        if start_vertices[0] == end_vertices[0]:
            errors.append(
                "Wierzchołek początkowy i końcowy są identyczne "
                "(spodziewano się dwóch różnych wspólnych wierzchołków)"
            )

    return errors


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(DATA_ROOT.glob("*/*.xyz"))

    total_files = 0
    valid_files = 0
    invalid_files = 0

    class_counts = defaultdict(int)
    class_valid = defaultdict(int)
    class_invalid = defaultdict(int)

    invalid_details = []

    for file_path in all_files:
        total_files += 1
        graph_class = file_path.parent.name
        class_counts[graph_class] += 1

        try:
            edges = read_xyz_file(file_path)
            errors = validate_graph(edges)

            if errors:
                invalid_files += 1
                class_invalid[graph_class] += 1
                invalid_details.append((file_path, errors))
            else:
                valid_files += 1
                class_valid[graph_class] += 1

        except Exception as exc:
            invalid_files += 1
            class_invalid[graph_class] += 1
            invalid_details.append((file_path, [f"Błąd odczytu/pliku: {exc}"]))

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("Raport walidacji plików .xyz\n")
        f.write("============================\n\n")

        f.write(f"Katalog danych: {DATA_ROOT}\n")
        f.write(f"Liczba wszystkich plików: {total_files}\n")
        f.write(f"Poprawne pliki: {valid_files}\n")
        f.write(f"Błędne pliki: {invalid_files}\n\n")

        f.write("Podsumowanie per klasa\n")
        f.write("----------------------\n")
        for graph_class in sorted(class_counts):
            f.write(
                f"{graph_class}: razem={class_counts[graph_class]}, "
                f"poprawne={class_valid[graph_class]}, "
                f"błędne={class_invalid[graph_class]}\n"
            )

        f.write("\nSzczegóły błędnych plików\n")
        f.write("-------------------------\n")
        if not invalid_details:
            f.write("Brak błędnych plików.\n")
        else:
            for file_path, errors in invalid_details:
                f.write(f"\n{file_path}\n")
                for err in errors:
                    f.write(f"  - {err}\n")

    print(f"Sprawdzono plików: {total_files}")
    print(f"Poprawne: {valid_files}")
    print(f"Błędne: {invalid_files}")
    print(f"Raport zapisano do: {REPORT_PATH}")


if __name__ == "__main__":
    main()