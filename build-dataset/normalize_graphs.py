from pathlib import Path
from collections import defaultdict
import math


PROJECT_ROOT = Path(r"C:\Users\Adam\.vscode\grafy")
INPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "xyz_graphs"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "normalized_graphs"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "normalize_graphs_report.txt"

EXPECTED_EDGES = [0, 1, 2]
EXPECTED_POINTS_PER_EDGE = 101
EPS = 1e-12


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


def validate_graph_structure(edges: dict[int, list[tuple[int, float, float, float]]], file_path: Path) -> None:
    """
    Sprawdza podstawową strukturę grafu:
    - edge_id = 0,1,2
    - 101 punktów na krawędź
    - point_id = 0..100
    - wspólne początki i końce
    """
    edge_ids = sorted(edges.keys())
    if edge_ids != EXPECTED_EDGES:
        raise ValueError(f"{file_path} | niepoprawne edge_id: {edge_ids}")

    expected_point_ids = list(range(EXPECTED_POINTS_PER_EDGE))
    starts = []
    ends = []

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

        starts.append((records[0][1], records[0][2], records[0][3]))
        ends.append((records[-1][1], records[-1][2], records[-1][3]))

    if not all(points_equal(starts[0], s) for s in starts[1:]):
        raise ValueError(f"{file_path} | początki krawędzi nie są wspólnym wierzchołkiem")

    if not all(points_equal(ends[0], e) for e in ends[1:]):
        raise ValueError(f"{file_path} | końce krawędzi nie są wspólnym wierzchołkiem")

    if points_equal(starts[0], ends[0]):
        raise ValueError(f"{file_path} | wierzchołki początkowy i końcowy są identyczne")


def points_equal(p1: tuple[float, float, float], p2: tuple[float, float, float], tol: float = 1e-9) -> bool:
    return (
        abs(p1[0] - p2[0]) <= tol and
        abs(p1[1] - p2[1]) <= tol and
        abs(p1[2] - p2[2]) <= tol
    )


def vec_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_scale(v: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (v[0] * s, v[1] * s, v[2] * s)


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(dot(v, v))


def normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = norm(v)
    if n <= EPS:
        raise ValueError("Nie można znormalizować wektora o zerowej długości")
    return (v[0] / n, v[1] / n, v[2] / n)


def midpoint(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)


def mat_vec_mul(m: list[list[float]], v: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def identity_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def rotation_matrix_from_vector_to_z(u: tuple[float, float, float]) -> list[list[float]]:
    """
    Zwraca macierz rotacji, która obraca znormalizowany wektor u
    na wektor (0,0,1).
    """
    target = (0.0, 0.0, 1.0)
    c = dot(u, target)

    # Już na +Z
    if abs(c - 1.0) <= 1e-12:
        return identity_matrix()

    # Dokładnie na -Z -> obrót o 180° wokół osi X
    if abs(c + 1.0) <= 1e-12:
        return [
            [1.0, 0.0,  0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]

    v = cross(u, target)
    s = norm(v)

    vx = [
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ]

    vx2 = mat_mul(vx, vx)

    factor = (1.0 - c) / (s * s)

    I = identity_matrix()
    R = mat_add(mat_add(I, vx), mat_scale(vx2, factor))
    return R


def mat_add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [a[i][j] + b[i][j] for j in range(3)]
        for i in range(3)
    ]


def mat_scale(a: list[list[float]], s: float) -> list[list[float]]:
    return [
        [a[i][j] * s for j in range(3)]
        for i in range(3)
    ]


def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    out = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(3))
    return out


def normalize_graph(edges: dict[int, list[tuple[int, float, float, float]]]) -> dict[int, list[tuple[int, float, float, float]]]:
    """
    Normalizuje graf:
    1. translacja: środek odcinka między wspólnymi wierzchołkami -> (0,0,0)
    2. skala: ||v2 - v1|| = 1
    3. rotacja: kierunek v1->v2 -> +Z
    """
    start = (edges[0][0][1], edges[0][0][2], edges[0][0][3])
    end = (edges[0][-1][1], edges[0][-1][2], edges[0][-1][3])

    center = midpoint(start, end)
    axis_vec = vec_sub(end, start)
    axis_len = norm(axis_vec)

    if axis_len <= EPS:
        raise ValueError("Odległość między wspólnymi wierzchołkami jest zbyt mała")

    axis_unit = normalize(axis_vec)
    R = rotation_matrix_from_vector_to_z(axis_unit)

    normalized_edges = {}

    for edge_id, records in edges.items():
        new_records = []
        for point_id, x, y, z in records:
            p = (x, y, z)

            # translacja
            p = vec_sub(p, center)

            # skala
            p = vec_scale(p, 1.0 / axis_len)

            # rotacja
            p = mat_vec_mul(R, p)

            new_records.append((point_id, p[0], p[1], p[2]))

        normalized_edges[edge_id] = new_records

    return normalized_edges


def write_xyz_file(file_path: Path, edges: dict[int, list[tuple[int, float, float, float]]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        for edge_id in sorted(edges.keys()):
            for point_id, x, y, z in edges[edge_id]:
                f.write(f"{edge_id} {point_id} {x:.12f} {y:.12f} {z:.12f}\n")


def verify_normalization(edges: dict[int, list[tuple[int, float, float, float]]], file_path: Path) -> tuple[float, float, float]:
    """
    Szybki test po normalizacji:
    - środek odcinka między wierzchołkami powinien być blisko (0,0,0)
    - odległość między wierzchołkami powinna być bliska 1
    - oś v1->v2 powinna być bliska +Z
    """
    start = (edges[0][0][1], edges[0][0][2], edges[0][0][3])
    end = (edges[0][-1][1], edges[0][-1][2], edges[0][-1][3])

    center = midpoint(start, end)
    axis_vec = vec_sub(end, start)
    axis_len = norm(axis_vec)

    center_error = norm(center)
    length_error = abs(axis_len - 1.0)

    axis_unit = normalize(axis_vec)
    z_axis = (0.0, 0.0, 1.0)
    direction_error = norm(vec_sub(axis_unit, z_axis))

    return center_error, length_error, direction_error


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(INPUT_ROOT.glob("*/*.xyz"))

    processed = 0
    failed = 0
    failures = []

    center_errors = []
    length_errors = []
    direction_errors = []

    class_counts = defaultdict(int)

    for in_file in all_files:
        rel_path = in_file.relative_to(INPUT_ROOT)
        out_file = OUTPUT_ROOT / rel_path

        try:
            edges = read_xyz_file(in_file)
            validate_graph_structure(edges, in_file)

            normalized_edges = normalize_graph(edges)
            validate_graph_structure(normalized_edges, in_file)

            center_error, length_error, direction_error = verify_normalization(normalized_edges, in_file)
            center_errors.append(center_error)
            length_errors.append(length_error)
            direction_errors.append(direction_error)

            write_xyz_file(out_file, normalized_edges)

            processed += 1
            class_counts[in_file.parent.name] += 1

        except Exception as exc:
            failed += 1
            failures.append((in_file, str(exc)))

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("Raport normalizacji grafów\n")
        f.write("==========================\n\n")
        f.write(f"Wejście: {INPUT_ROOT}\n")
        f.write(f"Wyjście: {OUTPUT_ROOT}\n\n")
        f.write(f"Przetworzono poprawnie: {processed}\n")
        f.write(f"Błędy: {failed}\n\n")

        if center_errors:
            f.write("Błędy po normalizacji:\n")
            f.write(f"  max center_error    = {max(center_errors):.16e}\n")
            f.write(f"  max length_error    = {max(length_errors):.16e}\n")
            f.write(f"  max direction_error = {max(direction_errors):.16e}\n\n")

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