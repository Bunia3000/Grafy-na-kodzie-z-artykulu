from pathlib import Path
from collections import defaultdict
import math


PROJECT_ROOT = Path(r"C:\Users\Adam\.vscode\grafy")
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "xyz_graphs"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "xyz_geometry_report.txt"

EXPECTED_EDGES = 3
EXPECTED_POINTS_PER_EDGE = 101

# Progi diagnostyczne — na razie ostrożne, tylko do wykrywania podejrzanych przypadków
VERY_SMALL_SEGMENT_THRESHOLD = 1e-10
SMALL_SEGMENT_THRESHOLD = 1e-6
LARGE_SEGMENT_RATIO_THRESHOLD = 20.0
TOP_SUSPICIOUS_TO_SAVE = 200


def euclidean_distance(p1, p2) -> float:
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2 +
        (p1[2] - p2[2]) ** 2
    )


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(sorted_values, q: float) -> float:
    """
    Prosty percentyl dla listy już posortowanej.
    q w zakresie 0..100
    """
    if not sorted_values:
        return 0.0

    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = (len(sorted_values) - 1) * (q / 100.0)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))

    if lower == upper:
        return sorted_values[lower]

    weight = pos - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def format_stats(values, label: str) -> str:
    if not values:
        return f"{label}: brak danych\n"

    values_sorted = sorted(values)
    return (
        f"{label}:\n"
        f"  min   = {values_sorted[0]:.10f}\n"
        f"  p01   = {percentile(values_sorted, 1):.10f}\n"
        f"  p05   = {percentile(values_sorted, 5):.10f}\n"
        f"  mean  = {mean(values_sorted):.10f}\n"
        f"  med   = {percentile(values_sorted, 50):.10f}\n"
        f"  p95   = {percentile(values_sorted, 95):.10f}\n"
        f"  p99   = {percentile(values_sorted, 99):.10f}\n"
        f"  max   = {values_sorted[-1]:.10f}\n"
    )


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

            edge_id = int(parts[0])
            point_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])

            edges[edge_id].append((point_id, x, y, z))

    return dict(edges)


def extract_edge_points(edge_records):
    """
    Zamienia listę rekordów:
    [(point_id, x, y, z), ...]
    na listę punktów:
    [(x, y, z), ...]
    """
    return [(x, y, z) for _, x, y, z in edge_records]


def edge_segment_lengths(points):
    lengths = []
    for i in range(len(points) - 1):
        lengths.append(euclidean_distance(points[i], points[i + 1]))
    return lengths


def polyline_length(points):
    return sum(edge_segment_lengths(points))


def analyze_graph(file_path: Path):
    """
    Analiza jednego grafu.

    Zwraca słownik ze statystykami grafu i listą ostrzeżeń.
    """
    edges = read_xyz_file(file_path)

    warnings = []
    all_points = []

    edge_lengths = []
    all_segment_lengths = []

    zero_segments = 0
    tiny_segments = 0
    suspicious_large_jump = False

    # Zakładamy edge 0,1,2 po wcześniejszej walidacji strukturalnej
    for edge_id in [0, 1, 2]:
        edge_records = edges[edge_id]
        points = extract_edge_points(edge_records)
        all_points.extend(points)

        seg_lengths = edge_segment_lengths(points)
        all_segment_lengths.extend(seg_lengths)

        if seg_lengths:
            edge_lengths.append(sum(seg_lengths))

            min_seg = min(seg_lengths)
            max_seg = max(seg_lengths)
            avg_seg = mean(seg_lengths)

            zero_segments += sum(1 for x in seg_lengths if x <= VERY_SMALL_SEGMENT_THRESHOLD)
            tiny_segments += sum(1 for x in seg_lengths if x <= SMALL_SEGMENT_THRESHOLD)

            if avg_seg > 0 and max_seg / avg_seg > LARGE_SEGMENT_RATIO_THRESHOLD:
                suspicious_large_jump = True
                warnings.append(
                    f"Krawędź {edge_id}: duży skok segmentu "
                    f"(max/mean = {max_seg / avg_seg:.3f})"
                )
        else:
            edge_lengths.append(0.0)
            warnings.append(f"Krawędź {edge_id}: brak segmentów")

    # Wspólne wierzchołki
    start_vertex = (
        edges[0][0][1],
        edges[0][0][2],
        edges[0][0][3],
    )
    end_vertex = (
        edges[0][-1][1],
        edges[0][-1][2],
        edges[0][-1][3],
    )

    vertex_distance = euclidean_distance(start_vertex, end_vertex)

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    zs = [p[2] for p in all_points]

    bbox = {
        "xmin": min(xs),
        "xmax": max(xs),
        "ymin": min(ys),
        "ymax": max(ys),
        "zmin": min(zs),
        "zmax": max(zs),
    }

    # Proste wskaźniki podejrzaności
    suspicion_score = 0
    suspicion_score += zero_segments * 100
    suspicion_score += tiny_segments * 10
    suspicion_score += 50 if suspicious_large_jump else 0

    if vertex_distance <= VERY_SMALL_SEGMENT_THRESHOLD:
        warnings.append("Wierzchołki początkowy i końcowy są niemal identyczne geometrycznie")
        suspicion_score += 1000

    return {
        "file_path": file_path,
        "class_name": file_path.parent.name,
        "vertex_distance": vertex_distance,
        "edge_lengths": edge_lengths,
        "all_segment_lengths": all_segment_lengths,
        "zero_segments": zero_segments,
        "tiny_segments": tiny_segments,
        "bbox": bbox,
        "warnings": warnings,
        "suspicion_score": suspicion_score,
    }


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(DATA_ROOT.glob("*/*.xyz"))

    total_files = 0
    failed_files = 0

    global_vertex_distances = []
    global_edge_lengths = []
    global_segment_lengths = []

    class_stats = defaultdict(lambda: {
        "files": 0,
        "vertex_distances": [],
        "edge_lengths": [],
        "segment_lengths": [],
        "zero_segments": 0,
        "tiny_segments": 0,
        "warnings": 0,
    })

    suspicious_files = []
    read_errors = []

    global_xmin = math.inf
    global_xmax = -math.inf
    global_ymin = math.inf
    global_ymax = -math.inf
    global_zmin = math.inf
    global_zmax = -math.inf

    for file_path in all_files:
        total_files += 1

        try:
            info = analyze_graph(file_path)
        except Exception as exc:
            failed_files += 1
            read_errors.append((file_path, str(exc)))
            continue

        class_name = info["class_name"]
        class_stats[class_name]["files"] += 1
        class_stats[class_name]["vertex_distances"].append(info["vertex_distance"])
        class_stats[class_name]["edge_lengths"].extend(info["edge_lengths"])
        class_stats[class_name]["segment_lengths"].extend(info["all_segment_lengths"])
        class_stats[class_name]["zero_segments"] += info["zero_segments"]
        class_stats[class_name]["tiny_segments"] += info["tiny_segments"]
        class_stats[class_name]["warnings"] += len(info["warnings"])

        global_vertex_distances.append(info["vertex_distance"])
        global_edge_lengths.extend(info["edge_lengths"])
        global_segment_lengths.extend(info["all_segment_lengths"])

        bbox = info["bbox"]
        global_xmin = min(global_xmin, bbox["xmin"])
        global_xmax = max(global_xmax, bbox["xmax"])
        global_ymin = min(global_ymin, bbox["ymin"])
        global_ymax = max(global_ymax, bbox["ymax"])
        global_zmin = min(global_zmin, bbox["zmin"])
        global_zmax = max(global_zmax, bbox["zmax"])

        if info["suspicion_score"] > 0:
            suspicious_files.append(info)

    suspicious_files.sort(key=lambda x: x["suspicion_score"], reverse=True)

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("Raport geometrii surowych grafów .xyz\n")
        f.write("====================================\n\n")

        f.write(f"Katalog danych: {DATA_ROOT}\n")
        f.write(f"Liczba wszystkich plików: {total_files}\n")
        f.write(f"Liczba błędów odczytu: {failed_files}\n\n")

        f.write("Zakres współrzędnych w całym zbiorze:\n")
        f.write(f"  x: [{global_xmin:.10f}, {global_xmax:.10f}]\n")
        f.write(f"  y: [{global_ymin:.10f}, {global_ymax:.10f}]\n")
        f.write(f"  z: [{global_zmin:.10f}, {global_zmax:.10f}]\n\n")

        f.write(format_stats(global_vertex_distances, "Odległość między wspólnymi wierzchołkami"))
        f.write("\n")
        f.write(format_stats(global_edge_lengths, "Długości krawędzi"))
        f.write("\n")
        f.write(format_stats(global_segment_lengths, "Długości segmentów między kolejnymi punktami"))
        f.write("\n")

        total_zero_segments = sum(v["zero_segments"] for v in class_stats.values())
        total_tiny_segments = sum(v["tiny_segments"] for v in class_stats.values())
        total_warnings = sum(v["warnings"] for v in class_stats.values())

        f.write("Podsumowanie globalne:\n")
        f.write(f"  Segmenty zerowej długości: {total_zero_segments}\n")
        f.write(f"  Segmenty bardzo małe (<= {SMALL_SEGMENT_THRESHOLD}): {total_tiny_segments}\n")
        f.write(f"  Liczba ostrzeżeń: {total_warnings}\n\n")

        f.write("Podsumowanie per klasa:\n")
        f.write("-----------------------\n\n")

        for class_name in sorted(class_stats):
            stats = class_stats[class_name]
            f.write(f"Klasa: {class_name}\n")
            f.write(f"  Liczba plików: {stats['files']}\n")
            f.write(format_stats(stats["vertex_distances"], "  Odległość między wierzchołkami"))
            f.write(format_stats(stats["edge_lengths"], "  Długości krawędzi"))
            f.write(format_stats(stats["segment_lengths"], "  Długości segmentów"))
            f.write(f"  Segmenty zerowej długości: {stats['zero_segments']}\n")
            f.write(f"  Segmenty bardzo małe: {stats['tiny_segments']}\n")
            f.write(f"  Liczba ostrzeżeń: {stats['warnings']}\n")
            f.write("\n")

        f.write("Najbardziej podejrzane pliki:\n")
        f.write("-----------------------------\n\n")

        if not suspicious_files:
            f.write("Brak podejrzanych plików według obecnych progów.\n")
        else:
            for info in suspicious_files[:TOP_SUSPICIOUS_TO_SAVE]:
                f.write(f"{info['file_path']}\n")
                f.write(f"  suspicion_score: {info['suspicion_score']}\n")
                f.write(f"  vertex_distance: {info['vertex_distance']:.10f}\n")
                f.write(
                    f"  edge_lengths: "
                    f"{', '.join(f'{x:.10f}' for x in info['edge_lengths'])}\n"
                )
                f.write(f"  zero_segments: {info['zero_segments']}\n")
                f.write(f"  tiny_segments: {info['tiny_segments']}\n")
                if info["warnings"]:
                    for warning in info["warnings"]:
                        f.write(f"  warning: {warning}\n")
                f.write("\n")

        if read_errors:
            f.write("Błędy odczytu:\n")
            f.write("--------------\n\n")
            for file_path, err in read_errors:
                f.write(f"{file_path}\n")
                f.write(f"  {err}\n\n")

    print(f"Sprawdzono plików: {total_files}")
    print(f"Błędy odczytu: {failed_files}")
    print(f"Podejrzane pliki: {len(suspicious_files)}")
    print(f"Raport zapisano do: {REPORT_PATH}")


if __name__ == "__main__":
    main()