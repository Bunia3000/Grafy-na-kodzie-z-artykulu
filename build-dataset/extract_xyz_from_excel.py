import re
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\Adam\.vscode\grafy")
EXCEL_PATH = PROJECT_ROOT / "dataset" / "100_results_run-2.xlsx"
OUT_ROOT = PROJECT_ROOT / "data" / "raw" / "xyz_graphs"
SKIPPED_LOG_PATH = PROJECT_ROOT / "data" / "metadata" / "logs" / "extract_run_2_skipped_rows.txt"

TOPO_RE = re.compile(r"t(\d+_\d+)")
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_graph_type_from_text(text: str) -> Optional[str]:
    """Wyciąga typ grafu, np. '3_1', z tekstu zawierającego np. 't3_1'."""
    if not text:
        return None

    match = TOPO_RE.search(text)
    return match.group(1) if match else None


def parse_arc_points(text: str) -> List[Tuple[float, float, float]]:
    """
    Wyciąga punkty 3D z tekstu.
    Zwraca listę punktów [(x, y, z), ...].
    """
    nums = [float(x) for x in FLOAT_RE.findall(text)]

    if not nums:
        return []

    if len(nums) % 3 != 0:
        raise ValueError(
            f"Arc: liczby={len(nums)} nie dzielą się przez 3. Początek: {text[:120]!r}"
        )

    return [
        (nums[i], nums[i + 1], nums[i + 2])
        for i in range(0, len(nums), 3)
    ]


def write_xyz(out_path: Path, arcs: List[List[Tuple[float, float, float]]]) -> None:
    """
    Zapisuje jeden graf theta do pliku .xyz w formacie:
    edge_id point_id x y z
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for edge_id, pts in enumerate(arcs):
            for point_id, (x, y, z) in enumerate(pts):
                f.write(f"{edge_id} {point_id} {x} {y} {z}\n")


def row_to_text(row) -> str:
    """
    Łączy wszystkie komórki wiersza Excela w jeden string.
    Odporne na puste komórki i NaN.
    """
    parts = []

    for value in row:
        if value is None:
            continue

        if isinstance(value, float) and pd.isna(value):
            continue

        text = str(value).strip()
        if text:
            parts.append(text)

    return " ".join(parts)


def extract_arcs_from_row_text(row_text: str) -> Optional[Tuple[str, List[str]]]:
    """
    Z wiersza tekstowego próbuje wyciągnąć:
    - typ grafu
    - 3 tekstowe reprezentacje łuków

    Heurystyka:
    - typ grafu rozpoznajemy z 'tX_Y'
    - łuk to segment z dużą liczbą liczb i zwykle zawierający przecinki
    """
    gtype = parse_graph_type_from_text(row_text)
    if gtype is None:
        return None

    segments = [segment.strip() for segment in row_text.split("|") if segment.strip()]

    arc_candidates = []
    for segment in segments:
        nums = FLOAT_RE.findall(segment)

        # Heurystyka: łuk ma dużo liczb (punkty 3D) i zwykle zawiera przecinki.
        if len(nums) >= 60 and "," in segment:
            arc_candidates.append(segment)

    if len(arc_candidates) < 3:
        return None

    # Bierzemy 3 ostatnie kandydaty, tak jak w Twojej dotychczasowej wersji.
    return gtype, arc_candidates[-3:]


def log_skipped_row(log_path: Path, reason: str, row_idx: int, row_text: str) -> None:
    """
    Zapisuje jedną pominiętą linię do loga.
    Wiersz jest spłaszczony do jednej linii.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    compact = " ".join(str(row_text).replace("\t", " ").split())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] row={row_idx} reason={reason} | {compact}\n"

    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)


def reset_log_file(log_path: Path) -> None:
    """Czyści / tworzy plik loga przed nowym uruchomieniem skryptu."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as f:
        f.write("# Skipped rows log\n")


def make_output_filename(index: int) -> str:
    """
    Zwraca nazwę pliku z zerami z przodu, np.:
    1 -> 0001.xyz
    25 -> 0025.xyz
    """
    return f"{index:04d}.xyz"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    reset_log_file(SKIPPED_LOG_PATH)

    df = pd.read_excel(EXCEL_PATH, header=None)

    counters: dict[str, int] = {}
    written = 0
    skipped = 0
    errors = 0
    empty = 0

    for row_idx, row in df.iterrows():
        text = row_to_text(row)

        if not text:
            empty += 1
            log_skipped_row(SKIPPED_LOG_PATH, "empty_row_text", int(row_idx), "")
            continue

        extracted = extract_arcs_from_row_text(text)
        if extracted is None:
            skipped += 1

            gtype = parse_graph_type_from_text(text)
            if gtype is None:
                log_skipped_row(SKIPPED_LOG_PATH, "no_graph_type_found", int(row_idx), text)
            else:
                log_skipped_row(
                    SKIPPED_LOG_PATH,
                    "less_than_3_arc_candidates",
                    int(row_idx),
                    text,
                )
            continue

        gtype, arcs_text = extracted

        try:
            arc1 = parse_arc_points(arcs_text[0])
            arc2 = parse_arc_points(arcs_text[1])
            arc3 = parse_arc_points(arcs_text[2])
        except Exception as exc:
            errors += 1
            log_skipped_row(
                SKIPPED_LOG_PATH,
                f"arc_parse_exception:{type(exc).__name__}:{exc}",
                int(row_idx),
                text,
            )
            continue

        if not arc1 or not arc2 or not arc3:
            skipped += 1
            log_skipped_row(SKIPPED_LOG_PATH, "empty_arc_after_parse", int(row_idx), text)
            continue

        sample_index = counters.get(gtype, 0) + 1
        counters[gtype] = sample_index

        out_file = OUT_ROOT / gtype / make_output_filename(sample_index)
        write_xyz(out_file, [arc1, arc2, arc3])
        written += 1

    print(f"Excel: {EXCEL_PATH}")
    print(f"Katalog wyjściowy: {OUT_ROOT}")
    print(f"Zapisano: {written} plików .xyz")
    print(f"Pominięto: {skipped} wierszy (brak typu / brak 3 arc / puste arc)")
    print(f"Błędy parsowania arc: {errors}")
    print(f"Puste wiersze: {empty}")
    print(f"Log: {SKIPPED_LOG_PATH}")

    if counters:
        print("\nLiczba zapisanych plików per klasa:")
        for gtype in sorted(counters):
            print(f"  {gtype}: {counters[gtype]}")


if __name__ == "__main__":
    main()