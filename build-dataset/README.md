# 📊 Theta Graph Dataset Pipeline

Ten folder zawiera kompletny pipeline do budowy zbioru danych dla uczenia maszynowego na podstawie przestrzennych grafów theta (0-curves), zgodnie z opisem w powiązanej publikacji.

Pipeline przetwarza surowe dane geometryczne do ustandaryzowanych reprezentacji (`1knot`, `3loops`) oraz oblicza cechy geometryczne (`StS`, `StA`) odpowiednie dla modeli ML.

---

## 🔁 Przegląd pipeline’u

```
Excel → surowe XYZ → walidacja → normalizacja
      → konstrukcja 1knot / 3loops
      → resampling (256 punktów)
      → kanonikalizacja
      → obliczanie cech (StS, StA)
```

---

## 📂 Opis skryptów

### 1. 📥 Ekstrakcja danych

**`extract_xyz_from_excel.py`**
Ekstrahuje współrzędne 3D grafów theta z plików Excel i zapisuje je jako pliki `.xyz` (`edge_id`, `point_id`, `x`, `y`, `z`).

**`extract_xyz_from_exel_run-3.py`**
Wariant powyższego skryptu dla innego zbioru danych (run-3), zawiera dodatkowe statystyki datasetu.

---

### 2. ✅ Walidacja danych i diagnostyka

**`validate_xyz_graphs.py`**
Sprawdza poprawność strukturalną grafów `.xyz`:

* dokładnie 3 krawędzie
* poprawne wartości `edge_id` (0, 1, 2)
* 101 punktów na każdą krawędź
* spójne indeksowanie

**`check_xyz_geometry.py`**
Wykonuje testy geometryczne:

* wykrywa nietypowo krótkie segmenty
* identyfikuje duże skoki i nieciągłości
* oznacza potencjalnie uszkodzone grafy

---

### 3. ⚖️ Normalizacja

**`normalize_graphs.py`**
Normalizuje grafy do wspólnego układu odniesienia:

* centruje graf w początku układu (0,0,0)
* skaluje odległość między wierzchołkami do 1
* wyrównuje główną oś do osi Z

Zapewnia to spójność geometryczną w całym zbiorze danych.

---

### 4. 🔗 Transformacje grafów

**`build_1knot_xyz.py`**
Przekształca graf theta w jedną zamkniętą krzywą (`1knot`) przez konkatenację krawędzi.
Wyjście: ~299 punktów.

**`build_3loops_xyz.py`**
Tworzy trzy pętle (`3loops`) z par krawędzi:

* (0,1), (0,2), (1,2)
  Każda pętla to zamknięta krzywa (~201 punktów).

---

### 5. 🔄 Resampling

**`resample_1knot_xyz_256.py`**
Resampluje krzywe `1knot` do 256 punktów z użyciem interpolacji długości łuku.

**`resample_3loops_xyz_256.py`**
Resampluje każdą pętlę w `3loops` do 256 punktów.

Ten krok ujednolica rozmiar wejścia dla modeli ML.

---

### 6. 🧭 Kanonikalizacja

**`canonicalize_1knot_xyz_256.py`**
Usuwa niejednoznaczność reprezentacji krzywej poprzez:

* wybór kanonicznego punktu początkowego
* ustalenie kierunku przejścia

**`canonicalize_3loops_xyz_256.py`**
Kanonikalizuje:

* każdą pętlę osobno
* kolejność trzech pętli

Zapewnia, że identyczne grafy mają identyczne reprezentacje.

---

### 7. 🧠 Obliczanie cech

**`compute_1knot_sts_sta.py`**
Oblicza cechy geometryczne dla `1knot`:

* **StS** — macierz interakcji segment–segment
* **StA** — wektor cech segmentowych

**`compute_3loops_sts_sta.py`**
Oblicza StS i StA osobno dla każdej pętli w `3loops`, tworząc tensory gotowe do ML.

---

## 📌 Kluczowe pojęcia

* **Theta graph (0-curve)**: dwa wierzchołki połączone trzema krawędziami
* **1knot**: pojedyncza zamknięta krzywa utworzona ze wszystkich krawędzi
* **3loops**: trzy zamknięte krzywe zbudowane z par krawędzi
* **StS / StA**: deskryptory geometryczne oparte na interakcjach i orientacjach segmentów

---

## 🎯 Wynik

Pipeline generuje:

* ustandaryzowane reprezentacje `.xyz`
* krzywe po resamplingu (256 punktów)
* kanoniczne reprezentacje grafów
* pliki cech (`.npy`, `.txt`) gotowe do ML
