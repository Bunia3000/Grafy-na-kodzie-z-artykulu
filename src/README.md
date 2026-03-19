# 🤖 Pipeline ML dla grafów theta (0-krzywych)

Ta część projektu zawiera skrypty do:

* przygotowania danych do treningu
* analizy i czyszczenia datasetu
* definiowania modeli
* uruchamiania eksperymentów ML

Celem jest **odtworzenie eksperymentu z artykułu (StA)** , ale zamiast klasyfikacji węzłów — zastosowanie go do **grafów theta (0-krzywych)**.

---

## 🧠 Relacja do kodu z artykułu

Poniższe pliki są przerobionymi wersjami kodu z artykułu:

* `main.py` ← na podstawie `main_old.py`
* `helpers.py` ← na podstawie `helpers_old.py`
* `loaders.py` ← na podstawie `loaders_old.py`
* `models.py` ← na podstawie `models_old.py`

👉 Pliki `*_old.py` to **oryginalny kod z artykułu** — pozostawione jako referencja (nie są tutaj opisywane).

---

## 🔁 Przegląd pipeline’u treningowego

```id="ml-pipeline-pl"
dane processed → splity → diagnostyka / filtrowanie
               → loaders → model → trening → ewaluacja
```

---

## 📂 Główne skrypty treningowe

### 🚀 `main.py`

Główny skrypt do trenowania i testowania modeli.

* parsuje argumenty CLI (dtype, model, split, itd.)
* ładuje dane na podstawie splitów
* buduje model przez `helpers.py`
* uruchamia trening lub test
* zapisuje:

  * historię treningu
  * macierz pomyłek
  * metryki

👉 To jest **główny entrypoint do eksperymentów ML**. 

---

### 🧩 `helpers.py`

Warstwa pośrednia między konfiguracją a modelem.

* waliduje:

  * typ danych (`1knot_sta`, `3loops_sts`, itd.)
  * typ sieci (FFNN, CNN, RNN, itd.)
* generuje modele z `models.py`

👉 Pełni rolę **fabryki modeli i konfiguracji**. 

---

### 📦 `loaders.py`

Odpowiada za wczytywanie danych do formatu ML.

Obsługiwane reprezentacje:

* `xyz` — geometria
* `sta` — 1D writhe
* `sts` — 2D writhe

Funkcjonalności:

* ścisła walidacja formatów danych
* konwersja do tensorów
* obsługa:

  * `1knot` → `(256, …)`
  * `3loops` → `(3, 256, …)`

👉 Kluczowy element: **pliki → tensory dla modelu**. 

---

### 🧠 `models.py`

Definicje architektur sieci neuronowych.

Zawiera:

* FFNN (sieci gęste)
* CNN (dla macierzy StS)
* RNN / LSTM (dla danych sekwencyjnych jak StA)

Dodatkowo:

* automatyczne dopasowanie kształtu wejścia
* obsługa danych wielopętlowych (`3loops`)

👉 Implementacja modeli inspirowanych artykułem, dostosowana do grafów. 

---

## 📊 Przygotowanie danych

### 📁 `make_splits.py`

Tworzy podział danych na train / val / test.

* balansuje klasy (downsampling)
* zapewnia brak overlapów między splitami
* zapisuje wynik do JSON

👉 Zapewnia **uczciwe i powtarzalne eksperymenty**. 

---

### 🚫 `detect_outliers_and_make_filtered_splits.py`

Wykrywa outliery i tworzy „oczyszczone” splity.

* analizuje cechy z `StA` i `StS`
* używa robust z-score (MAD)
* wykrywa anomalne próbki
* usuwa je ze zbioru treningowego
* zapisuje nowe splity

👉 Poprawia jakość danych przed treningiem. 

---

## 🔍 Diagnostyka i analiza danych

### 🔬 `diagnose_data.py`

Zaawansowana analiza datasetu.

* porównuje klasy (np. `0_1 vs 3_1`)
* liczy statystyki dla `StA` i `StS`
* wykrywa:

  * proste cechy rozróżniające
  * duplikaty (hashowanie)
* wykonuje PCA
* generuje wykresy

👉 Pomaga zrozumieć **co model „widzi” w danych**. 

---

### 👀 `inspect_data.py`

Szybki podgląd danych.

* wypisuje:

  * min / max / mean / std
* działa dla `StA` i `StS`

👉 Prosty debug przed treningiem. 

---

## 🧪 Kod referencyjny (artykuł)

* `main_old.py`
* `helpers_old.py`
* `loaders_old.py`
* `models_old.py`

👉 Oryginalna implementacja z artykułu — używana jako punkt odniesienia.

---

## 🎯 Główna idea projektu

Zgodnie z artykułem:

> lokalne cechy geometryczne (np. writhe) pozwalają ML odtworzyć topologię krzywej 

### W artykule:

* klasyfikacja **węzłów**

### W tym projekcie:

* klasyfikacja **grafów theta (0-krzywych)**
* wykorzystanie:

  * `StA` (cechy 1D)
  * `StS` (cechy 2D)
  * reprezentacji `1knot` i `3loops`

---

## 🧩 Co to umożliwia

* sprawdzenie czy **local writhe działa dla grafów (nie tylko węzłów)**
* porównanie reprezentacji:

  * geometria (`xyz`) vs cechy (`StA`, `StS`)
* testowanie różnych architektur NN
* budowę pełnego pipeline’u ML (data → model → wyniki)

---

## 🚀 Przykładowe użycie

```bash id="ml-run-example"
python main.py \
  --data_root data/processed \
  --dtype 1knot_sta \
  --split_file splits/01v31_seed42.json \
  --mode train \
  --net RNN \
  --epochs 50 \
  --checkpoint_dir results/run1
```

---

## 📏 Baseline: Proste klasyfikatory

### 🧪 `benchmark_simple_classifiers.py`

Skrypt do benchmarkowania prostych modeli ML na przygotowanych danych.

👉 Jego celem jest sprawdzenie:

* czy dane **same w sobie są separowalne**,
* czy skomplikowane sieci NN są rzeczywiście potrzebne.

---

### ⚙️ Co robi skrypt

Dla wybranych eksperymentów (np. `0_1 vs 3_1`, `3_1 vs 4_1`):

1. **Wczytuje dane** (`1knot`, `3loops`, `StA`, `StS`)

2. **Ekstrahuje proste cechy statystyczne**, m.in.:

   * średnia, odchylenie standardowe
   * percentyle (1%, 5%, 50%, 99%)
   * norma L2
   * wartości bezwzględne
   * (dla `StA`) różnice między kolejnymi punktami
   * (dla `StS`) statystyki diagonalne i poza diagonalą

3. (Opcjonalnie) **przycina outliery** (`clipping 1–99%`)

4. Trenuje dwa klasyczne modele:

   * **Logistic Regression**
   * **Random Forest**

5. Ewaluacja:

   * accuracy
   * macro F1
   * macierz pomyłek


