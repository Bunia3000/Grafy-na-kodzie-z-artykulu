# 📊 Theta Graph Dataset Pipeline

This folder contains a complete pipeline for constructing a machine learning dataset from spatial theta graphs (0-curves) and derived structures, as described in the referenced paper.

The pipeline processes raw geometric data into standardized representations (`1knot`, `3loops`) and computes geometric features (`StS`, `StA`) suitable for ML models.

---

## 🔁 Pipeline Overview

```
Excel → raw XYZ → validation → normalization
      → 1knot / 3loops construction
      → resampling (256 pts)
      → canonicalization
      → feature computation (StS, StA)
```

---

## 📂 Scripts Description

### 1. 📥 Data Extraction

* **`extract_xyz_from_excel.py`**
  Extracts 3D coordinates of theta graphs from Excel files and saves them as `.xyz` files (`edge_id, point_id, x, y, z`).

* **`extract_xyz_from_exel_run-3.py`**
  Variant of the above script for a different dataset (`run-3`), with additional dataset statistics.

---

### 2. ✅ Data Validation & Diagnostics

* **`validate_xyz_graphs.py`**
  Validates structural correctness of `.xyz` graphs:

  * exactly 3 edges
  * correct `edge_id` values (0,1,2)
  * 101 points per edge
  * consistent indexing

* **`check_xyz_geometry.py`**
  Performs geometric sanity checks:

  * detects unusually short segments
  * identifies large jumps or discontinuities
  * flags potentially corrupted graphs

---

### 3. ⚖️ Normalization

* **`normalize_graphs.py`**
  Normalizes graphs into a common reference frame:

  * centers graph at origin
  * scales vertex distance to 1
  * aligns main axis with Z-axis

This ensures geometric consistency across the dataset.

---

### 4. 🔗 Graph Transformations

* **`build_1knot_xyz.py`**
  Converts a theta graph into a single closed curve (`1knot`) by concatenating edges.
  Output: ~299-point curve.

* **`build_3loops_xyz.py`**
  Constructs three loops (`3loops`) from edge pairs:

  * (0,1), (0,2), (1,2)
    Each loop is a closed curve (~201 points).

---

### 5. 🔄 Resampling

* **`resample_1knot_xyz_256.py`**
  Resamples `1knot` curves to **256 points** using arc-length interpolation.

* **`resample_3loops_xyz_256.py`**
  Resamples each loop in `3loops` to **256 points**.

This step standardizes input size for ML models.

---

### 6. 🧭 Canonicalization

* **`canonicalize_1knot_xyz_256.py`**
  Removes ambiguity in curve representation by:

  * selecting canonical starting point
  * fixing traversal direction

* **`canonicalize_3loops_xyz_256.py`**
  Canonicalizes:

  * each loop individually
  * ordering of the three loops

Ensures identical graphs have identical representations.

---

### 7. 🧠 Feature Computation

* **`compute_1knot_sts_sta.py`**
  Computes geometric features for `1knot`:

  * `StS` — segment-to-segment interaction matrix
  * `StA` — segment-based feature vector

* **`compute_3loops_sts_sta.py`**
  Computes `StS` and `StA` separately for each loop in `3loops`, producing ML-ready tensors.

---

## 📌 Key Concepts

* **Theta graph (0-curve):**
  Two vertices connected by three edges.

* **1knot:**
  Single closed curve derived from all edges.

* **3loops:**
  Three closed curves built from edge pairs.

* **StS / StA:**
  Geometric descriptors based on segment interactions and orientations.

---

## 🎯 Output

The pipeline produces:

* standardized `.xyz` representations
* resampled curves (256 points)
* canonical graph encodings
* feature files (`.npy`, `.txt`) ready for ML

---

## 🚀 Usage Notes

* Run scripts in pipeline order
* Ensure intermediate directories exist
* Validate data before normalization
* Canonicalization is critical for dataset consistency

