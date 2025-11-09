# 🐨 Koala SVD Watermarking

A lightweight, end-to-end **SVD-based medical image watermarking** demo built with Streamlit.

Koala is designed for scenarios where **diagnostic image quality must be preserved**, while securely embedding patient or exam metadata that remains recoverable and robust against common image operations.

👉 **Live demo:** [Koala SVD Watermarking App](https://svdwatermarking-vf8zkevgdwuqpafkzficj5.streamlit.app/)

---

## 🌐 Overview

Koala provides a clean workflow:

1. **Home** – Concept, objectives, and method summary
2. **Embed** – Upload image/DICOM → generate watermark → embed via SVD → test robustness → download results
3. **Extract** – Upload watermarked image + key → recover watermark

Useful for:

* Research & experiments in medical image security
* Teaching SVD watermarking concepts
* Quick benchmarking of robustness vs. imperceptibility

---

## 🎯 Core Design Objectives

Koala’s behavior is guided by four measurable targets:

1. **Imperceptibility**

   * Aim for **PSNR ≥ 50 dB** between original and watermarked image.
   * Ensures diagnostic features remain visually intact.

2. **Accurate Extraction**

   * Under no-attack / ideal conditions:

     * **BER < 0.03**
     * **NCC ≥ 0.70**
   * Ensures metadata is reliably recoverable.

3. **Robustness**

   * Under typical manipulations:

     * **BER < 0.05**, **NCC ≥ 0.60** (goal)
   * Attacks evaluated in-app:

     * Rotation
     * Salt & pepper noise
     * Blur
     * Cropping

4. **Storage Efficiency**

   * Supports converting **DICOM → JPEG/PNG** with watermarking,
     reducing file size (up to ~80%) while preserving quality and hidden data.

---

## 🧠 Method: Why SVD & Why the Middle Band?

Koala uses **Singular Value Decomposition** on the host image:

* Naive approach: embed watermark across all singular values
  ➜ robust but degrades PSNR (~48 dB).

* **Koala improvement**:

  * Embed watermark **only in the middle singular values** (e.g. between `embed_low=0.2` and `embed_high=0.8` of the spectrum).
  * Top SVs: control global structure/contrast → leave untouched.
  * Bottom SVs: noise-sensitive → avoid for stability.
  * Middle band: **optimal trade-off** → high PSNR + strong robustness.

The app:

* Computes **PSNR** (imperceptibility).
* Computes **BER & NCC** between original and recovered watermark.
* Simulates attacks and reports post-attack metrics for quick evaluation.

---

## 🔧 Functional Breakdown

### 1. Home Page

* Explains:

  * Clinical motivation (patient data integrity, secure embedding).
  * Limitations of traditional watermarking.
  * SVD-based middle-band embedding idea.
* Acts as documentation inside the app; no inputs required.

---

### 2. Embed Page (`Embed`)

**Inputs**

* Choose Input:

  * **PNG / JPG**
  * **DICOM**
* For **PNG/JPG**:

  * Enter: Patient ID, Doctor ID, Hospital ID, Visit Date
  * These are converted into compact, two-column watermark text via `generate_watermark_image_2col`.
* For **DICOM**:

  * Image is loaded via `load_dicom_grayscale`.
  * Metadata rows are extracted via `extract_all_metadata_rows` (excluding pixel data).
  * Watermark generated via `generate_watermark_multi_col`.

**Process**

1. Show preview of base image.
2. Let user pick **Embedding Strength** (`0.01`–`0.10`, default `0.02`).
3. On **"Embed Watermark"**:

   * Call `embed_watermark_svd(base_img, wm, strength, embed_low=0.2, embed_high=0.8)`.
   * Immediately:

     * Show **Original vs Watermarked** image.
     * Display **PSNR, BER, NCC**.

**Robustness Simulator**

* For each attack (`Rotation`, `SaltPepper`, `Blur`, `Crop`):

  * Generate damaged image via `damage_image`.
  * Recover watermark via `extract_watermark_svd`.
  * Show:

    * Damaged image
    * Recovered watermark
    * BER & NCC

**Export**

* Generates:

  * `watermarked.png`
  * `key.pkl`
* Packs into `watermarked_results.zip` for download.

These outputs are used later in the **Extract** page.

---

### 3. Extract Page (`Extract`)

**Inputs**

* Watermarked image (`.png/.jpg/.jpeg`)
* Key file (`key.pkl`)

**Process**

* Convert image to grayscale.
* Load key via `pickle`.
* Call `extract_watermark_svd(img, key)`.

**Output**

* Display the **recovered watermark image**.
* Status message confirming successful extraction.

---

## 📦 Dependencies

Install the required packages (adjust if your `watermark_module` has extras):

```txt
streamlit
numpy
opencv-python
Pillow
pydicom
```

Core logic is implemented in `watermark_module.py`, which provides:

* `embed_watermark_svd`
* `extract_watermark_svd`
* `calculate_psnr`
* `calculate_ber_and_ncc`
* `damage_image`
* `load_dicom_grayscale`
* `generate_watermark_image_2col`
* `generate_watermark_multi_col`
* `extract_all_metadata_rows`

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run Home.py
```

Then open the local URL shown in your terminal.
Use the sidebar to switch between **Home**, **Embed**, and **Extract**.

---

## ⚠️ Disclaimer

Koala SVD Watermarking is a **research and educational prototype**.
It is **not** a certified medical device and should not be used as-is in production clinical workflows without proper validation, regulatory checks, and security hardening.
