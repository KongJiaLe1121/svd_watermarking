# 🐨 Koala SVD Watermarking

A simple, end-to-end demo app for robust, **imperceptible medical image watermarking** using Singular Value Decomposition (SVD).

Koala focuses on a realistic medical setting: preserving diagnostic image quality while securely embedding patient-related metadata that can be verified and recovered later—even after common image manipulations.

---

## 🌐 What This App Does

Koala provides a **3-page Streamlit interface**:

1. **Home** – Concept, objectives, and system overview
2. **Embed** – Upload an image, generate watermark content, embed via SVD, test robustness, and download results
3. **Extract** – Given a watermarked image + key, recover the watermark

This setup is suitable for:

* Research prototypes
* Classroom demos
* Experimental validation of watermark robustness for medical or sensitive images

---

## 🎯 Key Objectives

The system is designed around four measurable goals:

1. **Imperceptibility**

   * Target: **PSNR ≥ 50 dB** between original and watermarked image
   * Rationale: Maintain diagnostic image fidelity.

2. **Accurate Extraction**

   * Target: **BER < 0.03**, **NCC ≥ 0.70** under ideal (no-attack) conditions
   * Rationale: Ensure embedded data is reliably recoverable.

3. **Robustness**

   * Target under attacks: **BER < 0.05**, **NCC ≥ 0.60**
   * Attacks tested in-app: rotation, salt & pepper noise, blur, cropping.

4. **Storage Efficiency**

   * Support for converting heavy **DICOM** into **PNG/JPEG** while preserving watermark and visual quality.

---

## 🧠 How It Works (High Level)

Koala uses **Singular Value Decomposition (SVD)** on the host image:

* **Original (naive) approach**: Spread watermark across all singular values
  ➜ Protects data, but hurts PSNR (~48 dB).

* **Koala approach**:
  Embed watermark **only into the middle singular values** (`embed_low=0.2`, `embed_high=0.8` region by default):

  * Top singular values control global structure/contrast → avoid touching.
  * Bottom singular values are noise-sensitive → avoid for stability.
  * Middle band = sweet spot for **imperceptibility + robustness**.

The app then:

* Computes **PSNR** to measure distortion.
* Computes **BER & NCC** between original watermark and recovered watermark.
* Simulates common attacks and reports post-attack BER/NCC.

---

## 🧩 Main Components

### 1. `Home` Page

Explains:

* The motivation (medical image integrity, secure metadata, robustness).
* The improved embedding strategy (middle SV focus).
* A quick view of system features and navigation.

No inputs; purely informational.

---

### 2. `Embed` Page

**Flow:**

1. Choose **Input Format**:

   * `PNG / JPG` → uploaded as grayscale.
   * `DICOM` → loaded and converted using `load_dicom_grayscale()`.

2. For `PNG / JPG`:

   * Input: **Patient ID, Doctor ID, Hospital ID, Visit Date**
   * These are combined into 2-column rows (e.g. `P:xxx`, `D:xxx`, etc.).
   * Watermark generated via `generate_watermark_image_2col()`.

3. For `DICOM`:

   * Metadata rows auto-extracted via `extract_all_metadata_rows(ds)` (excluding pixel data).
   * Watermark generated via `generate_watermark_multi_col()`.

4. Preview:

   * Original image
   * Generated watermark image

5. Choose **Embedding Strength**:

   * Slider: `0.01` → `0.10` (default `0.02`)

6. On **"Embed Watermark"**:

   * Calls `embed_watermark_svd(base_img, wm, strength, embed_low=0.2, embed_high=0.8)`
   * Immediately runs extraction on the watermarked image:

     * Shows **Original vs Watermarked**
     * Displays **PSNR, BER, NCC**

7. **Robustness Tests**:

   * For each attack (`Rotation`, `SaltPepper`, `Blur`, `Crop`):

     * Generates damaged image via `damage_image`
     * Recovers watermark with `extract_watermark_svd`
     * Displays:

       * Damaged image
       * Recovered watermark
       * BER & NCC

8. **Download**:

   * Bundles:

     * `watermarked.png`
     * `key.pkl`
   * into `watermarked_results.zip` for offline use and use in Extract page.

---

### 3. `Extract` Page

**Flow:**

1. Upload:

   * Watermarked image (`png/jpg/jpeg`)
   * Corresponding key file (`.pkl`) from the Embed step

2. The app:

   * Loads the image as grayscale
   * Loads key via `pickle`
   * Calls `extract_watermark_svd(img, key)`

3. Output:

   * Displays the **recovered watermark image**
   * Confirms successful extraction

---

## 📦 Dependencies

Core Python libraries (typical):

* `streamlit`
* `numpy`
* `opencv-python` (cv2)
* `Pillow`
* `pydicom` (for DICOM support)
* Plus your custom:

  * `watermark_module.py`:

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

## ▶️ How to Run

1. Create a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. From the project root:

```bash
streamlit run Home.py
```

4. Use the sidebar to navigate:

   * **Koala SVD Watermarking – Home**
   * **Koala SVD Watermarking – Embed**
   * **Koala SVD Watermarking – Extract**

---

## ✅ Notes & Extension Ideas

* The current app is a **research/demo tool**, not a certified medical product.
* You can:

  * Log all metrics (PSNR/BER/NCC) for batch experiments.
  * Extend attacks (JPEG compression, scaling, contrast changes).
  * Add authentication or audit logs for clinical scenarios.
  * Integrate with PACS/DICOM servers in future work.
