import cv2
import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian
import matplotlib.pyplot as plt
from datetime import datetime
import json

# -------------------------------
#  Utility Functions
# -------------------------------

def is_valid_yyyy_mm_dd(date_str: str) -> bool:
    """Return True if the string matches the YYYY-MM-DD pattern."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def generate_watermark_image_2col(rows, font_scale: float = 1.0,
                                  pad: int = 5, line_gap: int = 5) -> np.ndarray:
    """
    Build a 2-column text watermark.
    rows – iterable of (left_text, right_text) tuples
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes_l = [cv2.getTextSize(l, font, font_scale, 2)[0] for l, _ in rows]
    sizes_r = [cv2.getTextSize(r, font, font_scale, 2)[0] for _, r in rows]

    W = max(sl[0] + sr[0] + pad * 3 for sl, sr in zip(sizes_l, sizes_r))
    line_h = max(sl[1] for sl in sizes_l)
    H = line_h * len(rows) + line_gap * (len(rows) - 1) + pad * 2

    img = np.zeros((H, W), np.uint8)
    y = pad + line_h
    for (lt, rt), sl, sr in zip(rows, sizes_l, sizes_r):
        cv2.putText(img, lt, (pad, y), font, font_scale, 255, 2, cv2.LINE_AA)
        x_rt = W - pad - sr[0]
        cv2.putText(img, rt, (x_rt, y), font, font_scale, 255, 2, cv2.LINE_AA)
        y += line_h + line_gap
    return img

def generate_watermark_multi_col(rows, cols=4,
                                 font_scale=0.5, pad=5, line_gap=5):
    """
    Split 'rows' evenly into `cols` columns, then render side by side.
    Each column is drawn with generate_watermark_image_2col logic.
    """
    # chunk rows into sublists
    chunk = (len(rows) + cols - 1)//cols
    sublists = [rows[i*chunk:(i+1)*chunk] for i in range(cols)]
    # generate each column as its own image
    imgs = [generate_watermark_image_2col(sl, font_scale, pad, line_gap)
            for sl in sublists]
    # pad to same height
    H = max(im.shape[0] for im in imgs)
    es = []
    for im in imgs:
        h, w = im.shape
        if h < H:
            buf = np.zeros((H, w), np.uint8)
            buf[:h, :] = im
            im = buf
        es.append(im)
    # concatenate horizontally
    return np.hstack(es)

# -------------------------------
# DICOM Handling Functions
# -------------------------------

def load_dicom_grayscale(path):
    """
    Load a DICOM file, decompress if needed, normalize pixel data to 0-255,
    and return as uint8 grayscale image along with the dataset.
    """
    ds = pydicom.dcmread(path)
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        ds.decompress()
    img = ds.pixel_array.astype(np.float32)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    gray = img_norm.astype(np.uint8)
    if gray.ndim == 3:
        if gray.shape[2] == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = gray[0]
    return gray, ds

def extract_all_metadata_rows(ds):
    """
    Return a list of (key, value) tuples for every non-private, non‐PixelData element in the DICOM.
    Dates (DA) are converted to YYYY-MM-DD.
    """
    rows = []
    for elem in ds:
        # skip private or empty or huge pixel data
        if elem.tag.is_private:
            continue
        if elem.keyword == "PixelData":
            continue
        if elem.value is None:
            continue

        key = elem.keyword or str(elem.tag)
        val = elem.value

        # skip sequences
        if elem.VR == "SQ":
            continue

        # convert DICOM date (DA)
        if elem.VR == "DA":
            try:
                val = datetime.strptime(val, "%Y%m%d").strftime("%Y-%m-%d")
            except Exception:
                pass

        rows.append((key, str(val)))
    return rows

# -------------------------------
# Watermarking Functions (Partial SVD)
# -------------------------------

def embed_watermark_svd(image, watermark, strength=0.01,
                        embed_low=0.3, embed_high=0.7):
    """
    Embed a watermark into a 2D image via partial SVD band modification.
    Returns the watermarked image and parameters for extraction.
    """
    A = image.astype(np.float32)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    S = np.diag(s)
    n = S.shape[0]
    i1 = int(embed_low * n)
    i2 = int(embed_high * n)
    if i2 <= i1:
        i1, i2 = 0, n
    block = i2 - i1
    wm_resized = cv2.resize(watermark.astype(np.float32), (block, block), interpolation=cv2.INTER_NEAREST)
    S_block = S.copy()
    S_block[i1:i2, i1:i2] += strength * wm_resized
    Uw, Sw, Vtw = np.linalg.svd(S_block, full_matrices=False)
    watermarked = (U @ np.diag(Sw) @ Vt).clip(0,255).astype(np.uint8)
    params = {"Uw":Uw, "Vtw":Vtw, "original_S":S,
              "i1":i1, "i2":i2, "strength":strength,
              "wm_shape":watermark.shape}

    return watermarked, params

def extract_watermark_svd(wm_img, params):
    """
    Extract watermark from a watermarked 2D image using stored SVD parameters.
    """
    Uw = params["Uw"]
    Vtw = params["Vtw"]
    S0 = params["original_S"]
    i1, i2 = params["i1"], params["i2"]
    strength = params["strength"]
    h, w = params["wm_shape"]
    U2, s2, Vt2 = np.linalg.svd(wm_img.astype(np.float32), full_matrices=False)
    D = Uw @ np.diag(s2) @ Vtw
    W_sub = (D[i1:i2, i1:i2] - S0[i1:i2, i1:i2]) / strength
    recovered = cv2.resize(W_sub, (w, h), interpolation=cv2.INTER_NEAREST)
    return recovered.clip(0,255).astype(np.uint8)

# -------------------------------
# Performance Metrics
# -------------------------------

def calculate_psnr(original_img, processed_img):
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    """
    mse = np.mean((original_img.astype(np.float32) - processed_img.astype(np.float32))**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ber_and_ncc(original_wm, extracted_wm):
    """
    Compute Bit Error Rate (BER) and Normalized Cross-Correlation (NCC) between two watermark images.
    BER is pixel-wise error rate after binarization, NCC measures similarity of grayscale patterns.
    """
    # BER calculation
    orig_bin = (original_wm > 127).astype(np.uint8)
    ext_bin = (extracted_wm > 127).astype(np.uint8)
    total = orig_bin.size
    errors = np.sum(orig_bin != ext_bin)
    ber = errors / total
    # NCC calculation
    orig_f = original_wm.astype(np.float32)
    ext_f = extracted_wm.astype(np.float32)
    orig_mean = orig_f.mean()
    ext_mean = ext_f.mean()
    numerator = np.sum((orig_f - orig_mean) * (ext_f - ext_mean))
    denom = np.sqrt(np.sum((orig_f - orig_mean)**2) * np.sum((ext_f - ext_mean)**2))
    ncc = numerator / denom if denom != 0 else 0
    return ber, ncc

# -------------------------------
# Damage Simulation Functions
# -------------------------------

def rotate_img(image, angle):
    """Rotate image by angle degrees."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))


def add_salt_and_pepper(image, salt_prob=0.02, pepper_prob=0.02):
    """Add salt-and-pepper noise."""
    out = image.copy()
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 255
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return out


def blur_img(image, ksize=(5,5)):
    """Apply Gaussian blur."""
    return cv2.GaussianBlur(image, ksize, 0)


def crop_and_resize(image, ratio=0.1):
    """Crop edges by ratio and resize back."""
    h, w = image.shape
    ch, cw = int(h*ratio), int(w*ratio)
    cropped = image[ch:h-ch, cw:w-cw]
    return cv2.resize(cropped, (w, h))


def damage_image(image, mode):
    """Apply a damage mode: "Rotation","SaltPepper","Blur","Crop"."""
    if mode == "Rotation": return rotate_img(image, 15)
    if mode == "SaltPepper": return add_salt_and_pepper(image)
    if mode == "Blur": return blur_img(image)
    if mode == "Crop": return crop_and_resize(image)
    raise ValueError("Invalid damage mode")

# -------------------------------
# High-level DICOM APIs
# -------------------------------

def embed_watermark_dicom(input_dcm, output_dcm, watermark,
                          strength=0.01, embed_low=0.3, embed_high=0.7):
    """
    Embed SVD watermark into a DICOM file and save result.
    Returns extraction params.
    """
    img, ds = load_dicom_grayscale(input_dcm)
    watermarked, params = embed_watermark_svd(img, watermark,
                                              strength, embed_low, embed_high)
    save_dicom_from_grayscale(ds, watermarked, output_dcm)
    return params


def extract_watermark_dicom(input_dcm, params, out_png):
    """
    Extract watermark from watermarked DICOM and save as PNG."""
    img, _ = load_dicom_grayscale(input_dcm)
    wm = extract_watermark_svd(img, params)
    cv2.imwrite(out_png, wm)
    return wm

# -------------------------------
# Plotting Utility
# -------------------------------
def plot_damage_tests(original, watermarked, wm_original,
                      params, modes=["Rotation","SaltPepper","Blur","Crop"]):
    """
    Run and plot watermark recovery after various damage modes."""
    psnr = calculate_psnr(original, watermarked)
    rec = extract_watermark_svd(watermarked, params)
    ber, ncc = calculate_ber_and_ncc(wm_original, rec)
    fig, axes = plt.subplots(len(modes)+1, 3, figsize=(12,4*(len(modes)+1)))
    # No damage row
    axes[0,0].imshow(original, cmap='gray'); axes[0,0].set_title('Original')
    axes[0,1].imshow(watermarked, cmap='gray'); axes[0,1].set_title('Watermarked')
    axes[0,2].text(0.1,0.5,f"PSNR:{psnr:.2f}\nBER:{ber:.3f}\nNCC:{ncc:.3f}",fontsize=12)
    axes[0,2].axis('off')
    for i, m in enumerate(modes,1):
        dmg = damage_image(watermarked, m)
        rec_dmg = extract_watermark_svd(dmg, params)
        ber_d, ncc_d = calculate_ber_and_ncc(wm_original, rec_dmg)
        axes[i,0].imshow(dmg, cmap='gray'); axes[i,0].set_title(f'{m} Damaged')
        axes[i,1].imshow(rec_dmg, cmap='gray'); axes[i,1].set_title('Recovered WM')
        axes[i,2].text(0.1,0.5,f"BER:{ber_d:.3f}\nNCC:{ncc_d:.3f}",fontsize=12)
        axes[i,2].axis('off')
    plt.tight_layout(); plt.show()