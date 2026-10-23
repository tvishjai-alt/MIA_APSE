
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# APSE: NINE-IMAGE MRI ENHANCEMENT VISUALIZATION
#
# Displays:
# 1. Original MRI
# 2. Degraded MRI
# 3-9. Seven enhancement methods
#
# The benchmark's winning method is highlighted in green.
# No metrics are printed on the images.
# ============================================================


# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

BASE_DIR = (
    r"F:\VIT DOCS\FALL SEM '26\MEDICAL IMAGING ANALYSIS"
    r"\PROJECT\IXI_data\IXI_data"
)

DATA_DIR = os.path.join(BASE_DIR, "full_dataset")

RESULTS_DIR = os.path.join(
    DATA_DIR, "enhancement_results"
)

METADATA_PATH = os.path.join(
    DATA_DIR, "degradation_metadata_full.csv"
)

BENCHMARK_DIR = os.path.join(
    RESULTS_DIR, "all_methods_selection"
)

RESULTS_CSV = os.path.join(
    BENCHMARK_DIR, "all_methods_test_results.csv"
)

OUTPUT_IMAGE = os.path.join(
    RESULTS_DIR, "apse_nine_image_comparison.png"
)

# 0 = first sample, 1 = second sample, etc.
SAMPLE_INDEX = 0

METHODS = [
    "Identity",
    "Gaussian",
    "Median",
    "Bilateral",
    "ContrastStretch",
    "CLAHE",
    "UnsharpMask",
]


# ------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ------------------------------------------------------------

def find_column(df, candidates, required=True):
    """Find a column without case sensitivity."""

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.strip().lower() in lookup:
            return lookup[candidate.strip().lower()]

    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def resolve_path(path):
    """Resolve absolute or project-relative file paths."""

    path = os.path.expanduser(str(path))

    if os.path.isabs(path) and os.path.exists(path):
        return path

    candidates = [
        os.path.join(BASE_DIR, path),
        os.path.join(DATA_DIR, path),
        os.path.join(BENCHMARK_DIR, path),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not locate file:\n{path}"
    )


def load_image(path):
    """Load one saved 2D MRI slice."""

    image = np.squeeze(
        np.load(resolve_path(path))
    ).astype(np.float32)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D image; got shape {image.shape}"
        )

    if not np.all(np.isfinite(image)):
        raise ValueError(
            f"Image contains NaN or infinite values:\n{path}"
        )

    return image


def display_normalize(image):
    """
    Normalize intensities for display only.
    The underlying image and benchmark results remain unchanged.
    """

    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)

    mask = finite & (image > 0.02)

    if np.count_nonzero(mask) < 20:
        mask = finite

    values = image[mask]

    if values.size == 0:
        return np.zeros_like(image)

    low, high = np.percentile(values, [1, 99])

    if high <= low:
        low = float(np.min(values))
        high = float(np.max(values))

    if high <= low:
        return np.zeros_like(image)

    normalized = np.clip(
        (image - low) / (high - low),
        0,
        1
    )

    normalized[~finite] = 0

    return normalized


def get_candidate_path(sample_id, method):
    """Locate a saved enhancement image."""

    candidate_path = os.path.join(
        BENCHMARK_DIR,
        str(sample_id),
        "candidates",
        f"{method}.npy"
    )

    if not os.path.exists(candidate_path):
        raise FileNotFoundError(
            f"Missing output for method '{method}':\n"
            f"{candidate_path}\n\n"
            "Check that the seven-method benchmark completed."
        )

    return candidate_path


def show_mri(ax, image, title):
    """Display an MRI with a clear title."""

    ax.imshow(
        display_normalize(image),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        pad=9
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


# ------------------------------------------------------------
# 3. CHECK AND LOAD FILES
# ------------------------------------------------------------

for path in [METADATA_PATH, RESULTS_CSV]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found:\n{path}\n\n"
            "Check your folder paths and benchmark outputs."
        )

metadata = pd.read_csv(METADATA_PATH)
results = pd.read_csv(RESULTS_CSV)

print("Benchmark columns:", results.columns.tolist())
print("Metadata columns:", metadata.columns.tolist())


# ------------------------------------------------------------
# 4. IDENTIFY THE REQUIRED COLUMNS
# ------------------------------------------------------------

result_sample_col = find_column(
    results, ["SampleID", "Sample_ID", "sample_id"]
)

result_method_col = find_column(
    results, ["Method", "EnhancementMethod"]
)

ssim_col = find_column(results, ["SSIM"])
psnr_col = find_column(results, ["PSNR"])
mae_col = find_column(results, ["MAE"])

metadata_sample_col = find_column(
    metadata, ["SampleID", "Sample_ID", "sample_id"]
)

original_col = find_column(
    metadata,
    [
        "OriginalPath",
        "OriginalImagePath",
        "CleanPath",
        "ReferencePath",
        "OriginalFilePath",
        "CleanImagePath",
        "ReferenceImagePath",
    ]
)

degraded_col = find_column(
    metadata, ["DegradedPath", "DegradedImagePath"]
)

subject_col = find_column(
    metadata,
    ["Subject", "SubjectID", "Subject_ID"],
    required=False
)

degradation_col = find_column(
    metadata, ["Degradation"], required=False
)

split_col = find_column(
    metadata, ["Split"], required=False
)

results["_sample_key"] = (
    results[result_sample_col].astype(str)
)

metadata["_sample_key"] = (
    metadata[metadata_sample_col].astype(str)
)


# ------------------------------------------------------------
# 5. SELECT A SAMPLE
# ------------------------------------------------------------

unique_sample_ids = (
    results["_sample_key"]
    .drop_duplicates()
    .tolist()
)

if not unique_sample_ids:
    raise ValueError("No sample IDs found in the benchmark CSV.")

if not 0 <= SAMPLE_INDEX < len(unique_sample_ids):
    raise IndexError(
        f"SAMPLE_INDEX must be between 0 and "
        f"{len(unique_sample_ids) - 1}."
    )

sample_id = unique_sample_ids[SAMPLE_INDEX]

sample_results = results[
    results["_sample_key"] == sample_id
].copy()

metadata_matches = metadata[
    metadata["_sample_key"] == sample_id
]

if metadata_matches.empty:
    raise ValueError(
        f"Sample {sample_id} is missing from metadata."
    )

sample_metadata = metadata_matches.iloc[0]


# ------------------------------------------------------------
# 6. GET SUBJECT AND DEGRADATION LABELS
# ------------------------------------------------------------

subject = (
    str(sample_metadata[subject_col])
    if subject_col is not None
    else "Not specified"
)

degradation = (
    str(sample_metadata[degradation_col])
    if degradation_col is not None
    else "Not specified"
)

if split_col is not None:
    split_value = str(
        sample_metadata[split_col]
    ).strip().lower()

    if split_value != "test":
        print(
            f"Warning: selected sample split is '{split_value}', "
            "not 'test'."
        )


# ------------------------------------------------------------
# 7. SELECT THE BEST METHOD USING SAVED METRICS
# ------------------------------------------------------------

# Selection rule:
# 1. Highest SSIM
# 2. Highest PSNR if SSIM is tied
# 3. Lowest MAE if both are tied

sample_results = sample_results.sort_values(
    by=[ssim_col, psnr_col, mae_col],
    ascending=[False, False, True]
)

best_method = str(
    sample_results.iloc[0][result_method_col]
)

# Verify that the benchmark contains all seven methods.
available_methods = {
    str(method).strip().lower()
    for method in sample_results[result_method_col]
}

missing_methods = [
    method for method in METHODS
    if method.lower() not in available_methods
]

if missing_methods:
    raise ValueError(
        f"Benchmark results are missing methods for sample "
        f"{sample_id}: {missing_methods}"
    )

if best_method.lower() not in {
    method.lower() for method in METHODS
}:
    raise ValueError(
        f"Unknown winning method in results: {best_method}"
    )


# ------------------------------------------------------------
# 8. LOAD ORIGINAL, DEGRADED AND ENHANCED IMAGES
# ------------------------------------------------------------

original = load_image(
    sample_metadata[original_col]
)

degraded = load_image(
    sample_metadata[degraded_col]
)

if original.shape != degraded.shape:
    raise ValueError(
        f"Original shape {original.shape} differs from "
        f"degraded shape {degraded.shape}."
    )

candidate_images = {}

for method in METHODS:
    candidate_images[method] = load_image(
        get_candidate_path(sample_id, method)
    )

    if candidate_images[method].shape != original.shape:
        raise ValueError(
            f"Image shape mismatch for {method}."
        )


# Exactly nine panels in the intended order.
panels = [
    ("Original MRI", original),
    ("Degraded MRI", degraded),
]

panels.extend(
    (method, candidate_images[method])
    for method in METHODS
)

if len(panels) != 9:
    raise ValueError(
        f"Expected nine images, found {len(panels)}."
    )


# ------------------------------------------------------------
# 9. CREATE THE VISUALIZATION
# ------------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

fig, axes = plt.subplots(
    3, 3,
    figsize=(13, 14),
    facecolor="white"
)

# Separate heading and subtitle to prevent overlap.
fig.suptitle(
    f"APSE — MRI Enhancement Results\nSubject: {subject}",
    fontsize=18,
    fontweight="bold",
    y=0.985,
    linespacing=1.5
)

fig.text(
    0.5,
    0.918,
    f"Degradation type: {degradation}",
    ha="center",
    va="center",
    fontsize=11,
    color="dimgray"
)


# ------------------------------------------------------------
# 10. DISPLAY NINE IMAGES AND HIGHLIGHT THE WINNER
# ------------------------------------------------------------

for ax, (title, image) in zip(axes.flat, panels):

    is_best_method = (
        title.lower() == best_method.strip().lower()
        and title.lower() not in {
            "original mri",
            "degraded mri"
        }
    )

    if is_best_method:
        display_title = f"{title}\n★ BEST METHOD"
    else:
        display_title = title

    show_mri(
        ax,
        image,
        display_title
    )

    if is_best_method:
        # Thick green border around the winning enhancement.
        border = Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            edgecolor="green",
            linewidth=5,
            clip_on=False,
            zorder=10
        )

        ax.add_patch(border)

        ax.title.set_color("darkgreen")


# ------------------------------------------------------------
# 11. SPACING AND SAVE
# ------------------------------------------------------------

fig.subplots_adjust(
    left=0.035,
    right=0.965,
    top=0.88,
    bottom=0.035,
    wspace=0.08,
    hspace=0.24
)

os.makedirs(RESULTS_DIR, exist_ok=True)

fig.savefig(
    OUTPUT_IMAGE,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)


# ------------------------------------------------------------
# 12. SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("APSE VISUALIZATION SAVED SUCCESSFULLY")
print("=" * 65)

print(f"Subject: {subject}")
print(f"Sample ID: {sample_id}")
print(f"Degradation type: {degradation}")
print(f"Best reference-based enhancement: {best_method}")
print(f"Output image: {OUTPUT_IMAGE}")

print("\nDisplayed panels:")
for index, (title, _) in enumerate(panels, start=1):
    print(f"{index}. {title}")

print(
    "\nThe winning method is highlighted in green. "
    "No MAE, PSNR or SSIM values are displayed on the figure."
)

print(
    "\nImportant: the winner is selected using the saved clean-reference "
    "benchmark metrics. This is an experimental comparison, not a "
    "clinical selector that can independently choose a method for "
    "an unknown image."
)
