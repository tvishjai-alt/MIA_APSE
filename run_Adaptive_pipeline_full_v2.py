
import os
import joblib
import numpy as np
import pandas as pd

from scipy.ndimage import (
    gaussian_filter,
    median_filter,
    gaussian_laplace,
    sobel,
    uniform_filter
)
from skimage import exposure, restoration
from skimage.filters import unsharp_mask
from skimage.metrics import (
    structural_similarity,
    peak_signal_noise_ratio
)

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FULL_DIR = os.path.join(BASE_DIR, "full_dataset")
RESULTS_DIR = os.path.join(FULL_DIR, "enhancement_results")

MODEL_PATH = os.path.join(
    RESULTS_DIR, "apse_degradation_classifier_full.joblib"
)

METADATA_PATH = os.path.join(
    FULL_DIR, "degradation_metadata_full.csv"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR, "all_methods_selection"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# First, test one sample. After verification, set to None
# to process every row in the held-out Test split.
SAMPLE_ROW = None
# ============================================================
# 2. PATH RESOLUTION
# ============================================================

def resolve_path(path):
    if pd.isna(path):
        raise ValueError("An image path is missing.")

    path = os.path.expanduser(str(path))

    candidates = [
        path,
        os.path.join(BASE_DIR, path),
        os.path.join(FULL_DIR, path)
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        f"Cannot locate image: {path}\n"
        "Check the path stored in degradation_metadata_full.csv."
    )


def find_column(dataframe, possible_names, description):
    for name in possible_names:
        if name in dataframe.columns:
            return name

    raise KeyError(
        f"Could not identify the {description} column.\n"
        f"Tried: {possible_names}\n"
        f"Available columns: {list(dataframe.columns)}"
    )


# ============================================================
# 3. LOAD CLASSIFIER AND METADATA
# ============================================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.isfile(METADATA_PATH):
    raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

saved = joblib.load(MODEL_PATH)


def find_predictor(obj):
    """Find a fitted estimator, including inside nested dictionaries."""
    if hasattr(obj, "predict") and callable(obj.predict):
        return obj

    if isinstance(obj, dict):
        # Check common estimator keys first.
        preferred_keys = [
            "model", "classifier", "estimator",
            "pipeline", "best_model"
        ]

        for key in preferred_keys:
            if key in obj:
                try:
                    return find_predictor(obj[key])
                except (TypeError, ValueError):
                    pass

        for value in obj.values():
            try:
                return find_predictor(value)
            except (TypeError, ValueError):
                pass

    raise TypeError(
        "No fitted estimator with a predict() method was found "
        "inside the saved joblib artifact."
    )


model = find_predictor(saved)

if not isinstance(saved, dict):
    raise TypeError(
        "Expected the saved artifact to contain feature_columns."
    )

if "feature_columns" not in saved:
    raise KeyError(
        "The saved artifact has no 'feature_columns' entry."
    )

FEATURE_COLUMNS = list(saved["feature_columns"])

metadata = pd.read_csv(METADATA_PATH)

degraded_col = find_column(
    metadata,
    ["DegradedPath"],
    "degraded image path"
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
        "ReferenceImagePath"
    ],
    "clean reference image path"
)

sample_id_col = find_column(
    metadata,
    ["SampleID", "SampleId", "ID"],
    "sample ID"
)

split_col = find_column(
    metadata,
    ["Split", "DatasetSplit", "Set"],
    "dataset split"
)

degradation_col = find_column(
    metadata,
    ["Degradation", "ActualDegradation"],
    "degradation label"
)

# Restrict evaluation to held-out subjects.
test_metadata = metadata[
    metadata[split_col].astype(str).str.strip().str.lower()
    == "test"
].copy()

if test_metadata.empty:
    raise ValueError(
        "No rows labelled 'Test' were found. "
        "Inspect the Split column and update the filter if needed."
    )

if SAMPLE_ROW is not None:
    if SAMPLE_ROW < 0 or SAMPLE_ROW >= len(test_metadata):
        raise IndexError(
            f"SAMPLE_ROW must be between 0 and "
            f"{len(test_metadata) - 1}."
        )

    rows_to_process = test_metadata.iloc[[SAMPLE_ROW]]
else:
    rows_to_process = test_metadata

# ============================================================
# 4. IMAGE QUALITY FEATURES
# Keep consistent with classifier training.
# ============================================================

def extract_quality_features(image):
    image = np.asarray(image, dtype=np.float64)

    mask = image > 0.02

    if np.sum(mask) < 10:
        mask = np.ones_like(image, dtype=bool)

    pixels = image[mask]

    histogram, _ = np.histogram(
        pixels, bins=64, range=(0, 1)
    )

    probabilities = histogram / max(histogram.sum(), 1)
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    laplacian = gaussian_laplace(image, sigma=1)

    gradient_x = sobel(image, axis=0)
    gradient_y = sobel(image, axis=1)

    gradient_magnitude = np.sqrt(
        gradient_x**2 + gradient_y**2
    )

    residual = image - uniform_filter(image, size=3)
    masked_residual = residual[mask]
    residual_median = np.median(masked_residual)

    feature_row = {
        "MeanIntensity": np.mean(pixels),
        "IntensityStd": np.std(pixels),
        "IntensityRange": (
            np.percentile(pixels, 99)
            - np.percentile(pixels, 1)
        ),
        "HistogramEntropy": entropy,
        "LaplacianVariance": np.var(laplacian[mask]),
        "MeanGradientMagnitude": np.mean(
            gradient_magnitude[mask]
        ),
        "EstimatedNoise": 1.4826 * np.median(
            np.abs(masked_residual - residual_median)
        )
    }

    return pd.DataFrame(
        [feature_row], columns=FEATURE_COLUMNS
    )


# ============================================================
# 5. ALL SEVEN ENHANCEMENT METHODS
# Use the same parameters as your benchmark.
# ============================================================

METHODS = [
    "Identity",
    "Gaussian",
    "Median",
    "Bilateral",
    "ContrastStretch",
    "CLAHE",
    "UnsharpMask"
]


def apply_enhancement(image, method):
    image = np.asarray(image, dtype=np.float64)

    if method == "Identity":
        enhanced = image.copy()

    elif method == "Gaussian":
        enhanced = gaussian_filter(image, sigma=0.8)

    elif method == "Median":
        enhanced = median_filter(image, size=3)

    elif method == "Bilateral":
        enhanced = restoration.denoise_bilateral(
            image,
            sigma_color=0.08,
            sigma_spatial=2,
            channel_axis=None
        )

    elif method == "ContrastStretch":
        mask = image > 0.02

        if np.sum(mask) < 10:
            mask = np.ones_like(image, dtype=bool)

        lower, upper = np.percentile(
            image[mask], [1, 99]
        )

        if upper > lower:
            enhanced = exposure.rescale_intensity(
                image,
                in_range=(lower, upper),
                out_range=(0, 1)
            )
            enhanced[~mask] = 0
        else:
            enhanced = image.copy()

    elif method == "CLAHE":
        enhanced = exposure.equalize_adapthist(
            image,
            kernel_size=32,
            clip_limit=0.01
        )

    elif method == "UnsharpMask":
        enhanced = unsharp_mask(
            image,
            radius=1,
            amount=0.5,
            preserve_range=True
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return np.clip(enhanced, 0, 1)


# ============================================================
# 6. REFERENCE-BASED EVALUATION
# ============================================================

def evaluate_candidate(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Image shape mismatch: reference {reference.shape}, "
            f"candidate {candidate.shape}"
        )

    if reference.ndim != 2:
        raise ValueError(
            "Expected 2D grayscale MRI slices."
        )

    mae = float(np.mean(np.abs(reference - candidate)))

    psnr = float(
        peak_signal_noise_ratio(
            reference, candidate, data_range=1.0
        )
    )

    ssim = float(
        structural_similarity(
            reference, candidate, data_range=1.0
        )
    )

    return {
        "MAE": mae,
        "PSNR": psnr,
        "SSIM": ssim
    }


# ============================================================
# 7. PROCESS EACH HELD-OUT TEST SAMPLE
# ============================================================

all_results = []

for row_number, (_, sample) in enumerate(
    rows_to_process.iterrows(), start=1
):
    sample_id = str(sample[sample_id_col])

    print(
        f"\nProcessing {row_number}/"
        f"{len(rows_to_process)}: {sample_id}"
    )

    input_path = resolve_path(sample[degraded_col])
    reference_path = resolve_path(sample[original_col])

    degraded = np.load(input_path).astype(np.float64)
    reference = np.load(reference_path).astype(np.float64)

    if not np.isfinite(degraded).all():
        raise ValueError(f"Non-finite input values in {sample_id}")

    if not np.isfinite(reference).all():
        raise ValueError(f"Non-finite reference values in {sample_id}")

    if degraded.shape != reference.shape:
        raise ValueError(
            f"{sample_id}: input and reference dimensions differ."
        )

    # This script assumes the dataset is already normalized to [0, 1].
    if (
        degraded.min() < -1e-6
        or degraded.max() > 1.000001
        or reference.min() < -1e-6
        or reference.max() > 1.000001
    ):
        raise ValueError(
            f"{sample_id}: images are not normalized to [0, 1]. "
            "Inspect the dataset normalization before calculating "
            "metrics; do not silently rescale the images."
        )

    degraded = np.clip(degraded, 0, 1)
    reference = np.clip(reference, 0, 1)

    # Classifier prediction is retained for analysis only.
    features = extract_quality_features(degraded)
    predicted_degradation = str(model.predict(features)[0])

    sample_dir = os.path.join(OUTPUT_DIR, sample_id)
    candidates_dir = os.path.join(sample_dir, "candidates")
    os.makedirs(candidates_dir, exist_ok=True)

    # Save the degraded input for reference.
    np.save(
        os.path.join(sample_dir, "degraded_input.npy"),
        degraded
    )

    method_results = []

    # Apply and evaluate every method.
    for method in METHODS:
        enhanced = apply_enhancement(degraded, method)

        metrics = evaluate_candidate(reference, enhanced)

        candidate_path = os.path.join(
            candidates_dir, f"{method}.npy"
        )
        np.save(candidate_path, enhanced)

        method_results.append({
            "SampleID": sample_id,
            "Split": str(sample[split_col]),
            "ActualDegradation": str(sample[degradation_col]),
            "PredictedDegradation": predicted_degradation,
            "Method": method,
            **metrics,
            "CandidatePath": candidate_path
        })

    # Selection rule: highest SSIM, then highest PSNR,
    # then lowest MAE. This is a research benchmark rule,
    # NOT a clinical safety criterion.
    ranked = sorted(
        method_results,
        key=lambda r: (
            r["SSIM"],
            r["PSNR"],
            -r["MAE"]
        ),
        reverse=True
    )

    winner = ranked[0]

    # Save the selected candidate separately.
    best_image = np.load(winner["CandidatePath"])

    best_path = os.path.join(
        sample_dir, "best_reference_based_candidate.npy"
    )
    np.save(best_path, best_image)

    # Save the per-method metric table.
    metrics_path = os.path.join(
        sample_dir, "all_methods_metrics.csv"
    )
    pd.DataFrame(method_results).sort_values(
        "SSIM", ascending=False
    ).to_csv(metrics_path, index=False)

    # Record all method metrics plus the chosen winner.
    for result in method_results:
        result["SelectedWinnerByReferenceMetrics"] = (
            winner["Method"]
        )
        result["BestCandidatePath"] = best_path
        result["ReferencePath"] = reference_path
        result["InputPath"] = input_path

    all_results.extend(method_results)

    print("Actual degradation:", sample[degradation_col])
    print("Predicted degradation:", predicted_degradation)
    print("Winner by reference metrics:", winner["Method"])
    print("Winner SSIM:", round(winner["SSIM"], 6))
    print("Winner PSNR:", round(winner["PSNR"], 6))
    print("Winner MAE:", round(winner["MAE"], 6))
    print("Saved metrics:", metrics_path)


# ============================================================
# 8. COMBINED RESULTS
# ============================================================

if all_results:
    combined_path = os.path.join(
        OUTPUT_DIR, "all_methods_test_results.csv"
    )

    pd.DataFrame(all_results).to_csv(
        combined_path, index=False
    )

    print("\n" + "=" * 60)
    print("APSE: ALL-SEVEN-METHOD REFERENCE BENCHMARK")
    print("=" * 60)
    print("Test samples processed:", len(rows_to_process))
    print("Methods evaluated per sample:", len(METHODS))
    print("Combined results:", combined_path)
    print("\nImportant:")
    print(
        "The selected winner is based on reference-image metrics. "
        "This is an experimental oracle comparison, not clinical "
        "validation or a reference-free deployment strategy."
    )
