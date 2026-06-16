"""
Phase 2 Configuration — AI Training Pipeline
Ship Battery ISC Detection under Vibration Noise
"""
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TAU_SWEEP_FILE = DATA_DIR / "tau_sweep_results.mat"
EVAL_TAU_FILE = DATA_DIR / "eval_tau_results.mat"
VIBRATION_FILE = DATA_DIR / "vibration_data_all.mat"
RESULTS_DIR = PROJECT_ROOT / "ai" / "results"

# === Simulation Parameters ===
T_ONSET = 200.0          # ISC onset time (s)
R_INIT = 500.0            # Initial ISC resistance (Ohm)
R_FINAL = 5.0             # Final ISC resistance (Ohm)
I_LOAD = -2.9             # Constant discharge current (A)
V_MODULE_NOM = 28.8       # Module nominal voltage (V)

# === Data Selection (Simplification Decisions 2026-03-30) ===
MODEL_ID = 2              # Module (8S24P) only
SCENARIOS = [1, 2, 4]     # NoVib, MIL-STD, MSS+MIL-STD Head (Head 1-direction only)
SCENARIO_NAMES = {1: "NoVib", 2: "MIL-STD", 4: "MSS-Head"}
TAU_VALUES = [50, 300, 1800, 3600]
KR_VALUES = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]  # Ohm/g
KR_LABELS = [0.05, 0.1, 0.2, 0.5, 1.0]       # mOhm/g (display)

# === Preprocessing ===
RESAMPLE_HZ = 100          # Target sampling rate (100Hz captures MIL-STD 4-33Hz vibration)
T_AMB_K = 298.15          # Ambient temperature (K) for C conversion

# === Windowing ===
WINDOW_SIZE = 1000          # 10 seconds at 100Hz
MIN_STRIDE = 100            # Minimum stride (1 second at 100Hz)
TARGET_WINDOWS_PER_RUN = 560  # Target ~560 windows/run (adaptive stride)

# === Labeling ===
R_LABEL_THRESHOLD = 50.0   # ISC label when R_ISC <= 50 Ohm

# === Detection Deadline ===
R_DEADLINE = 10.0          # Deadline when R_ISC <= 10 Ohm (I_ISC >= I_load)
import math
DEADLINE_FACTOR = math.log((R_INIT - R_FINAL) / (R_DEADLINE - R_FINAL))  # 4.595

# === Train/Val/Test Split (tau-based, all k_R in every split) ===
TRAIN_TAUS = [50, 300, 3600]  # 3 ISC speeds for training
TEST_TAU = 1800               # Unseen ISC speed for test
EVAL_TAUS = [100, 1000]       # Additional unseen τ for detection delay evaluation
VAL_RATIO = 0.2               # 20% of train pool for val (stratified by scenario×tau)

# === Model Architecture ===
# Ablation flags:
#   ABLATION_V_ONLY      : V-only retraining for input-observability decomposition.
#                          True -> ModernTCN/LITE trained on V channel alone (1-channel input).
#   ABLATION_NOVIB_TRAIN : NoVib-only training distribution for vibration-distribution decomposition.
#                          True -> train/val sets restricted to scenario==1 (NoVib); test/eval cover
#                          all scenarios. Measures whether AI's vibration robustness stems from the
#                          vibration-aware training pipeline or from intrinsic model properties.
# Flags are independent; both False = baseline V+T + Vib-aware train (production setting).
ABLATION_V_ONLY = True
ABLATION_NOVIB_TRAIN = True
INPUT_CHANNELS = 1 if ABLATION_V_ONLY else 2  # [V] or [V, T]

# ModernTCN (ICLR 2024 Spotlight) — lightweight variant
MODERN_TCN_D_MODEL = 32    # Internal channel dimension
MODERN_TCN_STEM_KERNEL = 51  # Stem Conv kernel (0.51s, ~2 cycles of 4Hz vibration)
MODERN_TCN_KERNEL = 121    # DWConv kernel size (1.21s per block)
MODERN_TCN_BLOCKS = 3      # Number of ModernTCN blocks
MODERN_TCN_FFN_RATIO = 2   # ConvFFN expansion ratio (32→64→32)
# RF = 51 + 3*(121-1) = 411 samples = 4.11s

# LITE (IEEE DSAA 2023) — simplified, no custom filters
LITE_INCEPTION_KERNELS = [21, 41, 81, 161, 321]  # Multi-scale, odd kernels for exact same-padding
LITE_INCEPTION_FILTERS = 8                    # Filters per branch (→40 total)
LITE_DWS_CHANNELS = 32                        # DWSConv output channels
LITE_DWS1_KERNEL = 51                         # DWSConv1 kernel (0.51s)
LITE_DWS2_KERNEL = 41                         # DWSConv2 kernel (0.41s)
# RF = 321 + (51-1) + (41-1) = 411 samples = 4.11s

# === EKF Parameters ===
EKF_WINDOW_SEC = 10.0      # Residual averaging window (seconds)
EKF_SIGMA_MULT = 3.0       # Detection threshold multiplier (3σ)
EKF_Q_SOC = 1e-5           # Process noise std for SOC
EKF_Q_VRC = 1e-4           # Process noise std for V_RC1

# Module (8S24P) ECM parameters — extracted from Simscape simulation data
# OCV table: derived from NoVib tau=3600 run (OCV ≈ V + |I|*(R0+R1))
# Note: build_simscape_model.m OCV was wrong by ~1.2V; these are corrected values
EKF_SOC_VEC = [0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80,
               0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90]
EKF_OCV_MODULE = [31.211, 31.397, 31.571, 31.734, 31.888, 32.033, 32.172, 32.305,
                  32.437, 32.576, 32.711, 32.843, 32.973, 33.100, 33.226, 33.351,
                  33.475, 33.617]  # V (from simulation)
# R0, R1, tau1: from Simscape model params (verified correct at module level)
EKF_R0_MODULE = [2.333, 2.067, 1.933, 1.867, 1.933, 2.000, 2.067]  # mOhm
EKF_R1_MODULE = [6.667, 6.000, 5.000, 4.000, 5.000, 6.000, 6.667]  # mOhm
EKF_R0_SOC_VEC = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]  # SOC points for R0/R1/tau1
EKF_TAU1 = [30, 25, 20, 18, 20, 25, 30]                            # s
EKF_AH_MODULE = 69.6       # Ah (cell 2.9 × 24 parallel)
EKF_SOC_INIT = 0.9         # Initial SOC

# === Training ===
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 100
PATIENCE = 20              # Early stopping patience
RANDOM_SEED = 42
