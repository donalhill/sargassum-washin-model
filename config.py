"""Domain bounds, grid specification, and model default parameters."""

# Spatial domain (degrees) — Barbados + upstream ocean
LON_MIN, LON_MAX = -61.5, -57.5
LAT_MIN, LAT_MAX = 11.5, 14.5

# Grid resolution (forcing fields)
DLON = 0.075  # ~8 km
DLAT = 0.075

# Temporal
DT_FORCING_HOURS = 6        # forcing field cadence
N_DAYS = 2                  # forecast horizon
N_STEPS = N_DAYS * 24 // DT_FORCING_HOURS  # 56 forcing timesteps
DT_SUBSTEP_HOURS = 1        # RK4 integration substep

# Model defaults
ALPHA_DEFAULT = 0.02         # leeway coefficient (wind drag on sargassum)

# Uniform random seeding (Monte Carlo prior)
N_PARTICLES = 100_000       # total particles — uniform random in buffer zone
BUFFER_KM = 100             # radial buffer around island centre
SOURCE_BIN_DEG = 0.04       # binning resolution for source heatmap display

# Stochastic perturbation
ALPHA_HALF_RANGE = 0.005     # α ∈ [α ± 0.005]
CURRENT_NOISE_STD = 0.03     # m/s, AR(1) innovation
WIND_NOISE_STD = 0.5         # m/s, AR(1) innovation
AR1_DECAY = 0.9              # autocorrelation for noise

# Coastal proximity
COAST_BUFFER_DEG = 0.004     # ~0.45 km hit-detection buffer

# Smoothing
RISK_SMOOTH_SIGMA = 2        # segments, for gaussian_filter1d

# Random seed for reproducibility
GLOBAL_SEED = 42

# Island centre (for buffer zone)
ISLAND_CENTER_LON = -59.54
ISLAND_CENTER_LAT = 13.18
