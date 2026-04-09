from pathlib import Path

# Aircraft Limits
SEATS_AVAILABLE = 3  # Paying passenger seats (4 total minus pilot)
MAX_USABLE_WEIGHT = 1241.23  # lbs — combined limit: fuel + pax + payload
MAX_FUEL_GALLONS = 72.0  # Usable fuel capacity (gallons)
FUEL_LBS_PER_GALLON = 6.0  # Avgas weight (lbs/gal)
FUEL_BURN_GPH = 12.0  # Cruise fuel burn (gal/hr)
MAX_PAYLOAD_LBS = 200.0  # Hard baggage/cargo limit (lbs)
RESERVE_HOURS = 0.75  # VFR/IFR adjustable fuel reserve (hours)

# Pilot
PILOT_WEIGHT_LBS = 180  # Update to your actual weight

# Operating Costs
COST_PER_TACH_HOUR = 115.00  # Aircraft cost per tach hour ($)
TACH_TO_FLIGHT_HOUR_RATIO = 0.8  # Long-run avg: 1 tach hr = 0.8 flight hrs

# Pricing Bounds
MIN_MARGIN = 0.05  # Floor: never price below 5% above cost
PRICE_CEILING_MULTIPLIER = 5.0  # Max price = 5x operating cost per seat
PRICE_GRID_STEPS = 200  # Resolution of price search grid

# Logistic Regression (Layer 1)
LR_LEARNING_RATE = 0.05  # SGD step size for beta updates
LR_REGULARIZATION = 0.01  # L2 regularization lambda (prevents overfitting)
LR_MIN_SAMPLES = 5  # Minimum records before regression activates
# Initial beta coefficients [intercept, price, flight_hrs, lead_days, num_pax, is_round_trip]
LR_INITIAL_BETAS = [2.0, -0.01, -0.1, 0.005, -0.05, 0.1]

# MDP / Q-Learning (Layer 3)
MDP_DISCOUNT_FACTOR = 0.92  # gamma — future reward discount
MDP_LEARNING_RATE = 0.15  # alpha — Q-table update step size
MDP_EPSILON_START = 0.25  # Exploration rate (start)
MDP_EPSILON_MIN = 0.02  # Exploration rate (floor)
MDP_EPSILON_DECAY = 0.97  # Epsilon multiplied by this each episode
MDP_PRICE_BINS = 12  # Number of discrete price actions in Q-table
# State dimensions: [acceptance_bucket, lead_time_bucket, load_factor_bucket, season_bucket]
MDP_STATE_BINS = [4, 4, 3, 4]

# Storage
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "flight_history.json"
MODEL_FILE = BASE_DIR / "data" / "model_state.json"