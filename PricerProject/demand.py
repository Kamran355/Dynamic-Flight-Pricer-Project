import json
import math
import numpy as np
from pathlib import Path

from config import (
    LR_LEARNING_RATE, LR_REGULARIZATION, LR_MIN_SAMPLES,
    LR_INITIAL_BETAS, DATA_FILE, MODEL_FILE
)


# Helpers

def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        e = math.exp(z)
        return e / (1.0 + e)


def _build_feature_vector(
    price_per_pax: float,
    flight_hours: float,
    lead_days: float,
    num_pax: int,
    is_round_trip: bool
) -> list:
    return [
        1.0,                        # intercept / bias term
        price_per_pax,              # quoted price per passenger ($)
        flight_hours,               # one-way flight duration (hrs)
        lead_days,                  # days between quote and departure
        float(num_pax),             # number of paying passengers
        1.0 if is_round_trip else 0.0,  # round-trip indicator
    ]


# Model Class

class LogisticDemandModel:
    """
    Online logistic regression that updates its coefficients every time
    a new accept/deny outcome is recorded.
    """

    def __init__(self, model_file: Path = MODEL_FILE):
        self.model_file = model_file
        self._ensure_data_dir()
        state = self._load_state()
        self.betas    = state.get("betas",     list(LR_INITIAL_BETAS))
        self.n_updates = state.get("n_updates", 0)

    # Persistence

    def _ensure_data_dir(self):
        self.model_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> dict:
        if self.model_file.exists():
            with open(self.model_file) as f:
                data = json.load(f)
            return data.get("logistic_regression", {})
        return {}

    def _save_state(self):
        # Load full model file first so other layers aren't overwritten
        if self.model_file.exists():
            with open(self.model_file) as f:
                full = json.load(f)
        else:
            full = {}
        full["logistic_regression"] = {
            "betas":     self.betas,
            "n_updates": self.n_updates,
        }
        with open(self.model_file, "w") as f:
            json.dump(full, f, indent=2)

