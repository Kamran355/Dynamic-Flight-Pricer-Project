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

    # Inference

    def predict_accept_prob(
        self,
        price_per_pax: float,
        flight_hours: float,
        lead_days: float,
        num_pax: int,
        is_round_trip: bool
    ) -> float:
        # Return P(accept) in [0, 1] for the given flight/price configuration.
        if self.n_updates < LR_MIN_SAMPLES:
            return 0.5   # uninformative prior, not enough data yet

        x = _build_feature_vector(
            price_per_pax, flight_hours, lead_days, num_pax, is_round_trip
        )
        z = sum(b * xi for b, xi in zip(self.betas, x))
        return _sigmoid(z)

    def predict_accept_prob_curve(
        self,
        price_grid: np.ndarray,
        flight_hours: float,
        lead_days: float,
        num_pax: int,
        is_round_trip: bool
    ) -> np.ndarray:
        # Vectorised: return P(accept) for every price in price_grid.
        probs = np.array([
            self.predict_accept_prob(p, flight_hours, lead_days, num_pax, is_round_trip)
            for p in price_grid
        ])
        return probs

    # Learning

    def update(
        self,
        price_per_pax: float,
        flight_hours: float,
        lead_days: float,
        num_pax: int,
        is_round_trip: bool,
        accepted: bool
    ):
        """
        Perform one SGD step given a new (features, outcome) observation.

        Update rule:
            error   = y - sigmoid(beta @ x)
            beta_j += alpha * (error * x_j  -  lambda * beta_j)

        The L2 term (lambda * beta_j) shrinks large weights toward zero,
        acting as a regularizer to prevent overfitting on small datasets.
        """
        x = _build_feature_vector(
            price_per_pax, flight_hours, lead_days, num_pax, is_round_trip
        )
        z      = sum(b * xi for b, xi in zip(self.betas, x))
        p_hat  = _sigmoid(z)
        y      = 1.0 if accepted else 0.0
        error  = y - p_hat

        alpha  = LR_LEARNING_RATE
        lam    = LR_REGULARIZATION

        for j in range(len(self.betas)):
            # Skip L2 penalty on the intercept
            reg = lam * self.betas[j] if j > 0 else 0.0
            self.betas[j] += alpha * (error * x[j] - reg)

        self.n_updates += 1
        self._save_state()

    # Diagnostics

    def summary(self) -> dict:
        # Return a human-readable snapshot of the current model state
        feature_names = [
            "intercept", "price_per_pax", "flight_hours",
            "lead_days", "num_pax", "is_round_trip"
        ]
        return {
            "n_updates": self.n_updates,
            "min_samples_needed": LR_MIN_SAMPLES,
            "model_active": self.n_updates >= LR_MIN_SAMPLES,
            "coefficients": {
                name: round(b, 6)
                for name, b in zip(feature_names, self.betas)
            },
            "interpretation": {
                "price_per_pax": (
                    "Negative = higher price reduces P(accept). "
                    f"Current: {self.betas[1]:+.5f} per $1 price increase."
                ),
                "lead_days": (
                    "Positive = booking further in advance is slightly more likely to accept. "
                    f"Current: {self.betas[3]:+.5f} per day."
                ),
            }
        }