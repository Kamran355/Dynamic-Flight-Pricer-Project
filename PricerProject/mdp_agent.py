"""
State Space S:
    s = (acceptance_bucket, lead_time_bucket, load_factor_bucket, season_bucket)

    (see Technical Report for State Space buckets)

Action Space A:
    Each action maps to a price multiplier relative to the NLP optimal price.
    Action 0 = 60% of optimal, action (N-1) = 140% of optimal.
    This keeps the MDP grounded in cost reality while allowing ±40% exploration.

Q-Learning Update (Bellman equation approximation):
    Q(s, a) = Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

    where:
        r = realized reward (revenue - cost if accepted, 0 if denied)
        γ = discount factor (MDP_DISCOUNT_FACTOR)
        α = learning rate   (MDP_LEARNING_RATE)
        s' = next state (updated after observing accept/deny outcome)

Exploration Strategy:
    ε-greedy: with probability ε, choose a random action (explore);
    otherwise choose argmax_a Q(s, a) (exploit).
    ε decays multiplicatively each episode toward MDP_EPSILON_MIN,
    shifting from exploration to exploitation as data accumulates.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Tuple

from config import (
    MDP_DISCOUNT_FACTOR, MDP_LEARNING_RATE,
    MDP_EPSILON_START, MDP_EPSILON_MIN, MDP_EPSILON_DECAY,
    MDP_PRICE_BINS, MDP_STATE_BINS, MODEL_FILE
)


# State Discretization

def discretize_acceptance_rate(rate: float) -> int:
    # 0-3 bucket for recent route acceptance rate
    if rate < 0.25: return 0
    if rate < 0.50: return 1
    if rate < 0.75: return 2
    return 3


def discretize_lead_days(lead_days: float) -> int:
    # 0-3 bucket for booking lead time
    if lead_days < 1:   return 0   # same-day
    if lead_days < 3:   return 1   # very short notice
    if lead_days < 14:  return 2   # moderate lead time
    return 3                        # well in advance


def discretize_load_factor(num_pax: int, seats: int = 3) -> int:
    # 0-2 bucket for load factor (seats requested / available)
    lf = num_pax / seats
    if lf <= 0.33: return 0   # light
    if lf <= 0.67: return 1   # medium
    return 2                    # full / near-full


def discretize_season(month: int) -> int:
    # 0-3 bucket for quarter of year
    return (month - 1) // 3   # Q1=0, Q2=1, Q3=2, Q4=3


def build_state(
    acceptance_rate: float,
    lead_days: float,
    num_pax: int,
    month: int
) -> Tuple[int, int, int, int]:
    return (
        discretize_acceptance_rate(acceptance_rate),
        discretize_lead_days(lead_days),
        discretize_load_factor(num_pax),
        discretize_season(month),
    )


# Action --- Price Multiplier

def action_to_multiplier(action: int, n_actions: int = MDP_PRICE_BINS) -> float:

    # Map discrete action index to a price multiplier in [0.60, 1.40]
    # Action 0 = 60% of NLP optimal; action (n-1) = 140%

    lo, hi = 0.60, 1.40
    return lo + (hi - lo) * action / (n_actions - 1)


# Q-Table

def _empty_q_table() -> list:
    """
    Create a zeroed Q-table of shape
    [acc_bins, lead_bins, load_bins, season_bins, n_actions].
    Stored as a nested list for JSON serializability.
    """
    a, b, c, d = MDP_STATE_BINS
    n = MDP_PRICE_BINS
    return [[[[[ 0.0 for _ in range(n)]
               for _ in range(d)]
              for _ in range(c)]
             for _ in range(b)]
            for _ in range(a)]


def _q_get(table: list, state: tuple, action: int) -> float:
    s0, s1, s2, s3 = state
    return table[s0][s1][s2][s3][action]


def _q_set(table: list, state: tuple, action: int, value: float):
    s0, s1, s2, s3 = state
    table[s0][s1][s2][s3][action] = value


def _q_max(table: list, state: tuple) -> float:
    s0, s1, s2, s3 = state
    return max(table[s0][s1][s2][s3])


def _q_argmax(table: list, state: tuple) -> int:
    s0, s1, s2, s3 = state
    row = table[s0][s1][s2][s3]
    return int(np.argmax(row))


# Agent Class

class QLearningAgent:
    """
    Tabular Q-learning agent for flight pricing.
    Attributes:
    q_table   : nested list, shape [4,4,3,4,n_actions]
    epsilon   : current exploration rate
    n_episodes: total Q-learning updates performed
    """

    def __init__(self, model_file: Path = MODEL_FILE):
        self.model_file = model_file
        state = self._load_state()
        self.q_table    = state.get("q_table",    _empty_q_table())
        self.epsilon    = state.get("epsilon",    MDP_EPSILON_START)
        self.n_episodes = state.get("n_episodes", 0)

    # Persistence

    def _load_state(self) -> dict:
        if Path(self.model_file).exists():
            with open(self.model_file) as f:
                data = json.load(f)
            return data.get("q_learning", {})
        return {}

    def _save_state(self):
        path = Path(self.model_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with open(path) as f:
                full = json.load(f)
        else:
            full = {}
        full["q_learning"] = {
            "q_table":    self.q_table,
            "epsilon":    self.epsilon,
            "n_episodes": self.n_episodes,
        }
        with open(path, "w") as f:
            json.dump(full, f, indent=2)

    # Policy

    def select_action(self, state: tuple) -> Tuple[int, str]:
        """
        ε-greedy action selection.
        Returns (action_index, "explore"/"exploit").
        """
        if random.random() < self.epsilon:
            return random.randint(0, MDP_PRICE_BINS - 1), "explore"
        return _q_argmax(self.q_table, state), "exploit"

    def suggest_price_multiplier(self, state: tuple) -> Tuple[float, str]:
        """
        Return the greedy price multiplier (no exploration) plus a mode label.
        Used by the optimizer to get the MDP's best current guess.
        """
        action = _q_argmax(self.q_table, state)
        mult   = action_to_multiplier(action)
        mode   = "exploit (greedy policy)"
        return mult, mode

    # Learning

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
    ):

        # Perform one Q-learning (Bellman) update:
        current_q  = _q_get(self.q_table, state, action)
        max_next_q = _q_max(self.q_table, next_state)
        td_error   = reward + MDP_DISCOUNT_FACTOR * max_next_q - current_q
        new_q      = current_q + MDP_LEARNING_RATE * td_error
        _q_set(self.q_table, state, action, new_q)

        # Decay epsilon
        self.epsilon = max(MDP_EPSILON_MIN, self.epsilon * MDP_EPSILON_DECAY)
        self.n_episodes += 1
        self._save_state()

    def record_outcome(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
    ):
        # Convenience wrapper: update Q-table and persist
        self.update(state, action, reward, next_state)

    # Diagnostics

    def policy_summary(self) -> dict:
        # Return the greedy action and multiplier for every state
        a_bins, b_bins, c_bins, d_bins = MDP_STATE_BINS
        policy = {}
        for s0 in range(a_bins):
            for s1 in range(b_bins):
                for s2 in range(c_bins):
                    for s3 in range(d_bins):
                        s = (s0, s1, s2, s3)
                        act = _q_argmax(self.q_table, s)
                        policy[str(s)] = {
                            "action":     act,
                            "multiplier": round(action_to_multiplier(act), 3),
                            "q_value":    round(_q_get(self.q_table, s, act), 4),
                        }
        return {
            "n_episodes":    self.n_episodes,
            "epsilon":       round(self.epsilon, 4),
            "policy":        policy,
        }

    @property
    def is_active(self) -> bool:
        # True once the agent has seen at least one update
        return self.n_episodes > 0