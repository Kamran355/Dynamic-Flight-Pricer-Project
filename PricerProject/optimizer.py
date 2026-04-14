"""
Objective:
    R(p) = p * P(accept | p, x)    [expected revenue per seat]
"""
import math
import numpy as np
from scipy.optimize import minimize_scalar

from config import (
    MAX_USABLE_WEIGHT, MAX_FUEL_GALLONS, FUEL_LBS_PER_GALLON,
    FUEL_BURN_GPH, RESERVE_HOURS, PILOT_WEIGHT_LBS,
    COST_PER_TACH_HOUR, TACH_TO_FLIGHT_HOUR_RATIO,
    MIN_MARGIN, PRICE_CEILING_MULTIPLIER, PRICE_GRID_STEPS
)
from demand import LogisticDemandModel


# Weight & Fuel Utilities

def gallons_required(flight_hours: float) -> float:
    # Minimum gallons needed: burn + reserve
    return FUEL_BURN_GPH * (flight_hours + RESERVE_HOURS)

def fuel_weight(gallons: float) -> float:
    return gallons * FUEL_LBS_PER_GALLON


def tach_hours(flight_hours: float) -> float:
    return flight_hours * TACH_TO_FLIGHT_HOUR_RATIO


# Constraint Checkers

def check_weight_constraint(
    num_pax: int,
    avg_pax_weight: float,
    payload_lbs: float,
    gallons_at_departure: float,
) -> dict:
    w_fuel    = fuel_weight(gallons_at_departure)
    w_pax     = num_pax * avg_pax_weight
    w_total   = w_fuel + w_pax + payload_lbs + PILOT_WEIGHT_LBS
    slack     = MAX_USABLE_WEIGHT - w_total   # positive = feasible

    return {
        "feasible":   slack >= 0,
        "total_lbs":  round(w_total, 2),
        "slack_lbs":  round(slack, 2),
        "breakdown": {
            "fuel_lbs":    round(w_fuel, 2),
            "pax_lbs":     round(w_pax, 2),
            "payload_lbs": round(payload_lbs, 2),
            "pilot_lbs":   PILOT_WEIGHT_LBS,
        }
    }


def max_gallons_within_weight(
    num_pax: int,
    avg_pax_weight: float,
    payload_lbs: float
) -> float:

    # Given pax + payload, return the maximum gallons we can load without violating the weight constraint.
    non_fuel_weight = (num_pax * avg_pax_weight) + payload_lbs + PILOT_WEIGHT_LBS
    available_for_fuel = MAX_USABLE_WEIGHT - non_fuel_weight
    max_gal = available_for_fuel / FUEL_LBS_PER_GALLON
    return max(0.0, min(max_gal, MAX_FUEL_GALLONS))


# Fuel LP (Closed-Form Solution)

def solve_fuel_lp(
    flight_hours: float,
    price_origin: float,
    price_dest: float,
    is_round_trip: bool,
    max_gal_departure: float,
) -> dict:

    # Solve the fuel sourcing LP analytically.
    req_one_way = gallons_required(flight_hours)
    req_one_way = min(req_one_way, max_gal_departure)

    if not is_round_trip:
        cost = req_one_way * price_origin
        return {
            "strategy":        "origin_only",
            "gallons_origin":  round(req_one_way, 3),
            "gallons_dest":    0.0,
            "total_fuel_cost": round(cost, 2),
            "note": f"One-way. Load {req_one_way:.1f} gal at origin "
                    f"(${price_origin:.2f}/gal)."
        }

    req_return = gallons_required(flight_hours)
    req_return = min(req_return, MAX_FUEL_GALLONS)

    # Option A: fuel fully at origin for both legs (if tank fits)
    rt_needed = min(req_one_way + req_return, max_gal_departure)
    cost_a    = rt_needed * price_origin

    # Option B: minimal fuel at origin for outbound + refuel at destination
    cost_b = (req_one_way * price_origin) + (req_return * price_dest)

    # Option C: only origin fuel, carry enough for round trip if weight allows
    # (same as A if rt_needed fits; A already handles this)

    if cost_a <= cost_b:
        saving = round(cost_b - cost_a, 2)
        return {
            "strategy":        "origin_full",
            "gallons_origin":  round(rt_needed, 3),
            "gallons_dest":    0.0,
            "total_fuel_cost": round(cost_a, 2),
            "note": f"LP optimum: fuel full at origin (${price_origin:.2f}/gal). "
                    f"Saves ${saving:.2f} vs split fueling."
        }
    else:
        saving = round(cost_a - cost_b, 2)
        return {
            "strategy":        "split",
            "gallons_origin":  round(req_one_way, 3),
            "gallons_dest":    round(req_return, 3),
            "total_fuel_cost": round(cost_b, 2),
            "note": f"LP optimum: split fueling — origin ${price_origin:.2f}/gal outbound, "
                    f"dest ${price_dest:.2f}/gal return. Saves ${saving:.2f} vs fueling full."
        }


# Operating Cost

def compute_operating_cost(
    flight_hours: float,
    fuel_cost_total: float,
    is_round_trip: bool
) -> dict:
    legs         = 2 if is_round_trip else 1
    t_hours      = tach_hours(flight_hours) * legs
    aircraft_cost = t_hours * COST_PER_TACH_HOUR
    total        = aircraft_cost + fuel_cost_total

    return {
        "tach_hours":           round(t_hours, 3),
        "aircraft_cost":        round(aircraft_cost, 2),
        "fuel_cost":            round(fuel_cost_total, 2),
        "total_operating_cost": round(total, 2),
    }


# Revenue Optimizer (NLP)

def optimize_price(
    demand_model: LogisticDemandModel,
    operating_cost: float,
    num_pax: int,
    flight_hours: float,
    lead_days: float,
    is_round_trip: bool,
    q_learning_suggestion: float = None,
) -> dict:
    """
    When the logistic model has insufficient data, falls back to
    cost-plus pricing with BASE_MARKUP. When a Q-learning suggestion
    is available (Layer 3), it is blended with the optimizer's result
    using a weighted average that gives more weight to Q-learning as
    the MDP accumulates more experience.

    q_learning_suggestion : float or None
        Price recommended by the MDP policy (To Be Added in Layer 3). Blended in when
        both layers are active.
    """
    cost_per_seat = operating_cost / num_pax
    p_floor = cost_per_seat * (1.0 + MIN_MARGIN)
    p_ceil  = cost_per_seat * PRICE_CEILING_MULTIPLIER
    p_ceil  = max(p_ceil, p_floor * 2)   # ensure search space is meaningful

    # Grid search for global max
    price_grid = np.linspace(p_floor, p_ceil, PRICE_GRID_STEPS)
    probs      = demand_model.predict_accept_prob_curve(
        price_grid, flight_hours, lead_days, num_pax, is_round_trip
    )
    revenue_grid = price_grid * probs

    grid_best_idx = int(np.argmax(revenue_grid))
    grid_best_p   = float(price_grid[grid_best_idx])

    # Scalar refinement around grid best
    refine_lo = max(p_floor, grid_best_p - (p_ceil - p_floor) / PRICE_GRID_STEPS)
    refine_hi = min(p_ceil,  grid_best_p + (p_ceil - p_floor) / PRICE_GRID_STEPS)

    def neg_revenue(p):
        prob = demand_model.predict_accept_prob(
            p, flight_hours, lead_days, num_pax, is_round_trip
        )
        return -(p * prob)

    result   = minimize_scalar(neg_revenue, bounds=(refine_lo, refine_hi), method="bounded")
    opt_price = float(result.x) if result.success else grid_best_p
    opt_price = max(p_floor, min(p_ceil, opt_price))

    opt_prob    = demand_model.predict_accept_prob(
        opt_price, flight_hours, lead_days, num_pax, is_round_trip
    )
    opt_revenue = opt_price * opt_prob * num_pax

    # Blend with Q-learning if available
    final_price     = opt_price
    blend_note      = "NLP optimizer only (MDP not yet active)."
    q_blend_weight  = 0.0

    if q_learning_suggestion is not None:
        q_w = min(0.5, 0.05 * math.log1p(demand_model.n_updates))
        q_blend_weight = round(q_w, 3)
        final_price = (1 - q_w) * opt_price + q_w * q_learning_suggestion
        final_price = max(p_floor, min(p_ceil, final_price))
        blend_note  = (
            f"Blended: {100*(1-q_w):.0f}% NLP optimizer + "
            f"{100*q_w:.0f}% MDP Q-learning "
            f"(Q-suggestion: ${q_learning_suggestion:.2f})."
        )

    final_prob    = demand_model.predict_accept_prob(
        final_price, flight_hours, lead_days, num_pax, is_round_trip
    )
    final_revenue = final_price * final_prob * num_pax
    final_profit  = final_revenue - operating_cost

    return {
        "price_floor":          round(p_floor, 2),
        "price_ceiling":        round(p_ceil, 2),
        "nlp_optimal_price":    round(opt_price, 2),
        "nlp_accept_prob":      round(opt_prob, 4),
        "q_blend_weight":       q_blend_weight,
        "final_price_per_pax":  round(final_price, 2),
        "final_accept_prob":    round(final_prob, 4),
        "expected_revenue":     round(final_revenue, 2),
        "expected_profit":      round(final_profit, 2),
        "cost_per_seat":        round(cost_per_seat, 2),
        "blend_note":           blend_note,

        # Full revenue curve for reporting / plotting
        "price_grid":           price_grid.tolist(),
        "revenue_curve":        revenue_grid.tolist(),
    }