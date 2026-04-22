"""
Flow:
Collect inputs
Solve fuel LP  (L2 — optimizer.py)
Check weight constraint (optimizer.py)
Query MDP for price suggestion  (L3 — mdp_agent.py)
Run NLP optimizer blended with MDP  (L2)
Display results
Collect accept/deny outcome
Update logistic regression (L1 — demand_model.py)
Update Q-table (L3)
Persist full record (history.py)
"""

from datetime import datetime

from config import (
    SEATS_AVAILABLE, MAX_PAYLOAD_LBS, DATA_FILE, MODEL_FILE
)
from demand   import LogisticDemandModel
from mdp_agent      import QLearningAgent, build_state, action_to_multiplier, MDP_PRICE_BINS
from optimizer      import (
    solve_fuel_lp, compute_operating_cost, optimize_price,
    check_weight_constraint, max_gallons_within_weight
)
from history        import FlightHistory
from utils          import (
    prompt_float, prompt_int, prompt_str, prompt_bool,
    format_currency, print_header, print_section, print_warning, print_ok, print_info
)


def run_pricing_session():
    print_header("FLIGHT PRICING ENGINE")

    # Instantiate all three layers
    lr_model  = LogisticDemandModel(MODEL_FILE)
    q_agent   = QLearningAgent(MODEL_FILE)
    hist      = FlightHistory(DATA_FILE)

    # Inputs
    print_section("Flight Details")
    origin      = prompt_str("Origin airport (ICAO/IATA or name)")
    destination = prompt_str("Destination airport (ICAO/IATA or name)")
    date_str    = prompt_str("Departure date & time (e.g. 2025-07-04 14:30)")
    is_rt       = prompt_bool("Round trip?")

    print_section("Passengers & Payload")
    num_pax        = prompt_int("Passengers (excl. pilot)", 1, SEATS_AVAILABLE)
    avg_pax_weight = prompt_float("Average passenger weight (lbs)", 50, 400)
    payload_lbs    = prompt_float("Total payload (lbs, max 200)", 0, MAX_PAYLOAD_LBS)

    print_section("Fuel Prices")
    price_origin = prompt_float("Fuel price at origin ($/gal)")
    price_dest   = prompt_float("Fuel price at destination ($/gal)")

    print_section("Flight Duration")
    flight_hours = prompt_float("Estimated flight duration — one way (hrs)", 0.1, 20)
    route_key  = f"{origin.upper()}-{destination.upper()}"
    lead_days  = hist.compute_lead_days(date_str)
    acc_rate   = hist.route_acceptance_rate(route_key)

    try:
        dep_month = int(date_str[5:7])
    except Exception:
        dep_month = datetime.now().month

    # Fuel LP
    max_gal = max_gallons_within_weight(num_pax, avg_pax_weight, payload_lbs)
    fuel_rec = solve_fuel_lp(flight_hours, price_origin, price_dest, is_rt, max_gal)

    # Weight constraint
    weight_check = check_weight_constraint(
        num_pax, avg_pax_weight, payload_lbs, fuel_rec["gallons_origin"]
    )

    # Operating cost
    op_cost = compute_operating_cost(flight_hours, fuel_rec["total_fuel_cost"], is_rt)

    # MDP state & suggestion
    mdp_state  = build_state(acc_rate, lead_days, num_pax, dep_month)
    mdp_mult, mdp_mode = q_agent.suggest_price_multiplier(mdp_state)
    cost_per_seat      = op_cost["total_operating_cost"] / num_pax
    q_suggestion       = cost_per_seat * mdp_mult if q_agent.is_active else None

    # NLP price optimization
    pricing = optimize_price(
        demand_model          = lr_model,
        operating_cost        = op_cost["total_operating_cost"],
        num_pax               = num_pax,
        flight_hours          = flight_hours,
        lead_days             = lead_days,
        is_round_trip         = is_rt,
        q_learning_suggestion = q_suggestion,
    )

    # Map final price back to MDP action
    final_mult  = pricing["final_price_per_pax"] / cost_per_seat if cost_per_seat > 0 else 1.0
    mdp_lo, mdp_hi = 0.60, 1.40
    raw_action  = (final_mult - mdp_lo) / (mdp_hi - mdp_lo) * (MDP_PRICE_BINS - 1)
    mdp_action  = max(0, min(MDP_PRICE_BINS - 1, round(raw_action)))

    # Display
    print_header("RESULTS")
    print_section("Weight & Balance")
    status = "WITHIN LIMITS" if weight_check["feasible"] else "OVERWEIGHT"
    print(f"  Status       : {status}")
    print(f"  Total weight : {weight_check['total_lbs']} lbs  "
          f"(limit: 1241.23 lbs,  slack: {weight_check['slack_lbs']:+.1f} lbs)")
    bd = weight_check["breakdown"]
    print(f"  Breakdown    : Fuel {bd['fuel_lbs']} | Pax {bd['pax_lbs']} | "
          f"Payload {bd['payload_lbs']} | Pilot {bd['pilot_lbs']}")

    if not weight_check["feasible"]:
        print_warning("Overweight — cannot fly this configuration. "
                      "Reduce passengers, payload, or fuel load.")
        return

    print_section("Fuel  (LP Solution)")
    print(f"  Strategy       : {fuel_rec['strategy']}")
    print(f"  Gallons @ origin : {fuel_rec['gallons_origin']}")
    print(f"  Gallons @ dest   : {fuel_rec['gallons_dest']}")
    print(f"  Total fuel cost  : {format_currency(fuel_rec['total_fuel_cost'])}")
    print(f"  Note             : {fuel_rec['note']}")

    print_section("Operating Costs")
    print(f"  Aircraft ({op_cost['tach_hours']} tach hrs) : "
          f"{format_currency(op_cost['aircraft_cost'])}")
    print(f"  Fuel                       : {format_currency(op_cost['fuel_cost'])}")
    print(f"  Total operating cost       : {format_currency(op_cost['total_operating_cost'])}")
    print(f"  Cost per seat              : {format_currency(pricing['cost_per_seat'])}")

    print_section("Model Status")
    lr_active = lr_model.n_updates >= 5
    print(f"  Logistic regression : {'ACTIVE' if lr_active else 'WARMING UP'} "
          f"({lr_model.n_updates} updates)")
    print(f"  Q-learning agent    : {'ACTIVE' if q_agent.is_active else 'WARMING UP'} "
          f"({q_agent.n_episodes} episodes,  ε={q_agent.epsilon:.3f})")
    print(f"  Route acceptance    : {acc_rate*100:.1f}%  (recent history,  "
          f"lead time: {lead_days:.0f} days)")

    print_section("Optimal Pricing  (NLP + MDP Blend)")
    print(f"  Price floor            : {format_currency(pricing['price_floor'])}")
    print(f"  Price ceiling          : {format_currency(pricing['price_ceiling'])}")
    print(f"  NLP optimal price      : {format_currency(pricing['nlp_optimal_price'])}  "
          f"(P(accept) = {pricing['nlp_accept_prob']:.1%})")
    print(f"  ──")
    print(f"  Final price / pax      : {format_currency(pricing['final_price_per_pax'])}  "
          f"(P(accept) = {pricing['final_accept_prob']:.1%})")
    print(f"  Expected revenue       : {format_currency(pricing['expected_revenue'])}")
    print(f"  Expected profit        : {format_currency(pricing['expected_profit'])}")
    print(f"  Blend note             : {pricing['blend_note']}")

    # Accept / Deny
    print()
    accepted = prompt_bool("Was this pricing accepted by the passenger(s)?")

    # Realized reward
    if accepted:
        realized_revenue = pricing["final_price_per_pax"] * num_pax
        reward = realized_revenue - op_cost["total_operating_cost"]
    else:
        reward = 0.0

    # Layer 1 update: SGD on logistic regression
    lr_model.update(
        price_per_pax  = pricing["final_price_per_pax"],
        flight_hours   = flight_hours,
        lead_days      = lead_days,
        num_pax        = num_pax,
        is_round_trip  = is_rt,
        accepted       = accepted,
    )

    # Layer 3 update: Q-learning Bellman step
    next_acc_rate = (
        (acc_rate * 0.9 + 0.1) if accepted else (acc_rate * 0.9)
    )
    next_state = build_state(next_acc_rate, max(0, lead_days - 1), num_pax, dep_month)
    q_agent.record_outcome(mdp_state, mdp_action, reward, next_state)

    # Persist full record
    record = {
        "timestamp":        datetime.now().isoformat(),
        "origin":           origin,
        "destination":      destination,
        "route_key":        route_key,
        "date_requested":   date_str,
        "is_round_trip":    is_rt,
        "num_pax":          num_pax,
        "avg_pax_weight":   avg_pax_weight,
        "payload_lbs":      payload_lbs,
        "fuel_price_origin":price_origin,
        "fuel_price_dest":  price_dest,
        "flight_hours":     flight_hours,
        "lead_days":        lead_days,
        "fuel_recommendation": fuel_rec,
        "weight_check":     weight_check,
        "operating_cost":   op_cost,
        "pricing_result":   {k: v for k, v in pricing.items()
                             if k not in ("price_grid", "revenue_curve")},
        "mdp_state":        list(mdp_state),
        "mdp_action":       int(mdp_action),
        "realized_reward":  round(reward, 2),
        "accepted":         accepted,
        "lr_betas_snapshot": lr_model.betas[:],
    }
    hist.append(record)

    outcome_str = "ACCEPTED" if accepted else "DENIED"
    print_ok(f"\n  Record saved — {outcome_str}   "
             f"|  Reward: {format_currency(reward)}")
    print_info(f"  LR model updated (n={lr_model.n_updates})  |  "
               f"Q-agent updated (ep={q_agent.n_episodes}, ε={q_agent.epsilon:.3f})")


if __name__ == "__main__":
    run_pricing_session()