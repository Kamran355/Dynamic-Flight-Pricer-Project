"""
Analytics and model diagnostics report
Run via main.py or directly: python reports.py
"""

from config import DATA_FILE, MODEL_FILE
from demand import LogisticDemandModel
from mdp_agent    import QLearningAgent
from history      import FlightHistory
from utils        import (
    print_header, print_section, format_currency
)


def run_report():
    print_header("FLIGHT PRICING  —  ANALYTICS REPORT")

    hist     = FlightHistory(DATA_FILE)
    lr_model = LogisticDemandModel(MODEL_FILE)
    q_agent  = QLearningAgent(MODEL_FILE)

    records = hist.all_records()
    if not records:
        print("\n  No records yet. Run a pricing session first.")
        return

    # Overall summary
    print_section("Overall Summary")
    total     = len(records)
    accepted  = sum(1 for r in records if r.get("accepted"))
    denied    = total - accepted
    rewards   = [r.get("realized_reward", 0) for r in records if r.get("accepted")]
    print(f"  Total quotes   : {total}")
    print(f"  Accepted       : {accepted}   Denied: {denied}")
    print(f"  Acceptance rate: {100*accepted/total:.1f}%")
    print(f"  Total profit   : {format_currency(sum(rewards))}")
    if rewards:
        print(f"  Avg profit/flight : {format_currency(sum(rewards)/len(rewards))}")

    # Per-route
    print_section("Per-Route Statistics")
    summary = hist.route_summary()
    for route, data in sorted(summary.items()):
        print(f"\n  {route}")
        print(f"    Quotes   : {data['total']}  "
              f"({data['accepted']}  {data['denied']}  "
              f"— {data['acceptance_rate_pct']}% acceptance)")
        if data["avg_price"]:
            print(f"    Avg price / pax : {format_currency(data['avg_price'])}")
        if data["avg_revenue"]:
            print(f"    Avg revenue     : {format_currency(data['avg_revenue'])}")
        if data["total_revenue"]:
            print(f"    Total revenue   : {format_currency(data['total_revenue'])}")

    # Logistic regression state
    print_section("Layer 1 — Logistic Regression Coefficients")
    lr_summary = lr_model.summary()
    status = "ACTIVE" if lr_summary["model_active"] else \
             f"WARMING UP ({lr_summary['n_updates']}/{lr_summary['min_samples_needed']} samples)"
    print(f"  Status   : {status}")
    print(f"  Updates  : {lr_summary['n_updates']}")
    print()
    for feat, coef in lr_summary["coefficients"].items():
        direction = "▲" if coef > 0 else "▼"
        print(f"    {feat:<20} {coef:>+10.5f}  {direction}")
    print()
    for key, note in lr_summary["interpretation"].items():
        print(f"    {note}")

    # Q-learning state
    print_section("Layer 3 — Q-Learning Agent")
    print(f"  Episodes  : {q_agent.n_episodes}")
    print(f"  Epsilon   : {q_agent.epsilon:.4f}  "
          f"({'exploring' if q_agent.epsilon > 0.1 else 'exploiting'})")
    if q_agent.is_active:
        ps = q_agent.policy_summary()
        # Show a few representative states
        print(f"\n  Sample policy entries (state → recommended multiplier):")
        shown = 0
        for state_str, entry in ps["policy"].items():
            if entry["q_value"] != 0.0:
                print(f"    State {state_str}  →  "
                      f"{entry['multiplier']:.2f}x  "
                      f"(Q = {entry['q_value']:.3f})")
                shown += 1
            if shown >= 8:
                break
        if shown == 0:
            print("(No non-zero Q-values yet — agent still in early exploration)")


if __name__ == "__main__":
    run_report()