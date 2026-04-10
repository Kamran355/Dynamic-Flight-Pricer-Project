import json
from datetime import datetime, date
from pathlib import Path

from config import DATA_FILE

class FlightHistory:

    def __init__(self, data_file: Path = DATA_FILE):
        self.data_file = data_file
        self._ensure_store()

    def _ensure_store(self):
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.data_file.exists():
            self.data_file.write_text(json.dumps({"records": []}, indent=2))

    def _load(self) -> dict:
        with open(self.data_file) as f:
            return json.load(f)

    def _save(self, store: dict):
        with open(self.data_file, "w") as f:
            json.dump(store, f, indent=2, default=str)

    # Write

    def append(self, record: dict):
        """Add a completed pricing session to the history file."""
        store = self._load()
        store["records"].append(record)
        self._save(store)

    # Read

    def all_records(self) -> list:
        return self._load().get("records", [])

    def records_for_route(self, route_key: str) -> list:
        return [r for r in self.all_records() if r.get("route_key") == route_key]

    # Route Statistics

    def route_acceptance_rate(self, route_key: str, last_n: int = 20) -> float:
        """Compute the recent acceptance rate for a route
        Uses the last "last_n" records to stay responsive to trend changes
        Returns 0.5 (neutral state) if fewer than 2 records exist
        """

        recs = self.records_for_route(route_key)
        if len(recs) < 2:
            return 0.5
        recent = sorted(recs, key=lambda r: r.get("timestamp", ""))[-last_n:]
        return sum(1 for r in recent if r.get("accepted", False)) / len(recent)

    def compute_lead_days(self, date_str: str) -> float:
        """
        Return the number of days between now and the requested departure date
        Clamps to 0 minimum
        """
        try:
            dep = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
            delta = (dep - date.today()).days
            return max(0.0, float(delta))
        except Exception:
            return 7.0  # default 1 week if parsing fails
    def route_summary(self) -> dict:
        """Aggregate statistics per route for the reports module"""
        records = self.all_records()
        summary = {}
        for rec in records:
            key = rec.get("route_key", "UNKNOWN")
            if key not in summary:
                summary[key] = {
                    "total": 0, "accepted": 0, "denied": 0,
                    "prices": [], "revenues": []
                }
            s = summary[key]
            s["total"] += 1
            if rec.get("accepted"):
                s["accepted"] += 1
                rev = rec.get("pricing_result", {}).get("expected_revenue")
                if rev:
                    s["revenues"].append(rev)
            else:
                s["denied"] += 1
            price = rec.get("pricing_result", {}).get("final_price_per_pax")
            if price:
                s["prices"].append(price)

        for key, data in summary.items():
            n = data["total"]
            data["acceptance_rate_pct"] = round(100 * data["accepted"] / n, 1) if n else 0
            data["avg_price"] = round(sum(data["prices"]) / len(data["prices"]), 2) if data["prices"] else None
            data["avg_revenue"] = round(sum(data["revenues"]) / len(data["revenues"]), 2) if data["revenues"] else None
            data["total_revenue"] = round(sum(data["revenues"]), 2)

        return summary