
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