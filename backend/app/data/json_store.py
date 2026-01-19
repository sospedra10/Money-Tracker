from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


DEFAULT_CATEGORIES: list[str] = [
    "Bank",
    "Remuneration Account",
    "ETFs and Stocks",
    "Real Estate",
    "Crypto",
    "Reenlever",
    "Staking",
    "Others",
]

INVESTMENT_CATEGORIES: list[str] = [
    "Remuneration Account",
    "Real Estate",
    "ETFs and Stocks",
    "Bank",
    "Reenlever",
    "Staking",
]

NON_INVESTMENT_CATEGORIES: list[str] = ["Crypto", "Others"]

RISK_LIQUIDITY_PROFILE: dict[str, dict[str, float]] = {
    "Bank": {"risk": 1, "liquidity": 5, "expected_return": 0.00},
    "Remuneration Account": {"risk": 1, "liquidity": 4, "expected_return": 0.02},
    "ETFs and Stocks": {"risk": 4, "liquidity": 4, "expected_return": 0.065},
    "Real Estate": {"risk": 3, "liquidity": 2, "expected_return": 0.11},
    "Crypto": {"risk": 5, "liquidity": 3, "expected_return": 0.15},
    "Reenlever": {"risk": 4, "liquidity": 4, "expected_return": 0.11},
    "Staking": {"risk": 3, "liquidity": 3, "expected_return": 0.03},
    "Others": {"risk": 2, "liquidity": 3, "expected_return": 0.02},
}

VOLATILITY_ASSUMPTIONS: dict[str, float] = {
    "Bank": 0.01,
    "Remuneration Account": 0.02,
    "ETFs and Stocks": 0.18,
    "Real Estate": 0.12,
    "Crypto": 0.60,
    "Reenlever": 0.18,
    "Staking": 0.25,
    "Others": 0.10,
}


DateFormat = Literal["legacy", "iso"]


@dataclass(frozen=True)
class HistoryEntry:
    date: datetime
    category: str
    amount: float


class JsonStore:
    def __init__(
        self,
        data_file: Path,
        *,
        categories: list[str] | None = None,
        date_format: DateFormat = "legacy",
    ) -> None:
        self._data_file = Path(data_file)
        self._categories = categories or DEFAULT_CATEGORIES
        self._date_format = date_format
        self._lock = threading.Lock()

        self._ensure_file()

    @property
    def categories(self) -> list[str]:
        return list(self._categories)

    def _ensure_file(self) -> None:
        self._data_file.parent.mkdir(parents=True, exist_ok=True)
        if self._data_file.exists():
            return
        self._atomic_write({"history": []})

    def _parse_date(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if not isinstance(value, str):
            raise ValueError("Invalid date")
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return datetime.fromisoformat(value)

    def _format_date(self, value: datetime) -> str:
        if self._date_format == "iso":
            return value.isoformat(timespec="seconds")
        return value.strftime("%Y-%m-%d %H:%M:%S")

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        tmp_path = self._data_file.with_suffix(self._data_file.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self._data_file)

    def load_raw(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_file()
            with open(self._data_file, "r", encoding="utf-8") as handle:
                return json.load(handle)

    def _load_raw_unlocked(self) -> dict[str, Any]:
        self._ensure_file()
        with open(self._data_file, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_history(self) -> list[HistoryEntry]:
        raw = self.load_raw()
        history = raw.get("history") or []
        entries: list[HistoryEntry] = []
        for item in history:
            try:
                entries.append(
                    HistoryEntry(
                        date=self._parse_date(item.get("date")),
                        category=str(item.get("category")),
                        amount=float(item.get("amount")),
                    )
                )
            except Exception:
                continue
        return entries

    def append_entry(self, entry: HistoryEntry) -> None:
        if entry.category not in self._categories:
            raise ValueError(f"Unknown category: {entry.category}")
        if entry.amount < 0:
            raise ValueError("Amount must be >= 0")

        with self._lock:
            data = self._load_raw_unlocked()
            history = data.setdefault("history", [])
            history.append(
                {
                    "date": self._format_date(entry.date),
                    "category": entry.category,
                    "amount": float(entry.amount),
                }
            )
            self._atomic_write(data)
