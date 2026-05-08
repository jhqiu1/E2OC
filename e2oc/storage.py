"""Operator storage and history management for E2OC.

Reference:
    Qiu et al., "Evolving Interdependent Operators with Large Language Models
    for Multi-Objective Combinatorial Optimization", ICML 2026.
"""

from __future__ import annotations
import os
import json
import time
import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


@dataclass
class Batch:
    """Complete information for one EoH run batch."""

    operator: str
    cycle: Optional[int]
    ts: float
    n_samples: int
    template: Optional[str] = None
    individuals: List[Dict[str, Any]] = field(default_factory=list)
    valid_n: int = 0
    invalid_n: int = 0
    valid_rate: float = 0.0
    best_score: Optional[float] = None
    worst_score: Optional[float] = None
    mean_score: Optional[float] = None
    median_score: Optional[float] = None
    std_score: Optional[float] = None
    q_low: Optional[float] = None
    q_high: Optional[float] = None
    best_idx: Optional[int] = None


def _stats(scores: np.ndarray, q_alpha: float = 0.10) -> Dict[str, Optional[float]]:
    if scores.size == 0:
        return {
            "best": None, "worst": None, "mean": None, "median": None,
            "std": None, "q_low": None, "q_high": None, "best_idx": None,
        }

    best_idx = int(np.argmax(scores))
    return {
        "best": float(scores.max()),
        "worst": float(scores.min()),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "std": float(scores.std(ddof=1)) if scores.size > 1 else 0.0,
        "q_low": float(np.quantile(scores, q_alpha / 2.0)),
        "q_high": float(np.quantile(scores, 1.0 - q_alpha / 2.0)),
        "best_idx": best_idx,
    }


def _pct_gain(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return (new - old) / abs(old)


class Storages:
    """Minimal operator-based storage with template support.

    Tracks operator performance history, template changes, and provides
    statistics for MCTS-based strategy selection.

    Args:
        operators: List of operator names.
        q_alpha: Quantile interval.
        save_path: If provided, auto-save after each record operation.
        ucb_beta: UCB exploration coefficient for history statistics.
    """

    def __init__(
        self,
        operators: List[str],
        q_alpha: float = 0.10,
        save_path: Optional[str] = None,
        ucb_beta: float = 0.0,
    ):
        self.q_alpha = q_alpha
        self.save_path = save_path
        self.ucb_beta = float(ucb_beta)

        self.latest: Dict[str, Optional[Batch]] = {op: None for op in operators}
        self.history_scores: Dict[str, List[float]] = {op: [] for op in operators}
        self.history_counts: Dict[str, int] = {op: 0 for op in operators}
        self.history_valid_counts: Dict[str, int] = {op: 0 for op in operators}

        self.current_template: Dict[str, Optional[str]] = {op: None for op in operators}
        self.templates_seen: Dict[str, List[str]] = {op: [] for op in operators}
        self.latest_by_template: Dict[str, Dict[str, Optional[Batch]]] = {op: {} for op in operators}
        self.template_use_counts: Dict[str, Dict[str, int]] = {op: {} for op in operators}
        self.template_change_log: Dict[str, List[Dict[str, Any]]] = {op: [] for op in operators}

    def set_template(self, operator: str, cycle: Optional[int], template: Optional[str]) -> None:
        self._ensure_op_exists(operator)
        self.current_template[operator] = template
        ts = time.time()
        self.template_change_log[operator].append(
            {"cycle": cycle, "ts": ts, "template": template}
        )
        if template and template not in self.templates_seen[operator]:
            self.templates_seen[operator].append(template)

    def record_batch(
        self,
        operator: str,
        individuals: List[Tuple[str, float]] | List[Dict[str, Any]],
        cycle: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._ensure_op_exists(operator)
        normalized: List[Dict[str, Any]] = []

        for item in individuals:
            if isinstance(item, dict):
                code = str(item.get("code", ""))
                raw = item.get("score", float("-inf"))
            else:
                code, raw = item
                code, raw = str(code), raw

            try:
                score = float(raw)
            except Exception:
                score = float("-inf")

            valid = math.isfinite(score) and score != float("-inf")
            adjusted_score = score if valid else 0.0

            normalized.append({
                "code": code, "score": adjusted_score,
                "raw_score": score, "valid": valid,
            })

        return self._finalize_record(operator, normalized, cycle)

    def history(self, operator: str) -> Dict[str, Any]:
        arr = np.array(self.history_scores.get(operator, []), dtype=float)
        stats = _stats(arr, self.q_alpha)
        total = int(self.history_counts.get(operator, 0))
        valid = int(self.history_valid_counts.get(operator, 0))
        valid_rate = (valid / total) if total > 0 else None

        ucb = lcb = None
        if self.ucb_beta > 0 and stats["mean"] is not None:
            denominator = max(np.sqrt(arr.size), 1e-8)
            radius = self.ucb_beta * (
                stats["std"] / denominator if stats["std"] is not None else 0.0
            )
            ucb = stats["mean"] + radius
            lcb = stats["mean"] - radius

        return {
            "operator": operator,
            "n_individuals": int(arr.size),
            "best": stats["best"], "worst": stats["worst"],
            "mean": stats["mean"], "median": stats["median"],
            "std": stats["std"], "q_low": stats["q_low"], "q_high": stats["q_high"],
            "ucb": ucb, "lcb": lcb,
            "valid_count": valid, "total_count": total, "valid_rate": valid_rate,
        }

    def dump(self, path: Optional[str] = None) -> None:
        path = path or self.save_path
        if not path:
            return

        data = {
            "q_alpha": self.q_alpha, "ucb_beta": self.ucb_beta,
            "history_scores": self.history_scores,
            "history_counts": self.history_counts,
            "history_valid_counts": self.history_valid_counts,
            "latest": {
                op: (asdict(batch) if batch else None)
                for op, batch in self.latest.items()
            },
            "current_template": self.current_template,
            "templates_seen": self.templates_seen,
            "latest_by_template": {
                op: {t: (asdict(batch) if batch else None) for t, batch in tb.items()}
                for op, tb in self.latest_by_template.items()
            },
            "template_use_counts": self.template_use_counts,
            "template_change_log": self.template_change_log,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        if not os.path.isfile(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.q_alpha = float(data.get("q_alpha", self.q_alpha))
        self.ucb_beta = float(data.get("ucb_beta", self.ucb_beta))
        self.history_scores = {k: list(v) for k, v in data.get("history_scores", {}).items()}
        self.history_counts = {k: int(v) for k, v in data.get("history_counts", {}).items()}
        self.history_valid_counts = (
            {k: int(v) for k, v in data.get("history_valid_counts", {}).items()}
            if "history_valid_counts" in data
            else {k: 0 for k in self.history_counts.keys()}
        )

        raw_latest = data.get("latest", {})
        new_latest: Dict[str, Optional[Batch]] = {}
        for op, batch_data in raw_latest.items():
            if batch_data:
                batch_dict = dict(batch_data)
                if "step" in batch_dict:
                    batch_dict.pop("step", None)
                batch_dict.setdefault("template", None)
                new_latest[op] = Batch(**batch_dict)
            else:
                new_latest[op] = None
        self.latest = new_latest

        self.current_template = {k: v for k, v in data.get("current_template", {}).items()}
        self.templates_seen = {k: list(v) for k, v in data.get("templates_seen", {}).items()}

        self.latest_by_template = {}
        for op, template_batches in data.get("latest_by_template", {}).items():
            self.latest_by_template[op] = {}
            for template, batch_data in (template_batches or {}).items():
                if batch_data:
                    batch_dict = dict(batch_data)
                    if "step" in batch_dict:
                        batch_dict.pop("step", None)
                    batch_dict.setdefault("template",
                        template if batch_dict.get("template") is None else batch_dict.get("template"))
                    self.latest_by_template[op][template] = Batch(**batch_dict)
                else:
                    self.latest_by_template[op][template] = None

        self.template_use_counts = {
            op: {t: int(c) for t, c in (counts or {}).items()}
            for op, counts in data.get("template_use_counts", {}).items()
        }
        self.template_change_log = {
            op: list(log) for op, log in data.get("template_change_log", {}).items()
        }

        for op in set(list(self.latest.keys()) + list(self.history_scores.keys())):
            self._ensure_op_exists(op)

    def _finalize_record(self, operator, individuals, cycle):
        scores = (
            np.array([x["score"] for x in individuals], dtype=float)
            if individuals else np.array([], dtype=float)
        )
        valid_n = int(sum(bool(x.get("valid", False)) for x in individuals))
        total_n = int(scores.size)
        invalid_n = total_n - valid_n
        valid_rate = (valid_n / total_n) if total_n > 0 else 0.0

        stats = _stats(scores, self.q_alpha)
        template = self.current_template.get(operator)

        batch = Batch(
            operator=operator, cycle=cycle, ts=time.time(), n_samples=total_n,
            template=template, individuals=individuals,
            valid_n=valid_n, invalid_n=invalid_n, valid_rate=valid_rate,
            best_score=stats["best"], worst_score=stats["worst"],
            mean_score=stats["mean"], median_score=stats["median"],
            std_score=stats["std"], q_low=stats["q_low"], q_high=stats["q_high"],
            best_idx=stats["best_idx"],
        )

        self.latest[operator] = batch

        if total_n > 0:
            self.history_scores[operator].extend(scores.tolist())
            self.history_counts[operator] += total_n
            self.history_valid_counts[operator] += valid_n

        if template:
            self.latest_by_template[operator][template] = batch
            self.template_use_counts[operator][template] = (
                self.template_use_counts[operator].get(template, 0) + 1
            )
            if template not in self.templates_seen[operator]:
                self.templates_seen[operator].append(template)

        hist = self.history(operator)
        diff = {
            "mean_gain": _pct_gain(batch.mean_score, hist["mean"]),
            "best_gain": _pct_gain(batch.best_score, hist["best"]),
            "median_gain": _pct_gain(batch.median_score, hist["median"]),
        }

        if self.save_path:
            self.dump(self.save_path)

        return {"latest": asdict(batch), "history": hist, "diff": diff}

    def _ensure_op_exists(self, operator: str) -> None:
        for container in [self.latest, self.history_scores, self.history_counts, self.history_valid_counts]:
            if operator not in container:
                if container is self.latest:
                    container[operator] = None
                elif container in (self.history_counts, self.history_valid_counts):
                    container[operator] = 0
                else:
                    container[operator] = []

        if operator not in self.current_template:
            self.current_template[operator] = None
        if operator not in self.templates_seen:
            self.templates_seen[operator] = []
        if operator not in self.latest_by_template:
            self.latest_by_template[operator] = {}
        if operator not in self.template_use_counts:
            self.template_use_counts[operator] = {}
        if operator not in self.template_change_log:
            self.template_change_log[operator] = []
