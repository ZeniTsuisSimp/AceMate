"""
tracker.py — Weak topic tracker with local JSON persistence.

Tracks quiz performance per topic to identify weak areas that
need more practice. Data is stored in a local ``quiz_scores.json`` file.
"""

import json
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SCORE_FILE = "quiz_scores.json"


class ScoreTracker:
    """Manage per-topic quiz scores with JSON file persistence.

    Attributes:
        filepath: Path to the JSON file storing score data.
        data:     In-memory dict mapping topic → {correct, total}.
    """

    def __init__(self, filepath: str = DEFAULT_SCORE_FILE):
        """Initialize the tracker, loading existing data if available.

        Args:
            filepath: Path to the JSON storage file.
        """
        self.filepath = filepath
        self.data: dict[str, dict[str, int]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load score data from the JSON file - DISABLED FOR PRIVACY."""
        # PRIVACY MODE: Don't load previous quiz scores from disk
        # Start fresh each session
        self.data = {}

    def _save(self) -> None:
        """Persist score data to the JSON file - DISABLED FOR PRIVACY."""
        # PRIVACY MODE: Quiz results are NOT saved to disk
        # Scores exist only in current session memory
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_score(self, topic: str, correct: bool) -> None:
        """Record a quiz attempt for a topic.

        Args:
            topic:   The topic name.
            correct: True if the student answered correctly.
        """
        topic = topic.strip()
        if not topic:
            return

        if topic not in self.data:
            self.data[topic] = {"correct": 0, "total": 0}

        self.data[topic]["total"] += 1
        if correct:
            self.data[topic]["correct"] += 1

        self._save()
        logger.info(
            "Updated score for '%s': %d/%d",
            topic,
            self.data[topic]["correct"],
            self.data[topic]["total"],
        )

    def get_weak_topics(self, min_attempts: int = 1) -> list[dict]:
        """Get topics sorted by lowest score ratio (weakest first).

        Args:
            min_attempts: Minimum number of attempts required to be listed.

        Returns:
            List of dicts: {topic, correct, total, score_pct}, sorted
            ascending by score_pct.
        """
        scored = []
        for topic, stats in self.data.items():
            total = stats.get("total", 0)
            correct = stats.get("correct", 0)
            if total >= min_attempts:
                pct = round((correct / total) * 100, 1) if total > 0 else 0.0
                scored.append({
                    "topic": topic,
                    "correct": correct,
                    "total": total,
                    "score_pct": pct,
                })

        return sorted(scored, key=lambda x: x["score_pct"])

    def get_performance_summary(self) -> dict[str, dict]:
        """Get full performance data for all tracked topics.

        Returns:
            Dict mapping topic → {correct, total, score_pct}.
        """
        summary = {}
        for topic, stats in self.data.items():
            total = stats.get("total", 0)
            correct = stats.get("correct", 0)
            pct = round((correct / total) * 100, 1) if total > 0 else 0.0
            summary[topic] = {
                "correct": correct,
                "total": total,
                "score_pct": pct,
            }
        return summary

    def get_topic_score(self, topic: str) -> dict:
        """Get score for a single topic.

        Args:
            topic: Topic name.

        Returns:
            Dict with correct, total, score_pct (or zeros if not tracked).
        """
        stats = self.data.get(topic.strip(), {"correct": 0, "total": 0})
        total = stats.get("total", 0)
        correct = stats.get("correct", 0)
        pct = round((correct / total) * 100, 1) if total > 0 else 0.0
        return {"correct": correct, "total": total, "score_pct": pct}

    def reset(self) -> None:
        """Clear all tracked scores."""
        self.data = {}
        self._save()
        logger.info("All scores reset.")

    def reset_topic(self, topic: str) -> None:
        """Clear scores for a specific topic.

        Args:
            topic: Topic to reset.
        """
        topic = topic.strip()
        if topic in self.data:
            del self.data[topic]
            self._save()
            logger.info("Scores reset for topic '%s'.", topic)
