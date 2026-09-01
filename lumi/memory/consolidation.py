"""Memory Consolidation Daemon for LUMI.

Periodically cleans, deduplicates, and scores memories to keep
the knowledge base healthy and relevant.

Runs as a background thread, triggered on startup and then
every 6 hours.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Set

from ..core.logger import get_logger

if TYPE_CHECKING:
    from ..memory.manager import MemoryManager
    from ..memory.models import Fact

logger = get_logger("memory.consolidation")


class MemoryConsolidator:
    """Background daemon that maintains memory health.

    Responsibilities:
    1. Deduplication: Find near-duplicate facts and remove older copies
    2. Staleness decay: Reduce confidence of very old facts
    3. Orphan cleanup: Remove facts with no person_id and no useful content
    """

    # Run every 6 hours
    INTERVAL_SECONDS = 6 * 60 * 60
    # Facts older than this many days get confidence decay
    STALENESS_THRESHOLD_DAYS = 90
    # Confidence decay multiplier for stale facts
    STALENESS_DECAY = 0.7
    # Jaccard similarity threshold for deduplication
    DEDUP_SIMILARITY_THRESHOLD = 0.75
    # Minimum confidence before a fact is auto-deleted
    MIN_CONFIDENCE = 0.1

    def __init__(self, memory: MemoryManager) -> None:
        self.memory = memory
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the consolidation daemon."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._daemon_loop,
            daemon=True,
            name="MemoryConsolidator",
        )
        self._thread.start()
        logger.info("MemoryConsolidator daemon started.")

    def stop(self) -> None:
        """Stop the consolidation daemon."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("MemoryConsolidator daemon stopped.")

    def _daemon_loop(self) -> None:
        """Main loop: run consolidation immediately, then every INTERVAL_SECONDS."""
        # Run once immediately on startup
        self.run_consolidation()

        while self._running:
            # Sleep in small increments so we can stop quickly
            for _ in range(int(self.INTERVAL_SECONDS / 10)):
                if not self._running:
                    return
                time.sleep(10)
            self.run_consolidation()

    def run_consolidation(self) -> Dict[str, int]:
        """Execute a full consolidation pass.

        Returns:
            Stats dict with counts of actions taken.
        """
        stats = {"deduped": 0, "decayed": 0, "deleted": 0}

        try:
            all_facts = self.memory.recall_facts()
            if not all_facts:
                logger.debug("Consolidation: No facts to process.")
                return stats

            logger.info(f"Consolidation: Processing {len(all_facts)} facts...")

            # Group by person for person-scoped dedup
            by_person: Dict[str | None, List] = defaultdict(list)
            for f in all_facts:
                by_person[f.person_id].append(f)

            for person_id, facts in by_person.items():
                # 1. Deduplication
                deduped = self._dedup_facts(facts)
                stats["deduped"] += deduped

                # 2. Staleness decay
                decayed = self._apply_staleness_decay(facts)
                stats["decayed"] += decayed

                # 3. Auto-delete ultra-low-confidence facts
                deleted = self._cleanup_low_confidence(facts)
                stats["deleted"] += deleted

            logger.info(
                f"Consolidation complete: "
                f"{stats['deduped']} deduped, "
                f"{stats['decayed']} decayed, "
                f"{stats['deleted']} deleted."
            )

        except Exception as e:
            logger.error(f"Consolidation error: {e}", exc_info=True)

        return stats

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _dedup_facts(self, facts: List) -> int:
        """Remove near-duplicate facts, keeping the newest version."""
        if len(facts) < 2:
            return 0

        removed = 0
        # Sort by created_at DESC so we keep newer facts
        facts_sorted = sorted(facts, key=lambda f: f.created_at, reverse=True)
        keep_ids: Set[str] = set()
        delete_ids: Set[str] = set()

        for i, fact_a in enumerate(facts_sorted):
            if fact_a.id in delete_ids:
                continue
            keep_ids.add(fact_a.id)

            for fact_b in facts_sorted[i + 1 :]:
                if fact_b.id in delete_ids or fact_b.id in keep_ids:
                    continue
                sim = self._jaccard_similarity(
                    fact_a.fact_text.lower(), fact_b.fact_text.lower()
                )
                if sim >= self.DEDUP_SIMILARITY_THRESHOLD:
                    delete_ids.add(fact_b.id)

        for fid in delete_ids:
            try:
                self.memory.forget_fact(fid)
                removed += 1
            except Exception as e:
                logger.warning(f"Failed to delete duplicate fact {fid}: {e}")

        return removed

    # ------------------------------------------------------------------
    # Staleness Decay
    # ------------------------------------------------------------------

    def _apply_staleness_decay(self, facts: List) -> int:
        """Reduce confidence of facts older than STALENESS_THRESHOLD_DAYS."""
        now = datetime.now()
        decayed = 0

        for f in facts:
            try:
                created = datetime.fromisoformat(f.created_at)
                age_days = (now - created).days
            except (ValueError, TypeError):
                continue

            if age_days > self.STALENESS_THRESHOLD_DAYS and f.confidence > self.MIN_CONFIDENCE:
                new_confidence = max(f.confidence * self.STALENESS_DECAY, self.MIN_CONFIDENCE)
                if new_confidence != f.confidence:
                    try:
                        self.memory.db.execute_write(
                            "UPDATE facts SET confidence = ?, updated_at = ? WHERE id = ?",
                            (new_confidence, datetime.now().isoformat(), f.id),
                        )
                        decayed += 1
                    except Exception as e:
                        logger.warning(f"Failed to decay fact {f.id}: {e}")

        return decayed

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_low_confidence(self, facts: List) -> int:
        """Delete facts with confidence below MIN_CONFIDENCE."""
        deleted = 0
        for f in facts:
            if f.confidence <= self.MIN_CONFIDENCE:
                try:
                    self.memory.forget_fact(f.id)
                    deleted += 1
                except Exception:
                    pass
        return deleted

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """Word-level Jaccard similarity between two strings."""
        words_a = set(text_a.split())
        words_b = set(text_b.split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)
