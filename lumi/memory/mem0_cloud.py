"""
Mem0 Cloud API Engine for LUMI.

Uses the official Mem0 managed platform (api.mem0.ai) to store and
retrieve advanced semantic memories without local pip dependencies.
"""

import json
import os
import threading
import urllib.request
import urllib.error
from typing import Any, Optional

from ..core.logger import get_logger

logger = get_logger("mem0_cloud")


class Mem0CloudEngine:
    """
    Integrates with the official Mem0 Cloud API.
    Bypasses the need for heavy local dependencies on the Raspberry Pi.
    """

    BASE_URL = "https://api.mem0.ai/v1/memories/"

    def __init__(self, memory: Optional[Any] = None) -> None:
        self.memory = memory
        self.api_key = os.environ.get("MEM0_API_KEY")
        if not self.api_key:
            logger.warning(
                "Mem0CloudEngine: MEM0_API_KEY is missing. Mem0 API will not work."
            )
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public Interface (matches LumiMem0Engine signature)
    # ------------------------------------------------------------------

    def process_conversation_turn_async(
        self, person_id: str, person_name: str, user_text: str, ai_text: str
    ) -> None:
        """Sends the conversation turn to Mem0 Cloud API for memory extraction."""
        u = (user_text or "").strip()
        if not u:
            return

        # 1. Deterministic extraction to local SQLite
        if self.memory and hasattr(self.memory, "remember_fact"):
            try:
                from .mem0_engine import LumiMem0Engine
                LumiMem0Engine._extract_deterministic_facts(self, person_id, person_name, u)
            except Exception as e:
                logger.debug(f"Local deterministic extraction in Mem0CloudEngine: {e}")

        if not self.api_key:
            return

        thread = threading.Thread(
            target=self._add_memory_sync,
            args=(person_id, u, (ai_text or "").strip()),
            daemon=True,
            name=f"Mem0Cloud_Worker_{person_id}",
        )
        thread.start()

    def recall_facts_sync(
        self,
        person_id: str,
        query: str = "What are the most important facts about this user?",
    ) -> str:
        """Searches Mem0 Cloud API for relevant memories."""
        if not self.api_key:
            return ""

        try:
            url = f"{self.BASE_URL}search/"
            payload = {"query": query, "user_id": person_id}
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=5) as response:
                res = json.loads(response.read().decode())
                memories = [m.get("memory", "") for m in res if m.get("memory")]
                return ", ".join(memories[:5]) if memories else ""
        except (TimeoutError, urllib.error.URLError) as e:
            logger.warning(f"Mem0 Cloud API search timed out or unreachable: {e}")
            return ""
        except Exception as e:
            logger.warning(f"Mem0 Cloud API Search Error: {e}")
            return ""

    def remember_fact_sync(self, person_id: str, fact: str) -> bool:
        """Directly adds a factual memory for a person in Mem0 Cloud API and local SQLite."""
        if self.memory and hasattr(self.memory, "remember_fact"):
            try:
                self.memory.remember_fact(fact_text=fact, person_id=person_id)
            except Exception:
                pass

        if not self.api_key or not fact or not fact.strip():
            return False
        with self._lock:
            try:
                payload = {
                    "messages": [
                        {"role": "user", "content": f"Remember this fact: {fact.strip()}"},
                    ],
                    "user_id": person_id,
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    self.BASE_URL,
                    data=data,
                    headers={
                        "Authorization": f"Token {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    res = json.loads(response.read().decode())
                    logger.info(f"Mem0 Cloud API direct fact saved for {person_id}: {res}")
                    return True
            except urllib.error.HTTPError as e:
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8", errors="ignore")
                except Exception:
                    pass
                logger.warning(f"Mem0 Cloud API HTTP {e.code} ({e.reason}): {err_body}")
                return False
            except Exception as e:
                logger.warning(f"Mem0 Cloud API remember_fact_sync Error: {e}")
                return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_memory_sync(self, person_id: str, user_text: str, ai_text: str) -> None:
        if not user_text or not user_text.strip():
            return

        with self._lock:
            try:
                messages = [{"role": "user", "content": user_text.strip()}]
                if ai_text and ai_text.strip():
                    messages.append({"role": "assistant", "content": ai_text.strip()})

                payload = {
                    "messages": messages,
                    "user_id": person_id,
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    self.BASE_URL,
                    data=data,
                    headers={
                        "Authorization": f"Token {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )

                with urllib.request.urlopen(req, timeout=10) as response:
                    res = json.loads(response.read().decode())
                    logger.info(
                        f"Mem0 Cloud API saved memory for {person_id}. Response: {res}"
                    )
            except urllib.error.HTTPError as e:
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8", errors="ignore")
                except Exception:
                    pass
                logger.warning(f"Mem0 Cloud API HTTP {e.code} ({e.reason}): {err_body}")
            except urllib.error.URLError as e:
                logger.warning(f"Mem0 Cloud API Connection Error: {e}")
            except Exception as e:
                logger.warning(f"Mem0 Cloud API Error: {e}")