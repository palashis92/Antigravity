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
from typing import Optional

from ..core.logger import get_logger

logger = get_logger("mem0_cloud")


class Mem0CloudEngine:
    """
    Integrates with the official Mem0 Cloud API.
    Bypasses the need for heavy local dependencies on the Raspberry Pi.
    """

    BASE_URL = "https://api.mem0.ai/v1/memories/"

    def __init__(self):
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
        if not self.api_key:
            return

        thread = threading.Thread(
            target=self._add_memory_sync,
            args=(person_id, user_text, ai_text),
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
        """Directly adds a factual memory for a person in Mem0 Cloud API."""
        if not self.api_key:
            return False
        with self._lock:
            try:
                payload = {
                    "messages": [
                        {"role": "user", "content": f"Remember this fact: {fact}"},
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
            except Exception as e:
                logger.error(f"Mem0 Cloud API remember_fact_sync Error: {e}")
                return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_memory_sync(self, person_id: str, user_text: str, ai_text: str) -> None:
        with self._lock:
            try:
                payload = {
                    "messages": [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": ai_text},
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
                    logger.info(
                        f"Mem0 Cloud API saved memory for {person_id}. Response: {res}"
                    )
            except urllib.error.URLError as e:
                logger.error(f"Mem0 Cloud API Connection Error: {e}")
            except Exception as e:
                logger.error(f"Mem0 Cloud API Error: {e}")