"""Tool Calling Registry for Web Search, Weather, Time, and Offline Checks."""

from __future__ import annotations

import datetime
import urllib.request
import json
from typing import Any, Callable, Dict, Optional

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

from ..core.logger import get_logger

logger = get_logger("ai.tools")


class ToolRegistry:
    """Provides tools that LUMI can autonomously invoke during conversation."""

    def __init__(self) -> None:
        self.tools: Dict[str, Callable[..., Any]] = {}
        self.schemas: Dict[str, Dict[str, Any]] = {}
        
        self.register("get_current_datetime", self.get_current_datetime, "Get the current date and time.")
        self.register("check_internet", self.check_internet, "Check if the internet connection is active.")
        self.register("web_search", self.web_search, "Search the web for information.", {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"]
        })
        self.register("get_weather", self.get_weather, "Get the current weather for a city.", {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The city name (e.g. Dhaka)."}},
            "required": ["city"]
        })

    def register(self, name: str, func: Callable[..., Any], description: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        """Dynamically register a new tool with the registry."""
        self.tools[name] = func
        self.schemas[name] = {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": parameters or {"type": "object", "properties": {}}
        }

    def is_online(self) -> bool:
        """Quick check for active internet connection."""
        try:
            urllib.request.urlopen("https://1.1.1.1", timeout=2.0)
            return True
        except Exception:
            return False

    def check_internet(self) -> str:
        """Report current internet status in Bangla."""
        if self.is_online():
            return "ইন্টারনেট সংযোগ চালু রয়েছে।"
        return "বর্তমানে ইন্টারনেট সংযোগ বিচ্ছিন্ন। তবে আমি আমার অফলাইন মেমরি ও স্থানীয় কার্যক্ষমতা দিয়ে সাহায্য করতে পারছি।"

    def get_current_datetime(self) -> str:
        """Return formatted current date and time in Bangla/English."""
        now = datetime.datetime.now()
        return f"আজকের তারিখ: {now.strftime('%d %B, %Y')}, সময়: {now.strftime('%I:%M %p')}"

    def web_search(self, query: str) -> str:
        """Perform a quick web search (or report offline if unavailable)."""
        if not self.is_online():
            return f"ইন্টারনেট সংযোগ না থাকায় '{query}' বিষয়ে সরাসরি অনুসন্ধান করা যাচ্ছে না।"
        
        if not DDGS:
            return "DuckDuckGo search library is not installed."
            
        logger.info(f"Executing web search for: '{query}'")
        try:
            results = DDGS().text(query, max_results=3)
            if not results:
                return f"'{query}' সম্পর্কে কোনো তথ্য পাওয়া যায়নি।"
            
            snippets = [f"- {r.get('title', '')}: {r.get('body', '')}" for r in results]
            return f"'{query}' সম্পর্কে পাওয়া তথ্য:\n" + "\n".join(snippets)
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"অনুসন্ধান করার সময় একটি ত্রুটি হয়েছে: {e}"

    def get_weather(self, city: str = "Dhaka") -> str:
        """Fetch current weather report."""
        if not self.is_online():
            return f"ইন্টারনেট সংযোগ ছাড়া {city}-এর আবহাওয়ার সর্বশেষ তথ্য দেখা সম্ভব হচ্ছে না।"
        
        try:
            req = urllib.request.Request(f"https://wttr.in/{urllib.parse.quote(city)}?format=j1", headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                
                current = data.get("current_condition", [{}])[0]
                temp_c = current.get("temp_C", "অজানা")
                desc = current.get("weatherDesc", [{"value": ""}])[0].get("value", "")
                humidity = current.get("humidity", "অজানা")
                
                # Translating some common conditions roughly for Bangla fallback
                condition_en = desc.lower()
                condition_bn = desc
                if "clear" in condition_en or "sunny" in condition_en: condition_bn = "পরিষ্কার"
                elif "cloud" in condition_en: condition_bn = "মেঘলা"
                elif "rain" in condition_en: condition_bn = "বৃষ্টি"
                
                return f"{city}-তে বর্তমান তাপমাত্রা প্রায় {temp_c}° সেলসিয়াস, আবহাওয়া {condition_bn}, এবং আর্দ্রতা {humidity}%।"
        except Exception as e:
            logger.error(f"Weather fetch error: {e}")
            return f"{city}-এর আবহাওয়ার তথ্য এই মুহূর্তে পাওয়া যাচ্ছে না।"
