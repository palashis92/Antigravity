"""Unified WhatsApp Integration Client for LUMI.

Supports:
1. Meta WhatsApp Cloud API (Official free tier)
2. Twilio WhatsApp API
3. Green API / UltraMsg (Personal WhatsApp account via QR code)
4. Graceful Mock / Simulation fallback for offline/development testing
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

from ..core.logger import get_logger

logger = get_logger("integrations.whatsapp")


class WhatsAppClient:
    """Unified client for sending WhatsApp text messages and PDF documents."""

    def __init__(self) -> None:
        # 1. Meta WhatsApp Cloud API credentials
        self.meta_token = os.getenv("WHATSAPP_API_TOKEN")
        self.meta_phone_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

        # 2. Twilio WhatsApp credentials
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_from = os.getenv("TWILIO_WHATSAPP_NUMBER")

        # 3. Green API credentials (Personal WhatsApp via QR)
        self.green_instance = os.getenv("GREEN_API_INSTANCE_ID") or os.getenv("WHATSAPP_INSTANCE_ID")
        self.green_token = os.getenv("GREEN_API_TOKEN") or os.getenv("WHATSAPP_INSTANCE_TOKEN")

        # Detect active provider
        if self.meta_token and self.meta_phone_id:
            self.provider = "META_CLOUD"
        elif self.twilio_sid and self.twilio_token and self.twilio_from:
            self.provider = "TWILIO"
        elif self.green_instance and self.green_token:
            self.provider = "GREEN_API"
        else:
            self.provider = "SIMULATION"

        logger.info(f"WhatsAppClient initialized using provider: {self.provider}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_text(self, phone: str, message: str) -> Tuple[bool, str]:
        """Send a text message to a WhatsApp phone number.

        Args:
            phone: Target phone number (e.g. '01712345678', '+8801712345678').
            message: Text message content.

        Returns:
            Tuple of (success: bool, status_message: str)
        """
        clean_phone = self.format_phone(phone)
        if not clean_phone:
            return False, f"অবৈধ ফোন নম্বর: '{phone}'"

        if not message.strip():
            return False, "মেসেজ খালি হতে পারে না।"

        if self.provider == "META_CLOUD":
            return self._send_meta_text(clean_phone, message)
        elif self.provider == "TWILIO":
            return self._send_twilio_text(clean_phone, message)
        elif self.provider == "GREEN_API":
            return self._send_green_text(clean_phone, message)
        else:
            # Simulation / Mock Mode
            logger.info(f"📱 [WHATSAPP SIMULATION] To: +{clean_phone}\nMessage: {message}")
            return True, f"(সিমুলেশন) +{clean_phone} নাম্বারে সফলভাবে মেসেজ পাঠানো হয়েছে: \"{message[:40]}...\""

    def send_document(
        self,
        phone: str,
        file_path: str,
        caption: str = "",
    ) -> Tuple[bool, str]:
        """Send a PDF or file document to a WhatsApp phone number.

        Args:
            phone: Target phone number.
            file_path: Absolute or relative file path to the PDF/file.
            caption: Optional caption or title for the document.

        Returns:
            Tuple of (success: bool, status_message: str)
        """
        clean_phone = self.format_phone(phone)
        if not clean_phone:
            return False, f"অবৈধ ফোন নম্বর: '{phone}'"

        target_file = Path(file_path)
        if not target_file.exists():
            return False, f"ডকুমেন্ট ফাইলটি পাওয়া যায়নি: '{file_path}'"

        if self.provider == "GREEN_API":
            return self._send_green_document(clean_phone, str(target_file), caption)
        elif self.provider == "META_CLOUD":
            return self._send_meta_document(clean_phone, str(target_file), caption)
        elif self.provider == "TWILIO":
            return self._send_twilio_document(clean_phone, str(target_file), caption)
        else:
            # Simulation
            logger.info(
                f"📎 [WHATSAPP SIMULATION] Sent document '{target_file.name}' to +{clean_phone} (Caption: {caption})"
            )
            return True, f"(সিমুলেশন) +{clean_phone} নাম্বারে '{target_file.name}' ডকুমেন্টটি সফলভাবে পাঠানো হয়েছে।"

    # ------------------------------------------------------------------
    # Meta Cloud API Handler
    # ------------------------------------------------------------------

    def _send_meta_text(self, phone: str, message: str) -> Tuple[bool, str]:
        url = f"https://graph.facebook.com/v19.0/{self.meta_phone_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.meta_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": phone,
            "type": "text",
            "text": {"body": message},
        }
        try:
            if requests:
                res = requests.post(url, json=payload, headers=headers, timeout=15)
                if res.status_code in (200, 201):
                    logger.info(f"Meta WhatsApp text sent to +{phone}")
                    return True, f"+{phone} নাম্বারে মেসেজ সফলভাবে পাঠানো হয়েছে।"
                return False, f"মেটা এপিআই ত্রুটি: {res.text}"
            else:
                req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
                with urllib.request.urlopen(req, timeout=15) as res:
                    if res.status in (200, 201):
                        return True, f"+{phone} নাম্বারে মেসেজ সফলভাবে পাঠানো হয়েছে।"
                return False, f"মেটা এপিআই ত্রুটি।"
        except Exception as e:
            logger.error(f"Meta WhatsApp request failed: {e}")
            return False, f"মেসেজ পাঠাতে সমস্যা হয়েছে: {e}"

    def _send_meta_document(self, phone: str, file_path: str, caption: str) -> Tuple[bool, str]:
        if not requests:
            return False, "ডকুমেন্ট আপলোড করার জন্য 'requests' প্যাকেজ প্রয়োজন।"

        # Upload media first
        upload_url = f"https://graph.facebook.com/v19.0/{self.meta_phone_id}/media"
        headers = {"Authorization": f"Bearer {self.meta_token}"}
        try:
            with open(file_path, "rb") as f:
                files = {
                    "file": (Path(file_path).name, f, "application/pdf"),
                    "messaging_product": (None, "whatsapp"),
                }
                upload_res = requests.post(upload_url, headers=headers, files=files, timeout=30)
            
            if upload_res.status_code not in (200, 201):
                return False, f"ডকুমেন্ট আপলোড ব্যর্থ: {upload_res.text}"

            media_id = upload_res.json().get("id")

            # Send document message using media_id
            msg_url = f"https://graph.facebook.com/v19.0/{self.meta_phone_id}/messages"
            payload = {
                "messaging_product": "whatsapp",
                "to": phone,
                "type": "document",
                "document": {
                    "id": media_id,
                    "filename": Path(file_path).name,
                    "caption": caption or Path(file_path).name,
                },
            }
            res = requests.post(msg_url, json=payload, headers={"Authorization": f"Bearer {self.meta_token}"}, timeout=15)
            if res.status_code in (200, 201):
                return True, f"+{phone} নাম্বারে PDF ডকুমেন্ট সফলভাবে পাঠানো হয়েছে।"
            return False, f"ডকুমেন্ট পাঠাতে ব্যর্থ: {res.text}"
        except Exception as e:
            logger.error(f"Meta document send failed: {e}")
            return False, f"ডকুমেন্ট পাঠাতে ব্যর্থ: {e}"

    # ------------------------------------------------------------------
    # Green API Handler (Direct personal number via QR)
    # ------------------------------------------------------------------

    def _send_green_text(self, phone: str, message: str) -> Tuple[bool, str]:
        url = f"https://api.green-api.com/waInstance{self.green_instance}/sendMessage/{self.green_token}"
        chat_id = f"{phone}@c.us"
        payload = {"chatId": chat_id, "message": message}
        try:
            if requests:
                res = requests.post(url, json=payload, timeout=15)
                if res.status_code == 200:
                    return True, f"+{phone} নাম্বারে মেসেজ সফলভাবে পৌঁছেছে।"
                return False, f"Green API ত্রুটি: {res.text}"
            else:
                req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=15) as res:
                    if res.status == 200:
                        return True, f"+{phone} নাম্বারে মেসেজ সফলভাবে পৌঁছেছে।"
                return False, f"Green API ত্রুটি।"
        except Exception as e:
            return False, f"মেসেজ পাঠানো ব্যর্থ: {e}"

    def _send_green_document(self, phone: str, file_path: str, caption: str) -> Tuple[bool, str]:
        if not requests:
            return False, "ডকুমেন্ট আপলোড করার জন্য 'requests' প্যাকেজ প্রয়োজন।"

        url = f"https://api.green-api.com/waInstance{self.green_instance}/sendFileByUpload/{self.green_token}"
        chat_id = f"{phone}@c.us"
        filename = Path(file_path).name
        try:
            with open(file_path, "rb") as f:
                files = {"file": (filename, f, "application/pdf")}
                data = {"chatId": chat_id, "caption": caption or filename, "fileName": filename}
                res = requests.post(url, data=data, files=files, timeout=30)
            if res.status_code == 200:
                return True, f"+{phone} নাম্বারে '{filename}' ডকুমেন্টটি পাঠানো হয়েছে।"
            return False, f"ডকুমেন্ট আপলোড ব্যর্থ: {res.text}"
        except Exception as e:
            return False, f"ডকুমেন্ট পাঠানো ব্যর্থ: {e}"

    # ------------------------------------------------------------------
    # Twilio API Handler
    # ------------------------------------------------------------------

    def _send_twilio_text(self, phone: str, message: str) -> Tuple[bool, str]:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_sid}/Messages.json"
        data = {
            "From": f"whatsapp:{self.twilio_from}",
            "To": f"whatsapp:+{phone}",
            "Body": message,
        }
        try:
            if requests:
                res = requests.post(url, data=data, auth=(self.twilio_sid, self.twilio_token), timeout=15)
                if res.status_code in (200, 201):
                    return True, f"+{phone} নাম্বারে হোয়াটসঅ্যাপ মেসেজ পাঠানো হয়েছে।"
                return False, f"Twilio ত্রুটি: {res.text}"
            else:
                auth_val = "Basic " + base64.b64encode(f"{self.twilio_sid}:{self.twilio_token}".encode()).decode()
                encoded_data = urllib.parse.urlencode(data).encode("utf-8")
                req = urllib.request.Request(url, data=encoded_data, headers={"Authorization": auth_val})
                with urllib.request.urlopen(req, timeout=15) as res:
                    if res.status in (200, 201):
                        return True, f"+{phone} নাম্বারে হোয়াটসঅ্যাপ মেসেজ পাঠানো হয়েছে।"
                return False, "Twilio ত্রুটি।"
        except Exception as e:
            return False, f"মেসেজ পাঠাতে সমস্যা: {e}"

    def _send_twilio_document(self, phone: str, file_path: str, caption: str) -> Tuple[bool, str]:
        # Twilio requires a public URL for media attachments
        return False, "Twilio-তে লোকাল PDF পাঠাতে পাবলিক হোস্টেড মিডিয়া ইউআরএল প্রয়োজন।"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def format_phone(raw_phone: str) -> str:
        """Sanitize and standardize phone numbers to international digits-only format."""
        digits = re.sub(r"[^\d]", "", raw_phone)
        if not digits:
            return ""

        # Default Bangladesh prefix if local format (e.g. 01712345678 -> 8801712345678)
        if digits.startswith("01") and len(digits) == 11:
            digits = "88" + digits
        elif digits.startswith("8801") and len(digits) == 13:
            pass  # already standard

        return digits
