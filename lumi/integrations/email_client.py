"""Unified Email Client Integration for LUMI.

Supports:
1. SMTP with TLS / STARTTLS (Gmail, Outlook, Yahoo, SendGrid, custom SMTP servers).
2. Optional file / PDF attachments.
3. Graceful Mock / Simulation fallback when SMTP credentials are not configured.
"""

from __future__ import annotations

import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional, Tuple

from ..core.logger import get_logger

logger = get_logger("integrations.email")


def _load_dotenv_if_needed() -> None:
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = val
        except Exception:
            pass


class EmailClient:
    """Unified client for sending emails with optional attachments."""

    def __init__(self) -> None:
        _load_dotenv_if_needed()
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "").strip() or None
        raw_pw = os.getenv("SMTP_PASSWORD")
        self.smtp_password = raw_pw.replace(" ", "").strip() if raw_pw else None
        self.smtp_from = os.getenv("SMTP_FROM") or self.smtp_user or "lumi@robot.local"
        self.use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")

        if self.smtp_user and self.smtp_password:
            self.mode = "SMTP"
        else:
            self.mode = "SIMULATION"

        logger.info(f"EmailClient initialized in mode: {self.mode} (Host: {self.smtp_host}:{self.smtp_port})")

    @property
    def is_configured(self) -> bool:
        """True if real SMTP credentials are provided in environment."""
        return self.mode == "SMTP"

    def send_email(
        self,
        to_address: str,
        subject: str,
        message: str,
        attachment_path: Optional[str] = None,
        html_message: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Send an email to a target recipient.

        Args:
            to_address: Recipient email address (e.g. 'john@example.com').
            subject: Subject line of the email.
            message: Plain text body of the email.
            attachment_path: Optional path to a file or PDF to attach.
            html_message: Optional HTML formatted body.

        Returns:
            Tuple of (success: bool, status_message: str)
        """
        clean_to = to_address.strip()
        if not clean_to or "@" not in clean_to:
            return False, f"অবৈধ ইমেইল ঠিকানা: '{to_address}'"

        if not subject.strip():
            subject = "LUMI Robot Notification"

        # -------------------------------------------------------------
        # Mode: Simulation / Mock fallback
        # -------------------------------------------------------------
        if self.mode != "SMTP":
            logger.info(
                f"📧 [EMAIL SIMULATION] To: {clean_to} | Subject: '{subject}'\n"
                f"Message: {message}\n"
                f"Attachment: {attachment_path or 'None'}"
            )
            return True, f"(সিমুলেশন) {clean_to} ঠিকানায় ইমেইল পাঠানো হয়েছে: '{subject}'"

        # -------------------------------------------------------------
        # Mode: Real SMTP Transmission
        # -------------------------------------------------------------
        try:
            msg = MIMEMultipart("mixed")
            msg["From"] = f"LUMI AI Robot <{self.smtp_from}>"
            msg["To"] = clean_to
            msg["Subject"] = subject

            # Body multipart
            body_part = MIMEMultipart("alternative")
            body_part.attach(MIMEText(message, "plain", "utf-8"))
            if html_message:
                body_part.attach(MIMEText(html_message, "html", "utf-8"))
            msg.attach(body_part)

            # Attachment handling
            if attachment_path:
                att_file = Path(attachment_path)
                if att_file.exists():
                    with open(att_file, "rb") as f:
                        part = MIMEApplication(f.read(), Name=att_file.name)
                    part["Content-Disposition"] = f'attachment; filename="{att_file.name}"'
                    msg.attach(part)
                else:
                    logger.warning(f"Attachment file not found: {attachment_path}")

            # Send via SMTP
            server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15)
            if self.use_tls:
                server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
            server.quit()

            logger.info(f"Email successfully sent to {clean_to} (Subject: '{subject}')")
            return True, f"{clean_to} ঠিকানায় সফলভাবে ইমেইল পাঠানো হয়েছে।"

        except Exception as e:
            err_msg = f"ইমেইল পাঠাতে ব্যর্থ হয়েছে: {e}"
            logger.error(err_msg, exc_info=True)
            return False, err_msg
