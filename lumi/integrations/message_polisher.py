"""AI Message Polisher & Smart Refinement Engine for LUMI.

Transforms rough, informal, or abbreviated spoken words into
eloquent, polite, and context-appropriate messages for WhatsApp.
"""

from __future__ import annotations

import os
from typing import Optional

from ..core.logger import get_logger

logger = get_logger("integrations.message_polisher")


def refine_whatsapp_message(
    raw_text: str,
    recipient_name: str = "",
    relationship: str = "",
) -> str:
    """Enhance and polish spoken user text before sending on WhatsApp.

    Args:
        raw_text: Raw transcription of what the user said (e.g. 'রহিমকে বলো কালকে মিটিং হবে না').
        recipient_name: Name of the person receiving the message.
        relationship: User's relationship to them (e.g. 'friend', 'colleague', 'brother').

    Returns:
        Refined, polite, and articulate message ready to send.
    """
    clean_text = raw_text.strip()
    if not clean_text:
        return ""

    # Don't polish if it's already a very formal or short message
    if len(clean_text) < 4:
        return clean_text

    system_instruction = (
        "You are LUMI's expert interpersonal communication assistant. "
        "The user wants to send a WhatsApp message to someone. The user dictated "
        "their message orally, so it may be rough, brief, colloquial, or slightly blunt.\n\n"
        "Your task is to rewrite and polish the message into natural, polite, respectful, "
        "and well-structured Bengali (বাংলা) (or English if the user dictated in English).\n\n"
        "Guidelines:\n"
        "1. Keep the exact factual meaning and core request intact.\n"
        "2. Add appropriate warm/respectful greeting (e.g., 'আসসালামু আলাইকুম [নাম] ভাই/আপু,')"
        " and polite closing (e.g., 'ধন্যবাদ।').\n"
        "3. Adjust tone based on relationship (casual & friendly if close friend/brother, "
        "professional & courteous if colleague, client, or elder).\n"
        "4. DO NOT invent extra facts or false promises.\n"
        "5. Output ONLY the polished final message text without any explanations or quotation marks."
    )

    context_info = f"প্রাপক (Recipient): {recipient_name or 'সম্মানিত ব্যক্তি'}\n"
    if relationship:
        context_info += f"সম্পর্ক (Relationship): {relationship}\n"
    user_prompt = f"{context_info}মূল বক্তব্য (Raw dictated text):\n\"{clean_text}\"\n\nপরিমার্জিত হোয়াটসঅ্যাপ বার্তা:"

    # 1. Try Gemini
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                ),
            )
            if resp and resp.text:
                polished = resp.text.strip().strip('"').strip("'")
                logger.info(f"AI polished message: '{clean_text}' -> '{polished}'")
                return polished
        except Exception as e:
            logger.debug(f"Gemini message polishing error: {e}")

    # 2. Try OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            if resp and resp.choices:
                polished = resp.choices[0].message.content.strip().strip('"').strip("'")
                logger.info(f"AI polished message (OpenAI): '{clean_text}' -> '{polished}'")
                return polished
        except Exception as e:
            logger.debug(f"OpenAI message polishing error: {e}")

    # Fallback: return original text
    return clean_text
