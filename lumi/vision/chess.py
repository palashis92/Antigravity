"""Chessboard Vision Service: Board Detection, Piece Identification, and FEN Generation.

Uses Gemini Vision (google-genai SDK) to analyse a chessboard photo and return
the FEN string, which is then fed to Stockfish for best-move analysis.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..core.logger import get_logger

logger = get_logger("vision.chess")


@dataclass
class ChessVisionResult:
    fen_string: str
    is_valid_board: bool
    confidence: float
    white_to_move: bool = True
    active_squares: List[str] = field(default_factory=list)


class ChessVision:
    """Detects physical chessboard grid and generates Forsyth-Edwards Notation (FEN).

    Uses the Gemini Vision API (via google-genai SDK) to interpret a raw camera
    frame of a physical chessboard and return a valid FEN string.  The FEN is
    then passed to Stockfish for engine-level best-move analysis.
    """

    _PROMPT = (
        "You are a chess expert. Look at this photo of a physical chessboard carefully.\n"
        "Return ONLY a JSON object (no markdown, no extra text) with these fields:\n"
        '  "fen": "<standard FEN string of the current board position>",\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "white_to_move": <true or false>\n'
        "Assume the side closest to the camera is White unless the board orientation is clearly different.\n"
        "If the image does not show a chessboard, return: "
        '{"fen": "", "confidence": 0.0, "white_to_move": true}'
    )

    def __init__(self) -> None:
        self._api_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_fen_from_frame(self, frame: Any) -> ChessVisionResult:
        """Process a BGR camera frame and return a FEN + metadata."""
        if frame is None:
            return ChessVisionResult(fen_string="", is_valid_board=False, confidence=0.0)

        # Encode frame to JPEG bytes
        try:
            import cv2
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            image_bytes = buf.tobytes()
        except Exception as e:
            logger.error(f"Frame encoding failed: {e}")
            return ChessVisionResult(fen_string="", is_valid_board=False, confidence=0.0)

        # Ask Gemini to parse the board
        try:
            return self._analyse_with_gemini(image_bytes)
        except Exception as e:
            logger.error(f"Gemini chess analysis failed: {e}")
            return ChessVisionResult(fen_string="", is_valid_board=False, confidence=0.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_api_key(self) -> Optional[str]:
        if not self._api_key:
            self._api_key = os.getenv("GEMINI_API_KEY")
        return self._api_key

    def _analyse_with_gemini(self, image_bytes: bytes) -> ChessVisionResult:
        """Send the image to Gemini Vision and parse the JSON response."""
        api_key = self._get_api_key()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

        import google.generativeai as genai  # type: ignore
        import PIL.Image
        import io

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        pil_image = PIL.Image.open(io.BytesIO(image_bytes))

        response = model.generate_content(
            [self._PROMPT, pil_image],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=256,
            ),
        )

        raw = response.text.strip()
        logger.debug(f"Gemini chess raw response: {raw}")

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)
        fen: str = data.get("fen", "").strip()
        confidence: float = float(data.get("confidence", 0.0))
        white_to_move: bool = bool(data.get("white_to_move", True))

        if not fen:
            logger.warning("Gemini returned empty FEN — board probably not visible.")
            return ChessVisionResult(
                fen_string="", is_valid_board=False, confidence=confidence
            )

        # Validate FEN with python-chess
        try:
            import chess
            board = chess.Board(fen)
            is_valid = board.is_valid()
            white_to_move = board.turn == chess.WHITE
        except Exception:
            is_valid = False

        logger.info(
            f"Chess FEN extracted: '{fen}' | valid={is_valid} | conf={confidence:.2f}"
        )
        return ChessVisionResult(
            fen_string=fen,
            is_valid_board=is_valid,
            confidence=confidence,
            white_to_move=white_to_move,
        )

