"""Chessboard Vision Service: Board Detection, Piece Identification, and FEN Generation."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, List, Optional

from ..core.logger import get_logger

logger = get_logger("vision.chess")


@dataclass
class ChessVisionResult:
    fen_string: str
    is_valid_board: bool
    confidence: float
    white_to_move: bool = True
    active_squares: List[str] = None


class ChessVision:
    """Detects physical chessboard grid and generates Forsyth-Edwards Notation (FEN)."""

    def __init__(self) -> None:
        self.model_path = os.path.join("data", "models", "chess_pieces.tflite")

    def extract_fen_from_frame(self, frame: Any) -> ChessVisionResult:
        """Process image frame of a physical chessboard and generate FEN."""
        if frame is None:
            return ChessVisionResult(
                fen_string="",
                is_valid_board=False,
                confidence=0.0,
            )

        try:
            import cv2
            import numpy as np
            import chess
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Find 7x7 inner corners of an 8x8 chessboard
            ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
            
            if ret:
                # Board found, warp to top-down view
                # Simple approximation for 8x8 corners logic
                # For a full 8x8 squares, we need the outer corners, but findChessboardCorners gives inner.
                # Here we just extrapolate or use the bounding rect.
                
                # Using Gemini as cloud fallback due to complexity of reliable physical board extraction
                pass
                
        except Exception as e:
            logger.error(f"Error in OpenCV board detection: {e}")

        # Cloud fallback for FEN generation
        try:
            import cv2
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            
            if api_key:
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
                
                _, buffer = cv2.imencode('.jpg', frame)
                base64_image = base64.b64encode(buffer).decode('utf-8')
                
                response = client.chat.completions.create(
                    model="gemini-1.5-flash",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Analyze this chessboard and return ONLY a JSON object with 'fen' (the standard FEN string) and 'confidence' (float 0.0 to 1.0). Assume it is White's turn to move."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                )
                
                response_text = response.choices[0].message.content
                import re
                import chess
                json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1))
                else:
                    data = json.loads(response_text)
                    
                fen = data.get("fen", "")
                confidence = float(data.get("confidence", 0.9))
                
                board = chess.Board(fen)
                return ChessVisionResult(
                    fen_string=fen,
                    is_valid_board=board.is_valid(),
                    confidence=confidence,
                    white_to_move=board.turn == chess.WHITE
                )
        except Exception as e:
            logger.error(f"Error in Cloud Fallback for Chess: {e}")

        return ChessVisionResult(
            fen_string="",
            is_valid_board=False,
            confidence=0.0
        )
