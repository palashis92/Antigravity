"""Stockfish UCI Engine Interface and Bangla Move Explanation."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Optional

from ..core.logger import get_logger

logger = get_logger("chess.stockfish")


@dataclass
class BestMoveEvaluation:
    fen: str
    best_move: str
    evaluation_score: float
    explanation_bn: str
    is_engine_available: bool = True


class ChessAnalysisEngine:
    """Interfaces with Stockfish binary and translates evaluations into Bangla commentary."""

    def __init__(self, stockfish_path: Optional[str] = None) -> None:
        self.stockfish_path = stockfish_path or shutil.which("stockfish")
        if not self.stockfish_path:
            for p in ["/usr/games/stockfish", "/usr/bin/stockfish", "stockfish.exe"]:
                if shutil.which(p):
                    self.stockfish_path = shutil.which(p)
                    break

        self.is_available = self.stockfish_path is not None
        if self.is_available:
            logger.info(f"Stockfish engine detected at '{self.stockfish_path}'.")
        else:
            logger.info("Stockfish engine not detected in system PATH. Operating in educational heuristic mode.")
            
        self.engine = None
        if self.is_available:
            try:
                import chess.engine
                self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            except Exception as e:
                logger.error(f"Failed to start Stockfish: {e}")
                self.is_available = False

    def __del__(self):
        try:
            if self.engine:
                self.engine.quit()
        except Exception:
            pass

    def analyze_position(self, fen: str, depth: int = 15) -> BestMoveEvaluation:
        """Evaluate a chess position given its FEN string."""
        if not self.is_available or not self.engine:
            return BestMoveEvaluation(
                fen=fen,
                best_move="",
                evaluation_score=0.0,
                explanation_bn="স্টকফিশ ইঞ্জিন পাওয়া যায়নি, তাই বিশ্লেষণ সম্ভব নয়।",
                is_engine_available=False,
            )

        try:
            import chess
            import chess.engine
            board = chess.Board(fen)
            if not board.is_valid():
                raise ValueError("Invalid FEN")

            info = self.engine.analyse(board, chess.engine.Limit(time=2.0))
            best_move = info.get("pv", [None])[0]
            if not best_move:
                return BestMoveEvaluation(
                    fen=fen,
                    best_move="",
                    evaluation_score=0.0,
                    explanation_bn="কোনো ভালো চাল পাওয়া যায়নি।",
                    is_engine_available=True,
                )
            
            score_obj = info["score"].white()
            if score_obj.is_mate():
                eval_score = 100.0 if score_obj.mate() > 0 else -100.0
            else:
                eval_score = score_obj.score() / 100.0

            best_move_str = best_move.uci()
            explanation = self.get_bangla_explanation(fen, best_move_str, eval_score)

            return BestMoveEvaluation(
                fen=fen,
                best_move=best_move_str,
                evaluation_score=eval_score,
                explanation_bn=explanation,
                is_engine_available=True,
            )
            
        except Exception as e:
            logger.error(f"Stockfish analysis failed: {e}")
            return BestMoveEvaluation(
                fen=fen,
                best_move="",
                evaluation_score=0.0,
                explanation_bn="বিশ্লেষণে ত্রুটি দেখা দিয়েছে।",
                is_engine_available=False,
            )

    def get_bangla_explanation(self, fen: str, best_move: str, score: float) -> str:
        pieces_bn = {
            "p": "সৈন্য",
            "n": "ঘোড়া",
            "b": "গজ",
            "r": "নৌকা",
            "q": "মন্ত্রী",
            "k": "রাজা"
        }
        
        try:
            import chess
            board = chess.Board(fen)
            move = chess.Move.from_uci(best_move)
            piece = board.piece_at(move.from_square)
            
            if piece:
                piece_name = pieces_bn.get(piece.symbol().lower(), "গুটি")
            else:
                piece_name = "গুটি"
                
            to_square = chess.square_name(move.to_square)
            
            score_text = f"পজিশনটি মোটামুটি সমান ({score:.1f})।"
            if score > 1.0:
                score_text = f"সাদার অবস্থা ভালো ({score:.1f})।"
            elif score < -1.0:
                score_text = f"কালোর অবস্থা ভালো ({score:.1f})।"
                
            return f"স্টকফিশের সেরা চাল হলো {best_move}। এটি আপনার {piece_name}কে {to_square} ঘরে নিয়ে যায়। {score_text}"
        except Exception:
            return f"স্টকফিশের সেরা চাল হলো {best_move}।"
