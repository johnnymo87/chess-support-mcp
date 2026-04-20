from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import chess
from mcp.server.fastmcp import FastMCP


PIECE_KEYS = ("Q", "R", "B", "N", "P")


def _sort_ray_from_king(king_sq: int, ray_set: "chess.SquareSet") -> List[int]:
    """Order squares in ray_set (collinear with king_sq) from king outward.

    board.pin() returns a SquareSet (unordered bitboard) representing the
    rank, file, or diagonal of a pin. Consumers want a stable order; we
    impose: king first, then monotonically along the pin line.
    """
    squares = [sq for sq in ray_set if sq != king_sq]
    if not squares:
        return [king_sq]
    kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
    # Use the nearest-by-Chebyshev-distance square to fix direction robustly.
    nearest = min(
        squares,
        key=lambda s: max(
            abs(chess.square_file(s) - kf),
            abs(chess.square_rank(s) - kr),
        ),
    )
    nf, nr = chess.square_file(nearest), chess.square_rank(nearest)
    df = 0 if nf == kf else (1 if nf > kf else -1)
    dr = 0 if nr == kr else (1 if nr > kr else -1)

    def projection(sq: int) -> int:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        return (f - kf) * df + (r - kr) * dr

    return sorted(ray_set, key=projection)


@dataclass
class GameState:
    """Holds a single in-memory chess game state."""

    board: chess.Board
    san_history: List[str]

    @classmethod
    def new(cls) -> "GameState":
        return cls(board=chess.Board(), san_history=[])

    def reset(self) -> None:
        self.board.reset()
        self.san_history.clear()

    def add_move_uci(self, uci: str) -> Dict[str, Any]:
        try:
            move = chess.Move.from_uci(uci)
        except Exception as exc:  # noqa: BLE001
            return {
                "accepted": False,
                "reason": "parse_error",
                "parse_error": str(exc),
            }
        if move not in self.board.legal_moves:
            return {
                "accepted": False,
                "reason": "illegal",
                "expected_turn": "white" if self.board.turn else "black",
            }
        san = self.board.san(move)
        self.board.push(move)
        self.san_history.append(san)
        return {"accepted": True}

    def is_move_legal(self, uci: str) -> Dict[str, Any]:
        try:
            move = chess.Move.from_uci(uci)
        except Exception as exc:  # noqa: BLE001
            return {"parse_error": str(exc), "legal": False}
        return {"legal": move in self.board.legal_moves}

    def all_moves(self) -> List[str]:
        return [m.uci() for m in self.board.move_stack]

    def all_moves_detailed(self) -> List[Dict[str, Any]]:
        details: List[Dict[str, Any]] = []
        for idx, move in enumerate(self.board.move_stack):
            details.append(
                {
                    "ply": idx + 1,
                    "uci": move.uci(),
                    "san": self.san_history[idx]
                    if idx < len(self.san_history)
                    else None,
                    "side": "white" if idx % 2 == 0 else "black",
                }
            )
        return details

    def last_n_moves(self, n: int) -> List[str]:
        if n <= 0:
            return []
        return [m.uci() for m in self.board.move_stack[-n:]]

    def last_n_moves_detailed(self, n: int) -> List[Dict[str, Any]]:
        if n <= 0:
            return []
        start = max(0, len(self.board.move_stack) - n)
        result: List[Dict[str, Any]] = []
        for idx in range(start, len(self.board.move_stack)):
            move = self.board.move_stack[idx]
            result.append(
                {
                    "ply": idx + 1,
                    "uci": move.uci(),
                    "san": self.san_history[idx]
                    if idx < len(self.san_history)
                    else None,
                    "side": "white" if idx % 2 == 0 else "black",
                }
            )
        return result

    def ascii_board(self) -> str:
        return str(self.board)

    def pieces_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for square, piece in self.board.piece_map().items():
            mapping[chess.square_name(square)] = piece.symbol()
        return mapping

    def material_counts(self) -> Dict[str, Dict[str, int]]:
        white = {k: 0 for k in PIECE_KEYS}
        black = {k: 0 for k in PIECE_KEYS}
        for piece in self.board.piece_map().values():
            sym_upper = piece.symbol().upper()
            if sym_upper == "K":
                continue
            if piece.color == chess.WHITE:
                white[sym_upper] += 1
            else:
                black[sym_upper] += 1
        return {"white": white, "black": black}

    def checkers_info(self) -> List[Dict[str, str]]:
        result: List[Dict[str, str]] = []
        for sq in self.board.checkers():
            piece = self.board.piece_at(sq)
            result.append(
                {
                    "square": chess.square_name(sq),
                    "piece": piece.symbol() if piece else "?",
                }
            )
        return result

    def absolute_pins_info(self) -> List[Dict[str, Any]]:
        pins: List[Dict[str, Any]] = []
        for color in (chess.WHITE, chess.BLACK):
            king_sq = self.board.king(color)
            if king_sq is None:
                continue
            color_name = "white" if color == chess.WHITE else "black"
            for square, piece in self.board.piece_map().items():
                if piece.color != color:
                    continue
                if not self.board.is_pinned(color, square):
                    continue
                ray_set = self.board.pin(color, square)
                ray_squares = _sort_ray_from_king(king_sq, ray_set)
                # The pinner is the first enemy piece on the ray beyond the
                # pinned piece (further from the king). board.pin() returns
                # the full rank/file/diagonal mask, which may include unrelated
                # enemy pieces (e.g. the enemy king on the same file); walking
                # outward from the pinned piece avoids picking them up.
                pinned_idx = ray_squares.index(square)
                pinner_sq = next(
                    (
                        sq
                        for sq in ray_squares[pinned_idx + 1 :]
                        if self.board.piece_at(sq) is not None
                        and self.board.piece_at(sq).color != color
                    ),
                    None,
                )
                if pinner_sq is None:
                    continue  # defensive; is_pinned() shouldn't return True without a pinner
                pinner_piece = self.board.piece_at(pinner_sq)
                pins.append(
                    {
                        "color": color_name,
                        "pinned_square": chess.square_name(square),
                        "pinned_piece": piece.symbol(),
                        "king_square": chess.square_name(king_sq),
                        "pinner_square": chess.square_name(pinner_sq),
                        "pinner_piece": pinner_piece.symbol() if pinner_piece else "?",
                        "ray": [chess.square_name(sq) for sq in ray_squares],
                    }
                )
        return pins

    def status(self) -> Dict[str, Any]:
        fen = self.board.fen()
        parts = fen.split()
        last_move_uci = (
            self.board.move_stack[-1].uci() if self.board.move_stack else None
        )
        last_move_san = self.san_history[-1] if self.san_history else None
        material = self.material_counts()
        material_diff = {
            k: material["white"][k] - material["black"][k] for k in PIECE_KEYS
        }
        return {
            "fen": fen,
            "side_to_move": "white" if self.board.turn else "black",
            "fullmove_number": self.board.fullmove_number,
            "halfmove_clock": self.board.halfmove_clock,
            "ply_count": len(self.board.move_stack),
            "castling_rights": parts[2] if len(parts) >= 3 else None,
            "en_passant_square": parts[3] if len(parts) >= 4 else None,
            "last_move_uci": last_move_uci,
            "last_move_san": last_move_san,
            "who_moved_last": (
                "white" if (len(self.board.move_stack) - 1) % 2 == 0 else "black"
            )
            if self.board.move_stack
            else None,
            "is_check": self.board.is_check(),
            "is_game_over": self.board.is_game_over(),
            "result": self.board.result(claim_draw=True)
            if self.board.is_game_over()
            else None,
            "pieces": self.pieces_map(),
            "checkers": self.checkers_info(),
            "absolute_pins": self.absolute_pins_info(),
            "material": material,
            "material_diff": material_diff,
        }


server = FastMCP(
    "chess-support-mcp",
    "MCP server that manages a single chess game: create/reset, add move, list moves, last N moves, ASCII board, move legality, and status.",
)

_GAME = GameState.new()


@server.tool()
def create_or_reset_game() -> Dict[str, Any]:
    """Create a new game or reset the current one to the initial position.

    Returns (in result):
    - ok: boolean
    - status: object with the current position metadata:
      - fen: string (Forsyth–Edwards Notation)
      - side_to_move: "white" | "black"
      - fullmove_number: int
      - halfmove_clock: int
      - ply_count: int (number of half-moves made)
      - castling_rights: string like "KQkq" or "-"
      - en_passant_square: algebraic square like "e3" or "-"
      - last_move_uci: string | null
      - last_move_san: string | null
      - who_moved_last: "white" | "black" | null
      - is_check: boolean
      - is_game_over: boolean
      - result: string like "1-0", "0-1", "1/2-1/2" or null
      - pieces: object mapping squares to piece symbols, e.g. {"e4":"P", "e5":"p"}
    - moves: array of UCI strings for all moves played so far (empty after reset)
    - moves_detailed: array of { ply:int, side:"white"|"black", uci:string, san:string }

    Notes:
    - This tool does not suggest moves; it only manages state.
    """

    _GAME.reset()
    return {
        "ok": True,
        "status": _GAME.status(),
        "moves": _GAME.all_moves(),
        "moves_detailed": [
            _format_move_detail(idx, move, _GAME.san_history)
            for idx, move in enumerate(_GAME.board.move_stack)
        ],
    }


@server.tool()
def get_status() -> Dict[str, Any]:
    """Get current position metadata for model-friendly planning.

    Returns (in result):
    - fen, side_to_move, fullmove_number, halfmove_clock, ply_count
    - castling_rights, en_passant_square
    - last_move_uci, last_move_san, who_moved_last
    - is_check, is_game_over, result
    - pieces: square-to-piece map (e.g., {"a2":"P", "e1":"K"})
    """

    return _GAME.status()


@server.tool()
def add_move(uci: str) -> Dict[str, Any]:
    """Apply a move in UCI format if legal.

    Parameters:
    - uci: string like "e2e4", "g1f3", promotions like "e7e8q".

    Returns (in result):
    - On success: { accepted:true, status: Status, moves:[...], moves_detailed:[...] }
      where Status is the same shape returned by get_status(), including last_move_{uci,san}.
    - On failure: { accepted:false, reason:"illegal", expected_turn:"white"|"black", status: Status }

    Notes:
    - This tool validates legality only; it does not suggest or score moves.
    """

    move_outcome = _GAME.add_move_uci(uci)
    response: Dict[str, Any] = {"accepted": bool(move_outcome.get("accepted"))}
    response["status"] = _GAME.status()
    if not response["accepted"]:
        if "reason" in move_outcome:
            response["reason"] = move_outcome["reason"]
        if "expected_turn" in move_outcome:
            response["expected_turn"] = move_outcome["expected_turn"]
        return response

    response["moves"] = _GAME.all_moves()
    response["moves_detailed"] = _GAME.all_moves_detailed()
    return response


@server.tool()
def is_legal(uci: str) -> Dict[str, Any]:
    """Check if a UCI move is legal in the current position.

    Returns (in result):
    - { legal:boolean }.
    - If the UCI string cannot be parsed, { parse_error:string, legal:false }.
    """

    return _GAME.is_move_legal(uci)


@server.tool()
def list_moves() -> List[str]:
    """Return all moves played so far in UCI, ordered from the start of the game."""

    return _GAME.all_moves()


def _format_move_detail(
    idx: int, move: chess.Move, san_history: List[str]
) -> Dict[str, Any]:
    return {
        "ply": idx + 1,
        "uci": move.uci(),
        "san": san_history[idx] if idx < len(san_history) else None,
        "side": "white" if idx % 2 == 0 else "black",
    }


@server.tool()
def list_moves_detailed() -> List[Dict[str, Any]]:
    """Return detailed move history.

    Each item: { ply:int, side:"white"|"black", uci:string, san:string }
    """

    return [
        _format_move_detail(idx, move, _GAME.san_history)
        for idx, move in enumerate(_GAME.board.move_stack)
    ]


@server.tool()
def last_moves(n: int = 1) -> List[str]:
    """Return the last N moves in UCI (default 1).

    Parameters:
    - n: integer >= 1. If n <= 0, returns an empty list.
    """

    return _GAME.last_n_moves(n)


@server.tool()
def last_moves_detailed(n: int = 1) -> List[Dict[str, Any]]:
    """Return the last N moves with details (default 1).

    Each item: { ply:int, side:"white"|"black", uci:string, san:string }.
    Parameters:
    - n: integer >= 1. If n <= 0, returns an empty list.
    """

    if n <= 0:
        return []
    start = max(0, len(_GAME.board.move_stack) - n)
    return [
        _format_move_detail(idx, _GAME.board.move_stack[idx], _GAME.san_history)
        for idx in range(start, len(_GAME.board.move_stack))
    ]


@server.tool()
def get_attackers_to(square: str) -> Dict[str, Any]:
    """Return all pieces attacking the given square, by color.

    Parameters:
    - square: algebraic square name like "e4".

    Returns:
    - { square, white: [{square, piece}, ...], black: [{square, piece}, ...] }
    - On parse error: { square, parse_error: str, white: [], black: [] }

    IMPORTANT: This is the static attack map. Pinned pieces are still counted
    as attackers, and x-ray attacks through the king are not considered. This
    tool does not answer "who can legally capture on this square right now" —
    to answer that, filter list_legal_moves_detailed() by destination square.
    """

    try:
        sq = chess.parse_square(square)
    except ValueError as exc:
        return {"square": square, "parse_error": str(exc), "white": [], "black": []}

    def format_attackers(color: chess.Color) -> List[Dict[str, str]]:
        entries: List[Dict[str, str]] = []
        for s in _GAME.board.attackers(color, sq):
            piece = _GAME.board.piece_at(s)
            entries.append(
                {
                    "square": chess.square_name(s),
                    "piece": piece.symbol() if piece else "?",
                }
            )
        return entries

    return {
        "square": square,
        "white": format_attackers(chess.WHITE),
        "black": format_attackers(chess.BLACK),
    }


@server.tool()
def list_legal_moves_detailed() -> List[Dict[str, Any]]:
    """Return all legal moves in the current position, sorted by UCI ascending.

    Each item: { uci: string, san: string }.

    Notes:
    - Ordering: sorted by uci ascending (stable, deterministic).
    - Empty list in checkmate or stalemate.
    - This tool enumerates legality; it does not suggest or score moves.
    """

    moves = [
        {"uci": m.uci(), "san": _GAME.board.san(m)} for m in _GAME.board.legal_moves
    ]
    moves.sort(key=lambda x: x["uci"])
    return moves


@server.tool()
def board_ascii() -> str:
    """Return an ASCII representation of the board from White's perspective.

    Notes:
    - This is a human-oriented view, suitable for displaying the board to users in UIs, logs, or chat.
    - For model reasoning, prefer the JSON map in get_status().pieces.
    """

    return _GAME.ascii_board()


def main() -> None:
    """Entry point: run MCP server over stdio."""

    server.run()


if __name__ == "__main__":
    main()
