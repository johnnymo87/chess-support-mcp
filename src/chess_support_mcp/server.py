from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import io

import chess
import chess.pgn
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


def _status_from_board(
    board: chess.Board,
    san_history: List[str] | None = None,
) -> Dict[str, Any]:
    """Build a Status dict from a chess.Board.

    Shared by GameState.status() (live self.board) and peek (an isolated
    copy). san_history is optional: when absent (or shorter than move_stack),
    last_move_san falls back to None. Peek passes a list it built against the
    copy; GameState passes its own self.san_history.

    The length guard on san_history is a deliberate tightening of the old
    GameState.status() behaviour: the old code returned san_history[-1]
    whenever san_history was truthy, regardless of whether it was the right
    length. This helper returns None when history is shorter than the move
    stack, which is the correct answer for that edge case. Do not simplify
    the guard back to the old form.

    Factored from the original GameState.status() so both callers produce
    identical Status shapes without duplication.
    """
    fen = board.fen()
    parts = fen.split()
    last_move_uci = board.move_stack[-1].uci() if board.move_stack else None
    if san_history and len(san_history) >= len(board.move_stack) and board.move_stack:
        last_move_san = san_history[-1]
    else:
        last_move_san = None

    white = {k: 0 for k in PIECE_KEYS}
    black = {k: 0 for k in PIECE_KEYS}
    for piece in board.piece_map().values():
        sym_upper = piece.symbol().upper()
        if sym_upper == "K":
            continue
        if piece.color == chess.WHITE:
            white[sym_upper] += 1
        else:
            black[sym_upper] += 1
    material = {"white": white, "black": black}
    material_diff = {k: white[k] - black[k] for k in PIECE_KEYS}

    pieces = {
        chess.square_name(sq): piece.symbol()
        for sq, piece in board.piece_map().items()
    }

    checkers = []
    for sq in board.checkers():
        piece = board.piece_at(sq)
        checkers.append(
            {
                "square": chess.square_name(sq),
                "piece": piece.symbol() if piece else "?",
            }
        )

    absolute_pins: List[Dict[str, Any]] = []
    for color in (chess.WHITE, chess.BLACK):
        king_sq = board.king(color)
        if king_sq is None:
            continue
        color_name = "white" if color == chess.WHITE else "black"
        for square, piece in board.piece_map().items():
            if piece.color != color:
                continue
            if not board.is_pinned(color, square):
                continue
            ray_set = board.pin(color, square)
            ray_squares = _sort_ray_from_king(king_sq, ray_set)
            pinned_idx = ray_squares.index(square)
            pinner_sq = next(
                (
                    sq
                    for sq in ray_squares[pinned_idx + 1 :]
                    if board.piece_at(sq) is not None
                    and board.piece_at(sq).color != color
                ),
                None,
            )
            if pinner_sq is None:
                continue
            pinner_piece = board.piece_at(pinner_sq)
            absolute_pins.append(
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

    return {
        "fen": fen,
        "side_to_move": "white" if board.turn else "black",
        "fullmove_number": board.fullmove_number,
        "halfmove_clock": board.halfmove_clock,
        "ply_count": len(board.move_stack),
        "castling_rights": parts[2] if len(parts) >= 3 else None,
        "en_passant_square": parts[3] if len(parts) >= 4 else None,
        "last_move_uci": last_move_uci,
        "last_move_san": last_move_san,
        "who_moved_last": (
            "white" if (len(board.move_stack) - 1) % 2 == 0 else "black"
        )
        if board.move_stack
        else None,
        "is_check": board.is_check(),
        "is_game_over": board.is_game_over(),
        "result": board.result(claim_draw=True) if board.is_game_over() else None,
        "pieces": pieces,
        "checkers": checkers,
        "absolute_pins": absolute_pins,
        "material": material,
        "material_diff": material_diff,
    }


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

    def undo_last_move(self) -> Dict[str, Any]:
        """Pop the most recent move off the stack, reverting the position.

        Uses python-chess's board.pop(), which authoritatively restores
        castling rights, en-passant square, halfmove clock, side-to-move,
        is_check / is_game_over, and the board itself. The parallel
        san_history list is popped in lockstep so last_move_san and
        list_moves_detailed stay correct.
        """
        if not self.board.move_stack:
            return {"accepted": False, "reason": "no_moves"}
        move = self.board.pop()
        san = self.san_history.pop() if self.san_history else None
        ply = len(self.board.move_stack) + 1
        side = "white" if (ply - 1) % 2 == 0 else "black"
        return {
            "accepted": True,
            "undone": {
                "uci": move.uci(),
                "san": san,
                "ply": ply,
                "side": side,
            },
        }

    def load_pgn_text(self, pgn: str) -> Dict[str, Any]:
        """Parse a PGN string and replay it atomically into this game.

        On success: mutates self.board / self.san_history to the post-PGN state
        and returns a dict with accepted=True plus metadata.
        On failure: leaves self unchanged and returns accepted=False plus a
        reason. Never partially applies.
        """
        try:
            game = chess.pgn.read_game(io.StringIO(pgn))
        except Exception as exc:  # noqa: BLE001 — io.StringIO raises TypeError on non-str; chess.pgn.read_game itself returns None rather than raising, but catch broadly for safety
            return {
                "accepted": False,
                "reason": "parse_error",
                "parse_error": str(exc),
                "moves_applied": 0,
            }

        if game is None:
            return {
                "accepted": False,
                "reason": "empty_pgn",
                "moves_applied": 0,
            }

        # python-chess collects SAN-parsing failures in game.errors rather than
        # raising. Crucially, valid moves *before* the bad SAN are still
        # present in game.mainline_moves() — so this check MUST come before
        # the empty-mainline check below, or we would silently apply a
        # partial replay on PGNs with an illegal move mid-stream.
        if game.errors:
            return {
                "accepted": False,
                "reason": "illegal_move",
                "parse_error": str(game.errors[0]),
                "moves_applied": 0,
            }

        mainline = list(game.mainline_moves())
        if not mainline:
            return {
                "accepted": False,
                "reason": "empty_pgn",
                "moves_applied": 0,
            }

        # Build the new state in locals first. Only swap into self at the end.
        new_board = game.board()  # respects [FEN ...] [SetUp "1"] if present
        starting_fen = new_board.fen()
        new_san_history: List[str] = []
        for move in mainline:
            # Compute SAN *before* pushing, to match add_move_uci's convention.
            new_san_history.append(new_board.san(move))
            new_board.push(move)

        # game.headers is a live MutableMapping; copy to a plain dict for
        # JSON serializability and to decouple from the game object's lifetime.
        headers = dict(game.headers)

        # Commit.
        self.board = new_board
        self.san_history = new_san_history

        return {
            "accepted": True,
            "moves_applied": len(mainline),
            "starting_fen": starting_fen,
            "headers": headers,
        }

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
        return _status_from_board(self.board, self.san_history)


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
def undo_last_move() -> Dict[str, Any]:
    """Revert the most recent move, restoring the prior position.

    Parameters: (none)

    Returns (in result):
    - On success: {
        accepted: true,
        status: Status,                 # full get_status() shape, post-undo
        moves: [uci, ...],              # updated history
        moves_detailed: [...],          # updated history
        undone: { uci, san, ply, side } # what disappeared
      }
    - On failure (empty stack): {
        accepted: false,
        reason: "no_moves",
        status: Status                  # unchanged
      }

    Notes:
    - Uses python-chess board.pop(), which restores castling rights,
      en-passant square, halfmove clock, side-to-move, and check/game-over
      flags exactly. No manual reconstruction.
    - Can unwind moves that were replayed by load_pgn. The floor is
      whatever starting position the PGN declared (standard start unless
      the PGN had a [FEN ...] header).
    - This tool mutates state; it does not suggest or score moves.
    """

    outcome = _GAME.undo_last_move()
    response: Dict[str, Any] = {
        "accepted": bool(outcome.get("accepted")),
        "status": _GAME.status(),
    }
    if not response["accepted"]:
        response["reason"] = outcome["reason"]
        return response

    response["undone"] = outcome["undone"]
    response["moves"] = _GAME.all_moves()
    response["moves_detailed"] = _GAME.all_moves_detailed()
    return response


@server.tool()
def load_pgn(pgn: str) -> Dict[str, Any]:
    """Load a full PGN string, replacing the current game with its position.

    Parameters:
    - pgn: string containing a PGN (headers optional; movetext required).

    Behavior:
    - Parses with python-chess's PGN reader and replays the mainline moves.
    - Variations (inside parentheses) are ignored; only the mainline is played.
    - [FEN "..."] headers are respected: replay starts from the declared
      position rather than the standard start. The [SetUp "1"] header is
      recognized but not required.
    - Atomic: the current game is only replaced if the entire PGN replays
      cleanly. On any failure the current game is preserved.

    Returns (in result):
    - On success: {
        accepted: true,
        moves_applied: int,
        starting_fen: str,
        status: Status,
        moves: [uci, ...],
        moves_detailed: [{ply, uci, san, side}, ...],
        headers: {tag: value, ...},
      }
    - On failure: {
        accepted: false,
        reason: "parse_error" | "empty_pgn" | "illegal_move",
        parse_error: str (when applicable),
        moves_applied: 0,
        status: <unchanged>,
      }

    Notes:
    - This tool validates and replays; it does not suggest or score moves.
    - If you need to append a single move in SAN to an existing game, this is
      not the right tool — use add_move with a UCI string.
    """

    outcome = _GAME.load_pgn_text(pgn)
    response: Dict[str, Any] = {
        "accepted": bool(outcome.get("accepted")),
        "moves_applied": outcome.get("moves_applied", 0),
        "status": _GAME.status(),
    }
    if not response["accepted"]:
        response["reason"] = outcome["reason"]
        if "parse_error" in outcome:
            response["parse_error"] = outcome["parse_error"]
        return response

    response["starting_fen"] = outcome["starting_fen"]
    response["headers"] = outcome["headers"]
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
