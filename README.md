## Chess Support MCP Server

An MCP server that manages the state of a chess game for LLMs/agents. It intentionally does not suggest moves. Instead, it provides tools to:

- Create/reset game
- Add a move (UCI)
- Undo the most recent move
- Peek a sequence of moves atomically (apply to a copy, see the result, state untouched)
- Load a full PGN (SAN, with optional start-position header) atomically
- List all moves
- Get last N moves
- Machine-friendly board JSON (square-to-piece map) in `get_status()`
- Check if a move is legal
- Enumerate all legal moves (UCI + SAN) in the current position
- Query the static attack map for any square
- Get status (FEN, whose turn, check, game over, result, checkers, absolute pins, material counts)

### Requirements

- Python 3.13+
- uv package manager

### Run via uvx directly from GitHub (no local checkout)

You can run this MCP server without cloning by using `uvx` with a Git URL. Replace placeholders with your repo info and optional tag/commit.

Generic MCP config (Inspector-style):

```json
{
  "servers": {
    "chess-support-mcp": {
      "transport": {
        "type": "stdio",
        "command": "uvx",
        "args": [
          "--from",
          "git+https://github.com/danilop/chess-support-mcp.git",
          "chess-support-mcp"
        ]
      }
    }
  }
}
```

Claude Desktop `mcpServers` example:

```json
{
  "mcpServers": {
    "chess-support-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/danilop/chess-support-mcp.git",
        "chess-support-mcp"
      ]
    }
  }
}
```

The first run may take longer while `uvx` resolves and builds the package; subsequent runs use cache.

### Configure as a local MCP server (JSON)

Use stdio with `uv run` (no hardcoded paths). Example generic JSON config:

```json
{
  "servers": {
    "chess-support-mcp": {
      "transport": {
        "type": "stdio",
        "command": "uv",
        "args": ["run", "chess-support-mcp"]
      }
    }
  }
}
```

Include a local path to your project without hardcoding a specific one by using a placeholder and setting the working directory via `cwd` (preferred), or by passing `--project`:

Option A (preferred: set working directory):

```json
{
  "servers": {
    "chess-support-mcp": {
      "transport": {
        "type": "stdio",
        "command": "uv",
        "args": ["run", "chess-support-mcp"],
        "cwd": "<ABSOLUTE_PATH_TO_PROJECT>"
      }
    }
  }
}
```

Option B (use uv's project flag):

```json
{
  "servers": {
    "chess-support-mcp": {
      "transport": {
        "type": "stdio",
        "command": "uv",
        "args": ["run", "--project", "<ABSOLUTE_PATH_TO_PROJECT>", "chess-support-mcp"]
      }
    }
  }
}
```

Claude Desktop configuration (in its JSON settings), using `mcpServers`:

```json
{
  "mcpServers": {
    "chess-support-mcp": {
      "command": "uv",
      "args": ["run", "chess-support-mcp"],
      "cwd": "<ABSOLUTE_PATH_TO_PROJECT>"
    }
  }
}
```

### Tools (Methods)

- `create_or_reset_game()` → Reset to initial position. Returns `status` (with `pieces` map), and `moves`.
- `get_status()` → Returns FEN; `side_to_move` (white/black); `fullmove_number`; `halfmove_clock`; `ply_count`; `last_move_uci`; `last_move_san`; `who_moved_last`; check flags; `is_game_over`; `result` when over; a `pieces` map for machine reasoning; `checkers` (list of `{square, piece}` for pieces giving check); `absolute_pins` (list of pin objects, each with `color`, `pinned_square`, `pinned_piece`, `king_square`, `pinner_square`, `pinner_piece`, and `ray` ordered from king outward); `material` (per-color piece counts `{Q,R,B,N,P}`, excluding kings, uncapped for promotions); and `material_diff` (per-piece `white[k] - black[k]`).
- `add_move(uci: str)` → Apply a move if legal (e.g., `e2e4`, `g1f3`, promotion like `e7e8q`). Returns `{ accepted, status }` and, on success, also `moves` and `moves_detailed`. On failure returns `{ accepted:false, reason:"illegal"|"parse_error", expected_turn? }` with `status` reflecting the unchanged position.
- `undo_last_move()` → Pop the most recent move off the stack. On success returns `{ accepted:true, status, moves, moves_detailed, undone:{uci, san, ply, side} }` with the position reverted. On an empty stack returns `{ accepted:false, reason:"no_moves", status }`. Uses python-chess `board.pop()`, which restores castling rights, en-passant square, halfmove clock, and check flags exactly. Can unwind moves that were replayed by `load_pgn` — the floor is whatever starting position the PGN declared.
- `peek(uci_sequence)` → Apply a sequence of UCI moves to a copy of the board, return the resulting position without mutating the game state. On success returns `{ accepted:true, peeked_status, applied:[{uci, san, ply, side}, ...], status }` where `status` is the UNCHANGED original. Failure envelopes: `reason:"empty_sequence"` for `[]`, `reason:"parse_error"` with `failed_at_index` + `failed_uci` + `parse_error` for an unparseable UCI string, `reason:"illegal"` with `failed_at_index` + `failed_uci` + `expected_turn` for a move that parses but isn't legal on the copy. Atomicity is structural — the tool operates on `board.copy()` and discards it, so self.board is never touched. UCI only; SAN input is out of scope.
- `load_pgn(pgn: str)` → Replace the current game by replaying a full PGN string. Atomic: the current game is only replaced on full success; any failure (unparseable PGN, illegal SAN mid-stream, empty movetext) leaves the prior game untouched. Respects `[FEN "..."]` start-position headers (the `[SetUp "1"]` header is recognized but not required). Variations inside parentheses are ignored — only the mainline is played. Returns `{ accepted, moves_applied, starting_fen, status, moves, moves_detailed, headers }` on success, or `{ accepted:false, reason:"parse_error"|"empty_pgn"|"illegal_move", parse_error?, moves_applied:0, status }` on failure.
- `is_legal(uci: str)` → Check legality of a UCI move in the current position.
- `list_legal_moves_detailed()` → All legal moves in the current position as `[{uci, san}, ...]`, sorted by UCI ascending. Empty list in checkmate or stalemate.
- `get_attackers_to(square: str)` → Static attack map for a square: `{ square, white: [{square, piece}, ...], black: [{square, piece}, ...] }`. **Note:** pinned pieces still count as attackers; this is the raw attack map, not a legality oracle. To ask "who can legally capture here right now", filter `list_legal_moves_detailed()` by destination square instead.
- `list_moves()` → All moves in UCI made so far.
- `list_moves_detailed()` → All moves with `ply`, `side`, `uci`, `san`.
- `last_moves(n: int=1)` → Last N moves in UCI.
- `last_moves_detailed(n: int=1)` → Last N moves with `ply`, `side`, `uci`, `san`.
- `board_ascii()` → ASCII board (optional, human-oriented). The normal API returns machine-friendly JSON in `status.pieces`.

### API design notes

- Moves are always provided in UCI (e.g., `e2e4`, `g1f3`, promotions `e7e8q`). The server infers side-to-move from position; you never specify white/black when sending a move.
- `get_status().side_to_move` tells the model whose turn it is. `who_moved_last`, `last_move_uci`, and `last_move_san` help with context.
- Detailed move lists are provided in separate `*_detailed` tools to keep the basic list simple and backwards compatible.

### Notes

- The server maintains one in-memory game.
- The server does not provide hints or best moves.
 
### Development

- Run tests:

```bash
uv run pytest -q
```

