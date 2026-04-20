import asyncio
from contextlib import asynccontextmanager

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


@asynccontextmanager
async def run_client():
    server = StdioServerParameters(
        command="uv", args=["run", "chess-support-mcp"]
    )  # stdio MCP server
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@pytest.mark.anyio
async def test_reset_and_status():
    async with run_client() as session:
        tools = await session.list_tools()
        names = {t.name for t in tools.tools}
        assert "create_or_reset_game" in names
        assert "get_status" in names

        resp = await session.call_tool("create_or_reset_game", {})
        out = resp.structuredContent["result"]
        assert out["ok"] is True
        status = out["status"]
        assert status["side_to_move"] == "white"
        assert status["ply_count"] == 0
        assert out["moves"] == []
        # Ensure pieces map exists and is non-empty in initial position
        assert isinstance(status["pieces"], dict)
        assert status["pieces"]["a2"] == "P" and status["pieces"]["e1"] == "K"

        # New enriched-status fields (added in this PR)
        assert isinstance(status["absolute_pins"], list)
        assert isinstance(status["checkers"], list)
        assert isinstance(status["material"], dict)
        assert isinstance(status["material_diff"], dict)
        assert status["material_diff"] == {"Q": 0, "R": 0, "B": 0, "N": 0, "P": 0}
        assert status["material"]["white"] == {"Q": 1, "R": 2, "B": 2, "N": 2, "P": 8}
        assert status["material"]["black"] == {"Q": 1, "R": 2, "B": 2, "N": 2, "P": 8}
        assert status["checkers"] == []
        assert status["absolute_pins"] == []

        status2 = await session.call_tool("get_status", {})
        s2 = status2.structuredContent["result"]
        assert s2["side_to_move"] == "white"


@pytest.mark.anyio
async def test_add_move_and_list():
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})

        add = await session.call_tool("add_move", {"uci": "e2e4"})
        a = add.structuredContent["result"]
        assert a["accepted"] is True
        assert a["status"]["last_move_san"] == "e4"

        # Illegal: same side tries to move again
        add2 = await session.call_tool("add_move", {"uci": "e2e4"})
        a2 = add2.structuredContent["result"]
        assert a2["accepted"] is False
        assert a2["reason"] in {"illegal", "parse_error"}
        if a2["reason"] == "illegal":
            assert a2["expected_turn"] == "black"
        # Status should be unchanged (still e4 played only)
        assert a2["status"]["last_move_uci"] == "e2e4"

        # Black moves
        add3 = await session.call_tool("add_move", {"uci": "e7e5"})
        a3 = add3.structuredContent["result"]
        assert a3["accepted"] is True
        assert a3["status"]["last_move_san"] == "e5"

        # History
        lm = await session.call_tool("list_moves", {})
        assert lm.structuredContent["result"] == ["e2e4", "e7e5"]

        lmd = await session.call_tool("list_moves_detailed", {})
        assert (
            lmd.structuredContent["result"][0]["ply"] == 1
            and lmd.structuredContent["result"][0]["side"] == "white"
            and lmd.structuredContent["result"][0]["san"] == "e4"
        )
        assert (
            lmd.structuredContent["result"][1]["ply"] == 2
            and lmd.structuredContent["result"][1]["side"] == "black"
            and lmd.structuredContent["result"][1]["san"] == "e5"
        )

        last1 = await session.call_tool("last_moves", {"n": 1})
        assert last1.structuredContent["result"] == ["e7e5"]

        last1d = await session.call_tool("last_moves_detailed", {"n": 1})
        assert last1d.structuredContent["result"][0]["uci"] == "e7e5"


@pytest.mark.anyio
async def test_legality_and_board_ascii():
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        legal = await session.call_tool("is_legal", {"uci": "e2e4"})
        assert legal.structuredContent["result"]["legal"] is True

        board = await session.call_tool("board_ascii", {})
        assert isinstance(board.structuredContent["result"], str)

        # Parse error path
        bad = await session.call_tool("add_move", {"uci": "not-a-uci"})
        badr = bad.structuredContent["result"]
        assert badr["accepted"] is False and badr["reason"] == "parse_error"


@pytest.mark.anyio
async def test_material_initial():
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        s = (await session.call_tool("get_status", {})).structuredContent["result"]
        assert s["material"]["white"] == {"Q": 1, "R": 2, "B": 2, "N": 2, "P": 8}
        assert s["material"]["black"] == {"Q": 1, "R": 2, "B": 2, "N": 2, "P": 8}
        assert s["material_diff"] == {"Q": 0, "R": 0, "B": 0, "N": 0, "P": 0}


@pytest.mark.anyio
async def test_material_after_capture():
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        for uci in ["e2e4", "d7d5", "e4d5"]:
            await session.call_tool("add_move", {"uci": uci})
        s = (await session.call_tool("get_status", {})).structuredContent["result"]
        assert s["material"]["white"]["P"] == 8
        assert s["material"]["black"]["P"] == 7
        assert s["material_diff"]["P"] == 1


@pytest.mark.anyio
async def test_checkers_zero():
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        s = (await session.call_tool("get_status", {})).structuredContent["result"]
        assert s["checkers"] == []


@pytest.mark.anyio
async def test_checkers_one():
    # Scholar's Mate: Qxf7#. After 7. Qxf7+, Black's king is in check from the queen.
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        for uci in ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]:
            await session.call_tool("add_move", {"uci": uci})
        s = (await session.call_tool("get_status", {})).structuredContent["result"]
        assert s["is_check"] is True
        assert s["checkers"] == [{"square": "f7", "piece": "Q"}]


@pytest.mark.anyio
@pytest.mark.skip(
    reason="Double-check setup via legal play from initial position is awkward; "
    "deferred until/unless we add a test-only FEN loader. "
    "See docs/plans/2026-04-19-enriched-status-design.md Test 9."
)
async def test_checkers_two():
    pass


@pytest.mark.anyio
async def test_absolute_pins_empty_initial():
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        s = (await session.call_tool("get_status", {})).structuredContent["result"]
        assert s["absolute_pins"] == []


@pytest.mark.anyio
async def test_absolute_pins_detected():
    # After 1.e4 e5 2.Qh5 Nf6 3.Qxe5+ Be7, White's queen on e5 absolutely pins
    # Black's Be7 to Ke8 along the e-file. Verified manually via python-chess.
    # NOTE: the more "classical" Bb5 pin (1.e4 e5 2.Nf3 Nc6 3.Bb5) is NOT an
    # absolute pin because the d7 pawn blocks the bishop's line of sight to
    # the king. See docs/plans/2026-04-19-enriched-status-design.md note on
    # Test 4 for the fixture rationale.
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        for uci in ["e2e4", "e7e5", "d1h5", "g8f6", "h5e5", "f8e7"]:
            await session.call_tool("add_move", {"uci": uci})
        s = (await session.call_tool("get_status", {})).structuredContent["result"]
        pins = s["absolute_pins"]
        assert len(pins) == 1
        p = pins[0]
        assert p["color"] == "black"
        assert p["pinned_square"] == "e7"
        assert p["pinned_piece"] == "b"
        assert p["king_square"] == "e8"
        assert p["pinner_square"] == "e5"
        assert p["pinner_piece"] == "Q"
        # Ray is the e-file; ordered from king (e8) outward along the pin line.
        assert p["ray"][0] == "e8"
        assert "e7" in p["ray"]
        assert "e5" in p["ray"]
        # board.pin() returns the full rank/file/diagonal mask, so the e-file
        # ray extends all the way to e1.
        assert p["ray"][-1] == "e1"


@pytest.mark.anyio
async def test_relative_pin_not_reported():
    # After 1.d4 d5 2.Bf4 Nc6 3.Nf3 Bg4, Bg4 relatively pins Nf3 to Qd1
    # (through the queen, not the king). No king involvement => no absolute
    # pin. The defensive assertion: Nf3 must NOT appear in absolute_pins.
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        for uci in ["d2d4", "d7d5", "c1f4", "b8c6", "g1f3", "c8g4"]:
            await session.call_tool("add_move", {"uci": uci})
        s = (await session.call_tool("get_status", {})).structuredContent["result"]
        for p in s["absolute_pins"]:
            assert p["pinned_square"] != "f3", (
                "Nf3 is only relatively pinned (to Qd1), should not appear in absolute_pins"
            )


@pytest.mark.anyio
async def test_list_legal_moves_detailed_initial():
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        result = (
            await session.call_tool("list_legal_moves_detailed", {})
        ).structuredContent["result"]
        assert len(result) == 20
        assert {"uci": "e2e4", "san": "e4"} in result
        # Sorted by UCI ascending
        ucis = [m["uci"] for m in result]
        assert ucis == sorted(ucis)


@pytest.mark.anyio
async def test_list_legal_moves_detailed_checkmate():
    # Fool's Mate
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
            await session.call_tool("add_move", {"uci": uci})
        status = (await session.call_tool("get_status", {})).structuredContent["result"]
        assert status["is_game_over"] is True
        result = (
            await session.call_tool("list_legal_moves_detailed", {})
        ).structuredContent["result"]
        assert result == []


@pytest.mark.anyio
async def test_list_legal_moves_detailed_sorted():
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        await session.call_tool("add_move", {"uci": "e2e4"})
        result = (
            await session.call_tool("list_legal_moves_detailed", {})
        ).structuredContent["result"]
        ucis = [m["uci"] for m in result]
        assert ucis == sorted(ucis)


@pytest.mark.anyio
async def test_get_attackers_to_basic():
    # After 1.e4 e5 2.Nf3, White's Nf3 attacks e5; no Black piece attacks e5.
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        for uci in ["e2e4", "e7e5", "g1f3"]:
            await session.call_tool("add_move", {"uci": uci})
        result = (
            await session.call_tool("get_attackers_to", {"square": "e5"})
        ).structuredContent["result"]
        assert result["square"] == "e5"
        white_squares = {a["square"] for a in result["white"]}
        assert "f3" in white_squares
        # Confirm piece info is included
        nf3 = next(a for a in result["white"] if a["square"] == "f3")
        assert nf3["piece"] == "N"


@pytest.mark.anyio
async def test_get_attackers_to_includes_pinned():
    # Reuse the Qe5-pins-Be7 position from test_absolute_pins_detected.
    # The pinned Be7 still attacks d6 and f6 and so on; it must appear in
    # the static attack map despite being pinned (documents the semantic
    # trap called out in the tool's docstring).
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        for uci in ["e2e4", "e7e5", "d1h5", "g8f6", "h5e5", "f8e7"]:
            await session.call_tool("add_move", {"uci": uci})
        result = (
            await session.call_tool("get_attackers_to", {"square": "d6"})
        ).structuredContent["result"]
        black_squares = {a["square"] for a in result["black"]}
        assert "e7" in black_squares, (
            "Pinned bishop on e7 should still appear as static-map attacker of d6"
        )


@pytest.mark.anyio
async def test_get_attackers_to_parse_error():
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        result = (
            await session.call_tool("get_attackers_to", {"square": "zz9"})
        ).structuredContent["result"]
        assert result["square"] == "zz9"
        assert "parse_error" in result
        assert result["white"] == []
        assert result["black"] == []


@pytest.mark.anyio
async def test_material_after_promotion():
    # 12-ply forcing line where Black's h-pawn marches to h1 with captures
    # and promotes to a queen. After 6...gxh1=Q, Black has 2 queens.
    # Verifies material counts don't silently cap at starting-position
    # maxima (promotion can legally produce >1 queen per side).
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        moves = [
            "e2e4",
            "h7h5",
            "e4e5",
            "h5h4",
            "e5e6",
            "d7e6",
            "d2d4",
            "h4h3",
            "d4d5",
            "h3g2",
            "d5e6",
            "g2h1q",
        ]
        for uci in moves:
            resp = await session.call_tool("add_move", {"uci": uci})
            assert resp.structuredContent["result"]["accepted"] is True, (
                f"Move {uci} unexpectedly rejected; check the line"
            )
        s = (await session.call_tool("get_status", {})).structuredContent["result"]
        # Black now has 2 queens (original Qd8 + promoted queen on h1).
        assert s["material"]["black"]["Q"] == 2, (
            f"Expected 2 black queens after promotion, got {s['material']['black']['Q']}"
        )
        # White has 1 queen, black has 2, so diff = -1.
        assert s["material_diff"]["Q"] == -1


@pytest.mark.anyio
async def test_load_pgn_happy_path():
    # Short legal opening. Covers headers, SAN with piece moves, and a capture.
    pgn = (
        '[Event "Test"]\n'
        '[Site "?"]\n'
        '[Date "2026.04.19"]\n'
        '[Round "1"]\n'
        '[White "Alice"]\n'
        '[Black "Bob"]\n'
        '[Result "*"]\n'
        "\n"
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Bxc6 dxc6 *\n"
    )
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        # Dirty the game first to prove load_pgn replaces state, not appends.
        await session.call_tool("add_move", {"uci": "d2d4"})

        resp = await session.call_tool("load_pgn", {"pgn": pgn})
        out = resp.structuredContent["result"]
        # Replacement verified: d2d4 must not appear in the new game's history.
        assert "d2d4" not in out["moves"], (
            "load_pgn must replace state, not append to it"
        )
        assert out["accepted"] is True
        assert out["moves_applied"] == 8
        # Standard start position FEN
        assert out["starting_fen"].startswith(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        )
        assert out["status"]["ply_count"] == 8
        assert out["status"]["last_move_san"] == "dxc6"
        assert out["moves"][:3] == ["e2e4", "e7e5", "g1f3"]
        # SAN history populated in moves_detailed
        assert out["moves_detailed"][0]["san"] == "e4"
        assert out["moves_detailed"][2]["san"] == "Nf3"
        assert out["moves_detailed"][6]["san"] == "Bxc6"  # piece capture
        assert out["moves_detailed"][-1]["san"] == "dxc6"  # pawn recapture
        assert out["moves_detailed"][0]["side"] == "white"
        assert out["moves_detailed"][1]["side"] == "black"
        # Headers surface through
        assert out["headers"]["White"] == "Alice"
        assert out["headers"]["Black"] == "Bob"


@pytest.mark.anyio
async def test_load_pgn_empty_rejected():
    # Garbage / empty input must not silently reset the game.
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        await session.call_tool("add_move", {"uci": "e2e4"})
        pre = (await session.call_tool("get_status", {})).structuredContent["result"]
        assert pre["ply_count"] == 1

        resp = await session.call_tool("load_pgn", {"pgn": "garbage with no moves"})
        out = resp.structuredContent["result"]
        assert out["accepted"] is False
        assert out["reason"] == "empty_pgn"
        assert out["moves_applied"] == 0
        # Game must be unchanged — compare against pre to prove atomicity.
        assert out["status"]["ply_count"] == pre["ply_count"]
        assert out["status"]["last_move_uci"] == pre["last_move_uci"]
        assert out["status"]["fen"] == pre["fen"]


@pytest.mark.anyio
async def test_load_pgn_illegal_move_rejected():
    # PGN with an illegal SAN mid-game. python-chess stops at the bad move and
    # populates game.errors. We must reject the whole PGN — not partially apply.
    pgn = (
        '[Event "Bad"]\n[Result "*"]\n\n'
        "1. e4 e5 2. Ke3 *\n"  # Ke3 is illegal: king on e1 cannot reach e3 in one step
    )
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        await session.call_tool("add_move", {"uci": "d2d4"})
        pre = (await session.call_tool("get_status", {})).structuredContent["result"]
        assert pre["ply_count"] == 1

        resp = await session.call_tool("load_pgn", {"pgn": pgn})
        out = resp.structuredContent["result"]
        assert out["accepted"] is False
        assert out["reason"] == "illegal_move"
        assert "Ke3" in out["parse_error"]
        assert out["moves_applied"] == 0
        # Game must be unchanged — no partial replay. Compare against pre.
        assert out["status"]["ply_count"] == pre["ply_count"]
        assert out["status"]["last_move_uci"] == pre["last_move_uci"]
        assert out["status"]["fen"] == pre["fen"]


@pytest.mark.anyio
async def test_load_pgn_with_fen_startpos():
    # PGN with [FEN ...] + [SetUp "1"] headers must be replayed from that FEN,
    # not from the standard start. Black-to-move middlegame fragment.
    pgn = (
        '[Event "Fragment"]\n'
        '[SetUp "1"]\n'
        '[FEN "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"]\n'
        '[Result "*"]\n'
        "\n"
        "2. Nf3 Nc6 *\n"
    )
    async with run_client() as session:
        await session.call_tool("create_or_reset_game", {})
        resp = await session.call_tool("load_pgn", {"pgn": pgn})
        out = resp.structuredContent["result"]
        assert out["accepted"] is True
        assert out["moves_applied"] == 2
        assert out["starting_fen"].startswith(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR"
        )
        # After Nf3 Nc6 from that position, knights are on f3 and c6.
        assert out["status"]["pieces"]["f3"] == "N"
        assert out["status"]["pieces"]["c6"] == "n"
        assert out["moves_detailed"][0]["san"] == "Nf3"
        assert out["moves_detailed"][1]["san"] == "Nc6"
        assert (
            out["status"]["side_to_move"] == "white"
        )  # after Nf3 + Nc6, white to move
