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
