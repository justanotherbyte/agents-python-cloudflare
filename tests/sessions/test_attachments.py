from __future__ import annotations

import base64
import hashlib

import fakes
import pytest

from agents.lifecycle import Lifecycle
from agents.sessions import Sessions


_ROW_BYTES = 1536 * 1024


async def _started():
    ctx = fakes.FakeCtx()
    sessions = Sessions()
    lifecycle = Lifecycle(ctx, host=object())
    lifecycle.use(sessions)
    await lifecycle.start()
    return sessions, ctx


def _data_url(media_type: str, payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _file_message(message_id: str, payload: bytes, media_type: str = "image/png"):
    return {
        "id": message_id,
        "role": "user",
        "parts": [
            {"type": "text", "text": "see attached"},
            {
                "type": "file",
                "mediaType": media_type,
                "filename": "attachment.bin",
                "url": _data_url(media_type, payload),
            },
        ],
    }


@pytest.mark.asyncio
async def test_attachment_is_addressed_stored_outside_the_message_and_hydrated():
    sessions, ctx = await _started()
    payload = b"hello"
    digest = hashlib.sha256(payload).hexdigest()
    pointer = f"attachment:sha256:{digest}"
    message = _file_message("message", payload)

    result = await sessions.session().append_message(message)

    assert result.message == message
    stored = ctx.storage.sql.exec(
        "SELECT content, content_chunks FROM cf_agents_session_messages"
    ).toArray()[0]
    assert pointer in stored["content"]
    assert _data_url("image/png", payload) not in stored["content"]
    assert stored["content_chunks"] == 0
    assert ctx.storage.sql.exec(
        "SELECT hash, bytes, media_type, chunks FROM cf_agents_session_attachment_meta"
    ).toArray() == [
        {"hash": digest, "bytes": 5, "media_type": "image/png", "chunks": 1}
    ]
    assert ctx.storage.sql.exec(
        "SELECT hash, idx, data FROM cf_agents_session_attachment_chunks"
    ).toArray() == [{"hash": digest, "idx": 0, "data": payload}]
    assert ctx.storage.sql.exec(
        "SELECT session_id, message_id, hash FROM cf_agents_session_attachment_refs"
    ).toArray() == [{"session_id": "", "message_id": "message", "hash": digest}]
    assert await sessions.session().get_message("message") == message


@pytest.mark.asyncio
async def test_text_and_invalid_inline_data_are_not_extracted():
    sessions, ctx = await _started()
    text = _file_message("text", b"plain", "text/plain")
    invalid = {
        "id": "invalid",
        "role": "user",
        "parts": [
            {
                "type": "file",
                "mediaType": "image/png",
                "url": "https://example.com/image.png",
                "data": base64.b64encode(b"fallback").decode("ascii"),
            },
            {
                "type": "file",
                "mediaType": "image/png",
                "url": "data:image/png,percent-encoded",
            },
        ],
    }

    await sessions.session().append_message(text)
    await sessions.session().append_message(invalid)

    assert (
        ctx.storage.sql.exec(
            "SELECT hash FROM cf_agents_session_attachment_meta"
        ).toArray()
        == []
    )
    assert await sessions.session().get_message("text") == text
    assert await sessions.session().get_message("invalid") == invalid


@pytest.mark.asyncio
async def test_base64_forms_and_media_type_rules_match_browser_decoding():
    sessions, ctx = await _started()
    message = {
        "id": "base64",
        "role": "user",
        "parts": [
            {"type": "file", "url": "data:;base64,YQ"},
            {
                "type": "media",
                "mediaType": "image/png",
                "data": " Ym\nM= ",
            },
            {
                "type": "file",
                "mediaType": "TEXT/plain",
                "url": _data_url("TEXT/plain", b"def"),
            },
            {
                "type": "file",
                "mediaType": "image/png",
                "url": "data:image/png;base64,YQ=",
            },
        ],
    }

    await sessions.session().append_message(message)

    assert ctx.storage.sql.exec(
        "SELECT bytes, media_type FROM cf_agents_session_attachment_meta ORDER BY bytes"
    ).toArray() == [
        {"bytes": 1, "media_type": "application/octet-stream"},
        {"bytes": 2, "media_type": "image/png"},
        {"bytes": 3, "media_type": "TEXT/plain"},
    ]
    read = await sessions.session().get_message("base64")
    assert read["parts"] == [
        {
            "type": "file",
            "url": "data:application/octet-stream;base64,YQ==",
            "mediaType": "application/octet-stream",
        },
        {"type": "media", "mediaType": "image/png", "data": "YmM="},
        {
            "type": "file",
            "mediaType": "TEXT/plain",
            "url": _data_url("TEXT/plain", b"def"),
        },
        {
            "type": "file",
            "mediaType": "image/png",
            "url": "data:image/png;base64,YQ=",
        },
    ]


@pytest.mark.asyncio
async def test_payloads_are_deduplicated_globally_and_live_until_the_last_reference():
    sessions, ctx = await _started()
    payload = b"shared" * 20_000
    first = sessions.session("first")
    second = sessions.session("second")
    await first.append_message(_file_message("one", payload))
    await second.append_message(_file_message("two", payload, "application/pdf"))

    assert ctx.storage.sql.exec(
        "SELECT COUNT(*) AS count FROM cf_agents_session_attachment_meta"
    ).toArray() == [{"count": 1}]
    assert ctx.storage.sql.exec(
        "SELECT COUNT(*) AS count FROM cf_agents_session_attachment_refs"
    ).toArray() == [{"count": 2}]

    await first.delete_messages(["one"])
    assert ctx.storage.sql.exec(
        "SELECT COUNT(*) AS count FROM cf_agents_session_attachment_meta"
    ).toArray() == [{"count": 1}]
    assert (await second.get_message("two"))["parts"][1]["url"].startswith(
        "data:image/png;base64,"
    )

    await second.clear_messages()
    for table in (
        "cf_agents_session_attachment_meta",
        "cf_agents_session_attachment_chunks",
        "cf_agents_session_attachment_refs",
    ):
        assert ctx.storage.sql.exec(
            f"SELECT COUNT(*) AS count FROM {table}"
        ).toArray() == [{"count": 0}]


@pytest.mark.asyncio
async def test_attachment_chunks_cover_zero_exact_and_over_boundary_payloads():
    sessions, ctx = await _started()
    payloads = [b"", b"a" * _ROW_BYTES, b"b" * (_ROW_BYTES + 1)]
    message = {
        "id": "boundaries",
        "role": "user",
        "parts": [
            {
                "type": "file",
                "mediaType": "application/octet-stream",
                "data": base64.b64encode(payload).decode("ascii"),
            }
            for payload in payloads
        ],
    }

    await sessions.session().append_message(message)

    metadata = ctx.storage.sql.exec(
        "SELECT bytes, chunks FROM cf_agents_session_attachment_meta ORDER BY bytes"
    ).toArray()
    assert metadata == [
        {"bytes": 0, "chunks": 0},
        {"bytes": _ROW_BYTES, "chunks": 1},
        {"bytes": _ROW_BYTES + 1, "chunks": 2},
    ]
    chunk_sizes = ctx.storage.sql.exec(
        "SELECT hash, idx, LENGTH(data) AS bytes "
        "FROM cf_agents_session_attachment_chunks ORDER BY hash, idx"
    ).toArray()
    sizes_by_hash = {}
    for row in chunk_sizes:
        sizes_by_hash.setdefault(row["hash"], []).append((row["idx"], row["bytes"]))
    assert sorted(sizes_by_hash.values()) == [
        [(0, _ROW_BYTES)],
        [(0, _ROW_BYTES), (1, 1)],
    ]
    assert ctx.storage.sql.exec(
        "SELECT content_chunks FROM cf_agents_session_messages"
    ).toArray() == [{"content_chunks": 0}]
    assert await sessions.session().get_message("boundaries") == message


@pytest.mark.asyncio
async def test_nested_media_is_extracted_only_through_the_depth_limit():
    sessions, ctx = await _started()
    shallow_media = {
        "type": "media",
        "mediaType": "image/png",
        "data": base64.b64encode(b"shallow").decode("ascii"),
    }
    deep_media = {
        "type": "media",
        "mediaType": "image/png",
        "data": base64.b64encode(b"deep").decode("ascii"),
    }
    shallow = shallow_media
    deep = deep_media
    for _ in range(6):
        shallow = {"nested": shallow}
    for _ in range(7):
        deep = {"nested": deep}
    message = {
        "id": "nested",
        "role": "assistant",
        "parts": [
            {"type": "tool-capture", "output": shallow},
            {"type": "tool-capture", "output": deep},
        ],
    }

    await sessions.session().append_message(message)

    assert ctx.storage.sql.exec(
        "SELECT bytes FROM cf_agents_session_attachment_meta"
    ).toArray() == [{"bytes": len(b"shallow")}]
    assert await sessions.session().get_message("nested") == message


@pytest.mark.asyncio
async def test_attachment_expansion_counts_toward_history_budgets():
    sessions, ctx = await _started()
    session = sessions.session()
    for index in range(3):
        await session.append_message(
            _file_message(f"message-{index}", bytes([index]) * 400_000)
        )

    stats = await session.get_history_row_stats()
    stored_rows = ctx.storage.sql.exec(
        "SELECT id, LENGTH(CAST(content AS BLOB)) AS bytes "
        "FROM cf_agents_session_messages ORDER BY seq"
    ).toArray()
    for stat, stored in zip(stats, stored_rows, strict=True):
        assert stat.bytes == stored["bytes"] + 533_336
    recent = await session.get_recent_history(stats[1].bytes + stats[2].bytes)
    assert [message["id"] for message in recent.messages] == [
        "message-1",
        "message-2",
    ]
    assert recent.truncated is True


@pytest.mark.asyncio
async def test_update_import_and_duplicate_import_own_attachment_references():
    sessions, ctx = await _started()
    session = sessions.session()
    first = _file_message("message", b"first")
    second = _file_message("message", b"second")
    await session.append_message(first)
    changes = ctx.storage.sql.exec("SELECT total_changes() AS count").toArray()[0][
        "count"
    ]

    assert await session.update_message(first) == first
    assert ctx.storage.sql.exec("SELECT total_changes() AS count").toArray() == [
        {"count": changes}
    ]
    assert await session.update_message(second) == second
    assert ctx.storage.sql.exec(
        "SELECT hash FROM cf_agents_session_attachment_meta"
    ).toArray() == [{"hash": hashlib.sha256(b"second").hexdigest()}]

    imported = _file_message("imported", b"historical", "application/pdf")
    await session.import_message(imported, parent_id=None, created_at_ms=10)
    assert await session.get_message("imported") == imported
    await session.import_message(
        _file_message("imported", b"orphan"),
        parent_id=None,
        created_at_ms=11,
    )
    assert ctx.storage.sql.exec(
        "SELECT hash FROM cf_agents_session_attachment_meta ORDER BY hash"
    ).toArray() == [
        {"hash": digest}
        for digest in sorted(
            (
                hashlib.sha256(b"second").hexdigest(),
                hashlib.sha256(b"historical").hexdigest(),
            )
        )
    ]
    assert await session.update_message(_file_message("missing", b"missing")) is None


@pytest.mark.asyncio
async def test_attachment_mutations_roll_back_with_message_mutations(monkeypatch):
    sessions, ctx = await _started()
    session = sessions.session()
    execute = ctx.storage.sql.exec

    def fail_chunk_write(query, *params):
        if "INSERT INTO cf_agents_session_attachment_chunks" in query:
            raise RuntimeError("attachment chunk failed")
        return execute(query, *params)

    monkeypatch.setattr(ctx.storage.sql, "exec", fail_chunk_write)
    with pytest.raises(RuntimeError, match="attachment chunk failed"):
        await session.append_message(_file_message("failed", b"payload"))
    assert execute("SELECT id FROM cf_agents_session_messages").toArray() == []
    assert execute("SELECT hash FROM cf_agents_session_attachment_meta").toArray() == []

    monkeypatch.undo()
    message = _file_message("stored", b"stored")
    await session.append_message(message)

    def fail_collection(query, *params):
        if "DELETE FROM cf_agents_session_attachment_meta" in query:
            raise RuntimeError("attachment collection failed")
        return execute(query, *params)

    monkeypatch.setattr(ctx.storage.sql, "exec", fail_collection)
    with pytest.raises(RuntimeError, match="attachment collection failed"):
        await session.delete_messages(["stored"])
    assert await session.get_message("stored") == message
    assert execute(
        "SELECT COUNT(*) AS count FROM cf_agents_session_attachment_refs"
    ).toArray() == [{"count": 1}]

    with pytest.raises(RuntimeError, match="attachment collection failed"):
        await session.clear_messages()
    assert await session.get_message("stored") == message

    monkeypatch.undo()
    monkeypatch.setattr(ctx.storage.sql, "exec", fail_chunk_write)
    with pytest.raises(RuntimeError, match="attachment chunk failed"):
        await session.update_message(_file_message("stored", b"replacement"))
    assert await session.get_message("stored") == message
    with pytest.raises(RuntimeError, match="attachment chunk failed"):
        await session.import_message(
            _file_message("import-failed", b"historical"),
            parent_id=None,
            created_at_ms=1,
        )
    assert (
        execute(
            "SELECT id FROM cf_agents_session_messages WHERE id = 'import-failed'"
        ).toArray()
        == []
    )


@pytest.mark.asyncio
async def test_missing_or_malformed_attachment_rows_leave_pointers_unresolved():
    sessions, ctx = await _started()
    session = sessions.session()
    first = _file_message("first", b"first")
    second = _file_message("second", b"second")
    await session.append_message(first)
    await session.append_message(second)
    first_hash = hashlib.sha256(b"first").hexdigest()
    second_hash = hashlib.sha256(b"second").hexdigest()
    forged = {
        "id": "forged",
        "role": "user",
        "parts": [
            {
                "type": "file",
                "mediaType": "image/png",
                "url": f"attachment:sha256:{second_hash}",
            }
        ],
    }
    await session.append_message(forged)
    assert await session.get_message("forged") == forged
    shared_window = await session.get_history()
    assert shared_window[1] == second
    assert shared_window[2] == forged
    ctx.storage.sql.exec(
        "DELETE FROM cf_agents_session_attachment_meta WHERE hash = ?", first_hash
    )
    ctx.storage.sql.exec(
        "UPDATE cf_agents_session_attachment_chunks SET idx = 2 WHERE hash = ?",
        second_hash,
    )

    assert (await session.get_message("first"))["parts"][1][
        "url"
    ] == f"attachment:sha256:{first_hash}"
    assert (await session.get_message("second"))["parts"][1][
        "url"
    ] == f"attachment:sha256:{second_hash}"

    repaired = _file_message("repaired", b"first")
    await session.append_message(repaired)
    assert await session.get_message("first") == first
    assert await session.get_message("repaired") == repaired

    ctx.storage.sql.exec(
        "UPDATE cf_agents_session_attachment_chunks SET idx = 0, data = ? "
        "WHERE hash = ?",
        b"xxxxxx",
        second_hash,
    )
    assert (await session.get_message("second"))["parts"][1][
        "url"
    ] == f"attachment:sha256:{second_hash}"


@pytest.mark.asyncio
async def test_file_attachment_token_estimates_match_the_target_heuristic():
    sessions, ctx = await _started()
    session = sessions.session()
    image = {
        "id": "image",
        "role": "user",
        "parts": [
            {
                "type": "file",
                "mediaType": "image/png",
                "url": _data_url("image/png", b"x"),
            }
        ],
    }
    document = {
        "id": "document",
        "role": "user",
        "parts": [
            {
                "type": "file",
                "mediaType": "application/pdf",
                "url": _data_url("application/pdf", b"x" * 800),
            }
        ],
    }
    astral = {
        "id": "astral",
        "role": "user",
        "parts": [
            {
                "type": "file",
                "mediaType": "application/pdf",
                "url": f"data:application/pdf;base64,{'🙂' * 5}",
            }
        ],
    }

    await session.append_message(image)
    await session.append_message(document)
    await session.append_message(astral)

    assert ctx.storage.sql.exec(
        "SELECT id, token_estimate FROM cf_agents_session_messages ORDER BY seq"
    ).toArray() == [
        {"id": "image", "token_estimate": 1_604},
        {"id": "document", "token_estimate": 205},
        {"id": "astral", "token_estimate": 6},
    ]
