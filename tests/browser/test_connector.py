from __future__ import annotations

import asyncio

import pytest

from agents.browser import (
    BrowserConnector,
    BrowserSessionOptions,
    DurableBrowserSessionStore,
    StoredBrowserSession,
)

from .fakes import FakeBrowser, FakeStorage


def connector(
    browser: FakeBrowser,
    storage: FakeStorage,
    *,
    options: BrowserSessionOptions | None = None,
    now: list[int] | None = None,
) -> BrowserConnector:
    return BrowserConnector(
        browser,
        DurableBrowserSessionStore(storage),
        session=options,
        now_ms=(lambda: now[0]) if now is not None else None,
    )


def methods(browser: FakeBrowser, method: str):
    return [request for request in browser.requests if request.method == method]


@pytest.mark.asyncio
async def test_one_shot_is_idempotent_per_execution_and_terminal_delete_is_once():
    browser = FakeBrowser()
    storage = FakeStorage()
    cdp = connector(browser, storage)

    first, second = await asyncio.gather(
        cdp.send("exec-a", "Browser.getVersion"),
        cdp.send("exec-a", "Target.getTargets"),
    )
    assert first == {"echo": "Browser.getVersion"}
    assert second == {"echo": "Target.getTargets"}
    assert len(methods(browser, "POST")) == 1
    assert len([request for request in browser.requests if request.upgrade]) == 1
    assert "browser-session:cdp:exec:exec-a" in storage.values

    await cdp.dispose_execution("exec-a")
    await cdp.dispose_execution("exec-a")
    assert len(methods(browser, "DELETE")) == 1
    assert "browser-session:cdp:exec:exec-a" not in storage.values


@pytest.mark.asyncio
async def test_racing_reuse_connectors_delete_the_redundant_create():
    browser = FakeBrowser()
    storage = FakeStorage()
    options = BrowserSessionOptions(mode="reuse", key="team")
    one = connector(browser, storage, options=options)
    two = connector(browser, storage, options=options)
    gate = asyncio.Event()
    browser.create_gate = gate

    a = asyncio.create_task(one.send("a", "Browser.getVersion"))
    b = asyncio.create_task(two.send("b", "Browser.getVersion"))
    await asyncio.sleep(0)
    gate.set()
    await asyncio.gather(a, b)

    assert len(methods(browser, "POST")) == 2
    assert len(methods(browser, "DELETE")) == 1
    assert list(storage.values) == ["browser-session:cdp:reuse:team"]


@pytest.mark.asyncio
async def test_failed_redundant_create_cleanup_is_durable_until_sweep():
    browser = FakeBrowser()
    storage = FakeStorage()
    options = BrowserSessionOptions(mode="reuse", key="team")
    one = connector(browser, storage, options=options)
    two = connector(browser, storage, options=options)
    gate = asyncio.Event()
    browser.create_gate = gate
    browser.delete_statuses = [500, 204]

    first = asyncio.create_task(one.send("a", "Browser.getVersion"))
    second = asyncio.create_task(two.send("b", "Browser.getVersion"))
    await asyncio.sleep(0)
    gate.set()
    await asyncio.gather(first, second)
    cleanup_keys = [key for key in storage.values if ":delete:" in key]
    assert len(cleanup_keys) == 1
    assert storage.values[cleanup_keys[0]]["pendingDelete"] is True

    await one.sweep()
    assert cleanup_keys[0] not in storage.values
    assert len(methods(browser, "DELETE")) == 2


@pytest.mark.asyncio
async def test_racing_one_shot_connectors_commit_one_session_for_same_execution():
    browser = FakeBrowser()
    storage = FakeStorage()
    one = connector(browser, storage)
    two = connector(browser, storage)
    gate = asyncio.Event()
    browser.create_gate = gate

    a = asyncio.create_task(one.send("same", "Browser.getVersion"))
    b = asyncio.create_task(two.send("same", "Target.getTargets"))
    await asyncio.sleep(0)
    gate.set()
    await asyncio.gather(a, b)

    assert len(methods(browser, "POST")) == 2
    assert len(methods(browser, "DELETE")) == 1
    assert list(storage.values) == ["browser-session:cdp:exec:same"]


@pytest.mark.asyncio
async def test_reuse_shares_one_session_and_terminal_execution_keeps_it():
    browser = FakeBrowser()
    storage = FakeStorage()
    cdp = connector(browser, storage, options=BrowserSessionOptions(mode="reuse"))
    await cdp.send("a", "Browser.getVersion")
    await cdp.send("b", "Browser.getVersion")
    await cdp.dispose_execution("a")
    await cdp.dispose_execution("b")

    assert len(methods(browser, "POST")) == 1
    assert len(methods(browser, "DELETE")) == 0
    assert "browser-session:cdp:reuse:default" in storage.values


@pytest.mark.asyncio
async def test_dynamic_promotion_survives_terminal_and_is_reused_later():
    browser = FakeBrowser()
    storage = FakeStorage()
    cdp = connector(browser, storage, options=BrowserSessionOptions(mode="dynamic"))
    await cdp.send("a", "Browser.getVersion")
    info = await cdp.start_session("a")
    await cdp.dispose_execution("a")
    await cdp.send("b", "Target.getTargets")

    assert info.session_id == "session-1"
    assert len(methods(browser, "POST")) == 1
    assert storage.values["browser-session:cdp:reuse:default"]["sessionId"] == (
        "session-1"
    )


@pytest.mark.asyncio
async def test_pause_drops_socket_reconnects_same_session_and_reattaches_handle():
    browser = FakeBrowser()
    storage = FakeStorage()
    cdp = connector(browser, storage)
    handle = await cdp.attach_to_target("a", "target-1")
    await cdp.send("a", "Page.enable", session_id=handle)
    first = browser.sockets[0]
    first_live = next(
        message["sessionId"]
        for message in first.sent
        if message["method"] == "Page.enable"
    )

    await cdp.on_pass_end("a", "paused")
    await cdp.send("a", "Runtime.evaluate", session_id=handle)
    second = browser.sockets[1]
    second_live = next(
        message["sessionId"]
        for message in second.sent
        if message["method"] == "Runtime.evaluate"
    )

    assert len(methods(browser, "POST")) == 1
    assert first.closed == 1
    assert first_live != second_live
    assert any(message["method"] == "Target.attachToTarget" for message in second.sent)


@pytest.mark.asyncio
async def test_expired_execution_after_pause_fails_loudly_without_replacement():
    browser = FakeBrowser()
    storage = FakeStorage()
    storage.values["browser-session:cdp:exec:a"] = {
        "sessionId": "gone",
        "createdAt": 1,
        "updatedAt": 1,
    }
    browser.list_statuses = [410]
    cdp = connector(browser, storage)

    with pytest.raises(RuntimeError, match="expired or was swept"):
        await cdp.send("a", "Browser.getVersion")
    assert len(methods(browser, "POST")) == 0
    assert storage.values["browser-session:cdp:exec:a"]["closedAt"] > 0

    with pytest.raises(RuntimeError, match="expired or was swept"):
        await cdp.send("a", "Browser.getVersion")
    assert len(methods(browser, "POST")) == 0
    assert "browser-session:cdp:exec:a" in storage.values


@pytest.mark.asyncio
async def test_sweep_tombstones_exec_deletes_shared_and_ages_tombstone():
    browser = FakeBrowser()
    storage = FakeStorage()
    now = [100_000]
    storage.values.update(
        {
            "browser-session:cdp:exec:old": {
                "sessionId": "exec-old",
                "createdAt": 1,
                "updatedAt": 1,
            },
            "browser-session:cdp:reuse:default": {
                "sessionId": "shared-old",
                "createdAt": 1,
                "updatedAt": 1,
            },
            "browser-session:cdp:exec:fresh": {
                "sessionId": "fresh",
                "createdAt": 99_999,
                "updatedAt": 99_999,
            },
        }
    )
    cdp = connector(
        browser,
        storage,
        options=BrowserSessionOptions(mode="dynamic"),
        now=now,
    )
    swept = await cdp.sweep(max_idle_ms=10, max_exec_idle_ms=10)

    assert {(item.key, item.session_id) for item in swept} == {
        ("cdp:exec:old", "exec-old"),
        ("cdp:reuse:default", "shared-old"),
    }
    assert storage.values["browser-session:cdp:exec:old"]["closedAt"] == now[0]
    assert "browser-session:cdp:exec:fresh" in storage.values
    assert len(methods(browser, "DELETE")) == 2

    now[0] += 11
    second = await cdp.sweep(max_idle_ms=10, max_exec_idle_ms=10)
    assert second == (type(swept[0])("cdp:exec:fresh", "fresh"),)
    assert "browser-session:cdp:exec:old" not in storage.values
    assert len(methods(browser, "DELETE")) == 3


@pytest.mark.asyncio
async def test_explicit_close_and_reset_are_idempotent_and_network_outside_lock():
    browser = FakeBrowser()
    storage = FakeStorage()
    cdp = connector(browser, storage, options=BrowserSessionOptions(mode="reuse"))
    await cdp.send("a", "Browser.getVersion")
    await cdp.close_session()
    await cdp.close_session()
    reset = await cdp.reset_session("a")

    assert reset.session_id == "session-2"
    assert len(methods(browser, "DELETE")) == 1


@pytest.mark.asyncio
async def test_reset_completion_cannot_delete_a_concurrent_replacement():
    class GatedDeleteBrowser(FakeBrowser):
        def __init__(self):
            super().__init__()
            self.delete_started = asyncio.Event()
            self.delete_gate = asyncio.Event()

        async def fetch(self, request):
            if request.method == "DELETE":
                self.delete_started.set()
                await self.delete_gate.wait()
            return await super().fetch(request)

    browser = GatedDeleteBrowser()
    storage = FakeStorage()
    options = BrowserSessionOptions(mode="reuse")
    storage.values["browser-session:cdp:reuse:default"] = {
        "sessionId": "old",
        "createdAt": 1,
        "updatedAt": 1,
    }
    storage.values["browser-session:cdp:exec:a"] = {
        "sessionId": "old",
        "createdAt": 1,
        "updatedAt": 1,
    }
    cdp = connector(browser, storage, options=options)
    resetting = asyncio.create_task(cdp.reset_session("a"))
    await browser.delete_started.wait()
    assert "browser-session:cdp:exec:a" not in storage.values
    storage.values["browser-session:cdp:reuse:default"] = {
        "sessionId": "replacement",
        "createdAt": 2,
        "updatedAt": 2,
    }
    browser.delete_gate.set()

    result = await resetting
    assert result.session_id == "replacement"
    assert storage.values["browser-session:cdp:reuse:default"]["sessionId"] == (
        "replacement"
    )


@pytest.mark.asyncio
async def test_dynamic_promotion_rechecks_ids_after_a_reuse_race():
    browser = FakeBrowser()
    storage = FakeStorage()
    storage.values["browser-session:cdp:exec:a"] = {
        "sessionId": "execution",
        "createdAt": 1,
        "updatedAt": 1,
    }
    storage.values["browser-session:cdp:reuse:default"] = {
        "sessionId": "stale",
        "createdAt": 1,
        "updatedAt": 1,
    }
    cdp = connector(browser, storage, options=BrowserSessionOptions(mode="dynamic"))
    checks = 0

    async def alive(_stored):
        nonlocal checks
        checks += 1
        if checks == 2:
            storage.values["browser-session:cdp:reuse:default"] = {
                "sessionId": "replacement",
                "createdAt": 2,
                "updatedAt": 2,
            }
            return False
        return True

    cdp._is_alive = alive
    with pytest.raises(RuntimeError, match="healthy shared"):
        await cdp.start_session("a")
    assert storage.values["browser-session:cdp:reuse:default"]["sessionId"] == (
        "replacement"
    )
    assert methods(browser, "DELETE") == []


@pytest.mark.asyncio
async def test_live_view_is_ephemeral_selectable_and_never_stored():
    browser = FakeBrowser()
    browser.live_targets = True
    storage = FakeStorage()
    cdp = connector(browser, storage, options=BrowserSessionOptions(mode="reuse"))
    await cdp.send("a", "Browser.getVersion")
    view = await cdp.live_view(mode="devtools")
    selected = await cdp.live_view_url("a", mode="tab")

    assert view is not None
    assert view.targets[0].target_id == "target-1"
    assert "mode=devtools" in view.targets[0].url
    assert "mode=tab" in selected.url
    assert "live.browser.run" not in str(storage.values)


@pytest.mark.asyncio
async def test_recording_option_flows_only_at_initial_session_creation():
    browser = FakeBrowser()
    storage = FakeStorage()
    cdp = connector(
        browser,
        storage,
        options=BrowserSessionOptions(mode="reuse", recording=True),
    )
    await cdp.send("a", "Browser.getVersion")
    await cdp.on_pass_end("a", "paused")
    await cdp.send("a", "Browser.getVersion")

    creates = methods(browser, "POST")
    assert len(creates) == 1
    assert "recording=true" in creates[0].url
    reconnects = [request for request in browser.requests if request.upgrade]
    assert all("recording=" not in request.url for request in reconnects)


def test_kitesurf_rejects_every_durable_configuration():
    with pytest.raises(ValueError, match="one-shot"):
        BrowserSessionOptions(browser="kitesurf", mode="reuse")
    with pytest.raises(ValueError, match="one-shot"):
        BrowserSessionOptions(browser="kitesurf", mode="dynamic")
    with pytest.raises(ValueError, match="keep_alive"):
        BrowserSessionOptions(browser="kitesurf", keep_alive_ms=1)
    with pytest.raises(ValueError, match="recording"):
        BrowserSessionOptions(browser="kitesurf", recording=True)


@pytest.mark.asyncio
async def test_kitesurf_uses_direct_socket_and_pause_cannot_resume():
    browser = FakeBrowser()
    storage = FakeStorage()
    cdp = connector(
        browser,
        storage,
        options=BrowserSessionOptions(browser="kitesurf"),
    )
    await cdp.send("a", "Browser.getVersion")
    assert browser.requests[0].url.endswith("?browser=kitesurf")
    assert browser.requests[0].upgrade
    assert len(methods(browser, "POST")) == 0

    with pytest.raises(ValueError, match="Live View"):
        await cdp.live_view()
    with pytest.raises(ValueError, match="protocol discovery"):
        await cdp.spec()
    await cdp.on_pass_end("a", "paused")
    with pytest.raises(RuntimeError, match="cannot be resumed"):
        await cdp.send("a", "Browser.getVersion")
    assert len(browser.requests) == 1

    await cdp.dispose_execution("a")
    assert storage.values == {}


@pytest.mark.asyncio
async def test_dynamic_promotion_replaces_only_an_expired_fenced_shared_session():
    browser = FakeBrowser()
    storage = FakeStorage()
    now = [100]
    storage.values["browser-session:cdp:reuse:default"] = {
        "sessionId": "previous",
        "createdAt": 1,
        "updatedAt": 1,
    }
    storage.values["browser-session:cdp:exec:a"] = {
        "sessionId": "current",
        "createdAt": 2,
        "updatedAt": 2,
    }
    cdp = connector(
        browser,
        storage,
        options=BrowserSessionOptions(mode="dynamic"),
        now=now,
    )
    browser.list_statuses = [200, 410, 200]
    info = await cdp.start_session("a")
    assert info.session_id == "current"
    assert methods(browser, "DELETE")[0].url.endswith("/previous")
    assert storage.values["browser-session:cdp:reuse:default"]["sessionId"] == (
        "current"
    )


@pytest.mark.asyncio
async def test_dynamic_promotion_does_not_replace_a_healthy_shared_session():
    browser = FakeBrowser()
    storage = FakeStorage()
    storage.values["browser-session:cdp:reuse:default"] = {
        "sessionId": "healthy-shared",
        "createdAt": 1,
        "updatedAt": 1,
    }
    storage.values["browser-session:cdp:exec:a"] = {
        "sessionId": "current",
        "createdAt": 2,
        "updatedAt": 2,
    }
    cdp = connector(browser, storage, options=BrowserSessionOptions(mode="dynamic"))

    with pytest.raises(RuntimeError, match="healthy shared"):
        await cdp.start_session("a")
    assert storage.values["browser-session:cdp:reuse:default"]["sessionId"] == (
        "healthy-shared"
    )
    assert methods(browser, "DELETE") == []


@pytest.mark.asyncio
async def test_failed_terminal_delete_is_durable_and_sweep_retries_it():
    browser = FakeBrowser()
    storage = FakeStorage()
    cdp = connector(browser, storage)
    await cdp.send("a", "Browser.getVersion")
    browser.delete_statuses = [500, 204]

    await cdp.dispose_execution("a")
    pending = storage.values["browser-session:cdp:exec:a"]
    assert pending["pendingDelete"] is True
    assert pending["deleteAfterCleanup"] is True

    await cdp.sweep(max_exec_idle_ms=0)
    assert "browser-session:cdp:exec:a" not in storage.values
    assert len(methods(browser, "DELETE")) == 2


@pytest.mark.asyncio
async def test_cached_use_touches_record_and_tombstone_sweep_invalidates_socket():
    browser = FakeBrowser()
    storage = FakeStorage()
    now = [100]
    cdp = connector(browser, storage, now=now)
    await cdp.send("a", "Browser.getVersion")
    record = storage.values["browser-session:cdp:exec:a"]
    record["updatedAt"] = 1
    now[0] = 100_000

    await cdp.send("a", "Target.getTargets")
    assert storage.values["browser-session:cdp:exec:a"]["updatedAt"] == now[0]
    socket = browser.sockets[0]
    storage.values["browser-session:cdp:exec:a"].update(
        {
            "closedAt": now[0],
        }
    )
    await cdp.sweep(max_exec_idle_ms=1_000)
    assert socket.closed == 1
    assert storage.values["browser-session:cdp:exec:a"]["closedAt"] == now[0]
