from __future__ import annotations

import types

import fakes
import pytest
from workers import Request

from agents import Agent
from agents.lifecycle.websockets import WebSockets


def _manager(websockets=()):
    ctx = fakes.FakeCtx(websockets=list(websockets))
    return WebSockets(socket_source=ctx.getWebSockets)


def _canonical(
    connection_id: str,
    *,
    tags: list[str] | None = None,
    state: object = None,
    **extensions: object,
) -> dict[str, object]:
    return {
        "__pk": {
            "id": connection_id,
            "tags": tags or [connection_id],
            "uri": "https://example.com/agents/example/name",
        },
        "__user": state,
        **extensions,
    }


def test_new_socket_uses_namespaced_attachment():
    server = _manager()
    socket = fakes.FakeSocket()
    request = Request("https://example.com/agents/example/name")

    connection = server.wrap(socket, server.new_attachment(request.url))
    attachment = socket.deserializeAttachment().to_py()

    assert attachment == {
        "__pk": {
            "id": connection.id,
            "tags": [connection.id],
            "uri": request.url,
        },
        "__user": None,
    }


def test_canonical_update_preserves_platform_and_sibling_extensions():
    target = {"url": "https://example.com/sub", "headers": {}}
    socket = fakes.FakeSocket(
        {
            "__pk": {
                "id": "socket-1",
                "tags": ["socket-1"],
                "uri": "https://example.com/agents/example/name",
                "relaySession": "relay-1",
            },
            "__user": None,
            "target": target,
            "id": "sibling-extension",
        }
    )
    server = _manager([socket])

    [connection] = server.get_connections()
    connection.set_state({"ready": True})
    connection._insert_tags(["room:a"])

    updated = socket.deserializeAttachment().to_py()
    assert updated["__user"] == {"ready": True}
    assert updated["__pk"]["relaySession"] == "relay-1"
    assert updated["target"] == target
    assert updated["id"] == "sibling-extension"
    assert updated["__pk"]["tags"] == ["socket-1", "room:a"]


def test_legacy_attachment_is_read_without_eager_rewrite_then_migrated_on_update():
    legacy = {
        "id": "legacy",
        "state": {"before": True},
        "tags": ["room:a"],
        "target": {"url": "https://example.com/sub", "headers": {}},
    }
    socket = fakes.FakeSocket(legacy)
    server = _manager([socket])

    assert socket.deserializeAttachment().to_py() == legacy

    [connection] = server.get_connections()
    assert connection.id == "legacy"
    assert connection.state == {"before": True}
    assert connection.tags == ["legacy", "room:a"]
    assert socket.deserializeAttachment().to_py() == legacy

    connection.set_state({"after": True})

    assert socket.deserializeAttachment().to_py() == {
        "__pk": {"id": "legacy", "tags": ["legacy", "room:a"]},
        "__user": {"after": True},
        "target": legacy["target"],
    }


def test_hydration_skips_foreign_malformed_and_throwing_attachments():
    class ThrowingSocket(fakes.FakeSocket):
        def deserializeAttachment(self):
            raise RuntimeError("cannot decode")

    canonical = fakes.FakeSocket(_canonical("canonical"))
    legacy = fakes.FakeSocket({"id": "legacy", "state": {}, "tags": []})
    sockets = [
        fakes.FakeSocket(None),
        fakes.FakeSocket("not-an-object"),
        fakes.FakeSocket({"foreign": True}),
        fakes.FakeSocket({"__pk": "bad", "id": "must-not-fallback"}),
        fakes.FakeSocket({"__pk": {"id": 7}, "id": "must-not-fallback"}),
        ThrowingSocket(),
        canonical,
        legacy,
    ]
    server = _manager(sockets)

    assert [connection.id for connection in server.get_connections()] == [
        "canonical",
        "legacy",
    ]


@pytest.mark.parametrize(
    "tags",
    [
        ["socket-1", ""],
        ["socket-1", 7],
        ["socket-1", "x" * 257],
        ["socket-1", *[str(index) for index in range(10)]],
    ],
)
def test_hydration_skips_malformed_canonical_tags(tags):
    socket = fakes.FakeSocket(
        {
            "__pk": {"id": "socket-1", "tags": tags},
            "__user": None,
        }
    )
    manager = WebSockets(socket_source=lambda tag: [socket])

    assert manager.get_connections() == ()


def test_hydration_normalizes_canonical_attachment_without_stored_tags():
    socket = fakes.FakeSocket(
        {
            "__pk": {"id": "socket-1"},
            "__user": None,
        }
    )
    manager = WebSockets(socket_source=lambda tag: [socket])

    [connection] = manager.get_connections()

    assert connection.tags == ["socket-1"]


def test_agent_hydration_skips_relay_target_with_malformed_url():
    malformed = fakes.FakeSocket(
        _canonical(
            "malformed",
            target={"url": "https://[", "headers": {}},
        )
    )
    valid = fakes.FakeSocket(_canonical("valid"))
    agent = fakes.build_agent(websockets=[malformed, valid])

    assert [connection.id for connection in agent.get_connections()] == ["valid"]


def test_repeated_adoption_returns_one_wrapper_for_one_physical_socket():
    socket = fakes.FakeSocket(_canonical("socket-1"))
    server = _manager()

    first = server.adopt(socket)
    second = server.adopt(socket)

    assert first is second
    assert server.get_connections() == (first,)


def test_wrap_atomically_registers_with_the_owning_module():
    manager = WebSockets()
    socket = fakes.FakeSocket()
    attachment = manager.new_attachment("https://example.com/agents/example/name")

    wrapped = manager.wrap(socket, attachment)

    assert manager.adopt(socket) is wrapped
    assert manager.get_connections() == (wrapped,)


def test_state_snapshots_cannot_mutate_cached_or_hibernated_state():
    socket = fakes.FakeSocket(_canonical("socket-1"))
    first = _manager([socket])
    [connection] = first.get_connections()
    supplied = {"items": ["stored"]}

    connection.set_state(supplied)
    supplied["items"].append("caller")
    returned = connection.state
    returned["items"].append("reader")

    assert connection.state == {"items": ["stored"]}

    reincarnated = _manager([socket])
    [restored] = reincarnated.get_connections()
    assert restored.state == {"items": ["stored"]}


def test_state_updater_returns_normalized_snapshot_and_rejects_nan():
    manager = WebSockets()
    socket = fakes.FakeSocket()
    connection = manager.wrap(
        socket,
        manager.new_attachment("https://example.com/agents/example/name"),
    )

    committed = connection.set_state(lambda current: [current, {"ready": True}])
    committed.append("caller")

    assert connection.state == [None, {"ready": True}]
    with pytest.raises(ValueError):
        connection.set_state(float("nan"))
    assert connection.state == [None, {"ready": True}]


def test_distinct_proxy_wrappers_for_one_socket_reuse_the_adopted_connection():
    class SocketAlias:
        def __init__(self, socket):
            self.socket = socket

        def __eq__(self, other):
            return isinstance(other, SocketAlias) and self.socket is other.socket

        def send(self, data):
            self.socket.send(data)

        def close(self, code=1000, reason=""):
            self.socket.close(code, reason)

        def serializeAttachment(self, value):
            self.socket.serializeAttachment(value)

        def deserializeAttachment(self):
            return self.socket.deserializeAttachment()

    socket = fakes.FakeSocket(_canonical("socket-1"))
    hydrated_proxy = SocketAlias(socket)
    wake_proxy = SocketAlias(socket)
    server = _manager([hydrated_proxy])

    [hydrated] = server.get_connections()

    assert server.adopt(wake_proxy) is hydrated
    assert server.get_connections() == (hydrated,)


def test_duplicate_public_ids_keep_distinct_sockets_and_make_lookup_ambiguous():
    first = fakes.FakeSocket(_canonical("shared", tags=["shared", "one"]))
    second = fakes.FakeSocket(_canonical("shared", tags=["shared", "two"]))
    server = _manager([first, second])

    connections = server.get_connections()

    assert len(connections) == 2
    assert connections[0] is not connections[1]
    assert server.get_connections("one") == (connections[0],)
    assert server.get_connections("two") == (connections[1],)
    with pytest.raises(
        ValueError, match="More than one connection found for id shared"
    ):
        server.get_connection("shared")


def test_tag_index_tracks_mutation_and_cleanup():
    manager = WebSockets()
    socket = fakes.FakeSocket()
    connection = manager.wrap(
        socket,
        manager.new_attachment("https://example.com/agents/example/name"),
    )

    connection._insert_tags(["room:a"])
    assert manager.get_connections("room:a") == (connection,)

    connection._insert_tags(["room:b"])
    assert manager.get_connections("room:a") == ()
    assert manager.get_connections("room:b") == (connection,)

    manager.discard(connection)
    assert manager.get_connections("room:b") == ()


def test_canonical_namespace_takes_precedence_over_legacy_fields():
    socket = fakes.FakeSocket(
        {
            "__pk": {"id": None},
            "id": "legacy-looking",
            "state": {},
            "tags": [],
        }
    )
    server = _manager([socket])

    assert server.get_connections() == ()


def test_hidden_facet_socket_does_not_make_root_lookup_ambiguous():
    root_socket = fakes.FakeSocket(_canonical("shared"))
    facet_socket = fakes.FakeSocket(
        _canonical(
            "shared",
            target={"url": "https://example.com/sub", "headers": {}},
        )
    )
    agent = fakes.build_agent(websockets=[root_socket, facet_socket])

    connection = agent.get_connection("shared")

    assert connection is not None
    assert connection._server is root_socket
    assert agent.get_connections() == [connection]


def test_constructor_does_not_enumerate_sockets_and_lookup_degrades_on_failure():
    class FailingSocketContext(fakes.FakeCtx):
        def __init__(self):
            super().__init__()
            self.enumerations = 0

        def getWebSockets(self, tag=None):
            self.enumerations += 1
            raise RuntimeError("socket storage unavailable")

    ctx = FailingSocketContext()

    server = WebSockets(socket_source=ctx.getWebSockets)

    assert ctx.enumerations == 0
    assert server.get_connections() == ()
    assert ctx.enumerations == 1


def test_facet_does_not_adopt_root_owned_sockets():
    socket = fakes.FakeSocket(_canonical("root-socket"))
    ctx = fakes.FakeCtx(
        name="cf-agents:v2:leaf:0123456789abcdef",
        websockets=[socket],
    )

    child = Agent(ctx, types.SimpleNamespace())

    assert child.get_connections() == []
