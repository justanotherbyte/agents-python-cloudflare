from __future__ import annotations

from typing import Any, cast

import fakes
import pytest

from agents.lifecycle import (
    Lifecycle,
    LifecycleCapability,
    LifecycleRouteAddress,
    LifecycleRouteContext,
    LifecycleRouteEnvelope,
    LifecycleRouteTransport,
    get_current_lifecycle_context,
)


class RouteCapability(LifecycleCapability):
    capability_id = "routes"

    def __init__(self):
        self.contexts: list[LifecycleRouteContext] = []

    async def on_route(self, context: LifecycleRouteContext) -> object:
        assert get_current_lifecycle_context() is None
        self.contexts.append(context)
        return {"received": context.payload}


def route_network():
    lifecycles: dict[str, Lifecycle] = {}
    envelopes: list[Any] = []

    async def transport(envelope: LifecycleRouteEnvelope) -> object:
        envelopes.append(envelope)
        target = lifecycles[envelope.target.key]
        return await target.route(
            version=envelope.version,
            source=envelope.source,
            target=envelope.target,
            capability_id=envelope.capability_id,
            payload=envelope.payload,
        )

    return lifecycles, envelopes, transport


@pytest.mark.asyncio
async def test_capability_routes_to_exact_address_and_owner_after_readiness():
    source = LifecycleRouteAddress("root", "root-data")
    target = LifecycleRouteAddress("root/facet", "facet-data")
    lifecycles, envelopes, transport = route_network()
    starts: list[str] = []

    async def receiver_start() -> None:
        starts.append("receiver")

    sender_lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        route_address=source,
        route_transport=transport,
    )
    receiver_lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        on_start=receiver_start,
        route_address=target,
        root_route_address=source,
        route_transport=transport,
    )
    sender = RouteCapability()
    receiver = RouteCapability()
    sender_lifecycle.use(sender)
    receiver_lifecycle.use(receiver)
    lifecycles[source.key] = sender_lifecycle
    lifecycles[target.key] = receiver_lifecycle

    result = await sender.lifecycle.routes.to(target, {"value": 1})

    assert result == {"received": {"value": 1}}
    assert starts == ["receiver"]
    assert sender.lifecycle.routes.source is source
    assert receiver.contexts == [LifecycleRouteContext(source, {"value": 1})]
    assert len(envelopes) == 1
    assert envelopes[0].version == 1
    assert envelopes[0].capability_id == "routes"


@pytest.mark.asyncio
async def test_facet_route_to_root_keeps_source_and_capability_scope():
    root = LifecycleRouteAddress("root", "root-data")
    facet = LifecycleRouteAddress("root/facet", "facet-data")
    lifecycles, _envelopes, transport = route_network()
    root_lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        route_address=root,
        route_transport=transport,
    )
    facet_lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        route_address=facet,
        root_route_address=root,
        route_transport=transport,
    )
    root_capability = RouteCapability()
    facet_capability = RouteCapability()
    root_lifecycle.use(root_capability)
    facet_lifecycle.use(facet_capability)
    lifecycles[root.key] = root_lifecycle
    lifecycles[facet.key] = facet_lifecycle

    await facet_capability.lifecycle.routes.to_root("payload")

    assert root_capability.contexts == [LifecycleRouteContext(facet, "payload")]


@pytest.mark.asyncio
async def test_unaddressed_root_routes_locally_without_a_source():
    lifecycle = Lifecycle(fakes.FakeCtx(), host=object())
    capability = RouteCapability()
    lifecycle.use(capability)

    assert await capability.lifecycle.routes.to_root("payload") == {
        "received": "payload"
    }
    assert capability.contexts == [LifecycleRouteContext(None, "payload")]


def test_route_address_data_is_opaque_transport_metadata():
    first = LifecycleRouteAddress("root/facet", "first")
    second = LifecycleRouteAddress("root/facet", "second")

    assert first == second
    assert hash(first) == hash(second)


def test_route_transport_is_a_public_type():
    _transport: LifecycleRouteTransport = route_network()[2]


@pytest.mark.asyncio
async def test_root_transport_rejects_addresses_outside_its_ownership():
    root = LifecycleRouteAddress("root", "root-data")
    owned = LifecycleRouteAddress("root/facet", "owned-data")
    foreign = LifecycleRouteAddress("foreign/facet", "foreign-data")
    transport = fakes.FakeRouteTransport("root", max_payload_bytes=64)
    transport.register(owned.key, "routes", lambda payload: payload)

    async def route(envelope: Any) -> object:
        return await transport.route(
            version=envelope.version,
            source=envelope.source.key,
            target=envelope.target.key,
            capability_id=envelope.capability_id,
            payload=envelope.payload,
        )

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        route_address=root,
        route_transport=route,
    )
    capability = RouteCapability()
    lifecycle.use(capability)

    assert await capability.lifecycle.routes.to(owned, {"ok": True}) == {"ok": True}
    with pytest.raises(PermissionError, match="outside the transport owner"):
        await capability.lifecycle.routes.to(foreign, None)


@pytest.mark.asyncio
async def test_incoming_route_rejects_invalid_version_target_and_capability():
    address = LifecycleRouteAddress("root", "root-data")
    foreign = LifecycleRouteAddress("foreign", "foreign-data")
    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        route_address=address,
    )
    capability = RouteCapability()
    lifecycle.use(capability)

    with pytest.raises(ValueError, match="unknown route version"):
        await lifecycle.route(
            version=2,
            source=address,
            target=address,
            capability_id="routes",
            payload=None,
        )
    with pytest.raises(PermissionError, match="does not belong"):
        await lifecycle.route(
            version=1,
            source=address,
            target=foreign,
            capability_id="routes",
            payload=None,
        )
    with pytest.raises(LookupError, match="unknown route capability"):
        await lifecycle.route(
            version=1,
            source=address,
            target=address,
            capability_id="missing",
            payload=None,
        )
    assert capability.contexts == []


@pytest.mark.asyncio
async def test_route_to_capability_without_handler_fails_without_fallback():
    class NoRouteCapability(LifecycleCapability):
        capability_id = "no-route"

    address = LifecycleRouteAddress("root", "root-data")
    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        route_address=address,
    )
    lifecycle.use(NoRouteCapability())
    fallback = RouteCapability()
    lifecycle.use(fallback, fallback=True)

    with pytest.raises(LookupError, match="cannot receive routes"):
        await lifecycle.route(
            version=1,
            source=address,
            target=address,
            capability_id="no-route",
            payload=None,
        )
    assert fallback.contexts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"too_long": "value"}, float("nan")])
async def test_outbound_routes_reject_oversized_and_non_strict_json(payload: object):
    address = LifecycleRouteAddress("root", "root-data")
    target = LifecycleRouteAddress("root/facet", "facet-data")
    transports = []

    async def transport(envelope: Any) -> None:
        transports.append(envelope)

    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        route_address=address,
        route_transport=transport,
        max_route_payload_bytes=4,
    )
    capability = RouteCapability()
    lifecycle.use(capability)

    with pytest.raises(ValueError):
        await capability.lifecycle.routes.to(target, payload)
    assert transports == []


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [True, 1.0, "1"])
async def test_incoming_route_requires_exact_integer_version(version: object):
    address = LifecycleRouteAddress("root", "root-data")
    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        route_address=address,
    )
    lifecycle.use(RouteCapability())

    with pytest.raises(ValueError, match="unknown route version"):
        await lifecycle.route(
            version=cast(Any, version),
            source=address,
            target=address,
            capability_id="routes",
            payload=None,
        )


@pytest.mark.asyncio
async def test_incoming_route_enforces_payload_limit():
    address = LifecycleRouteAddress("root", "root-data")
    lifecycle = Lifecycle(
        fakes.FakeCtx(),
        host=object(),
        route_address=address,
        max_route_payload_bytes=4,
    )
    lifecycle.use(RouteCapability())

    with pytest.raises(ValueError, match="configured byte limit"):
        await lifecycle.route(
            version=1,
            source=address,
            target=address,
            capability_id="routes",
            payload={"too_long": "value"},
        )
