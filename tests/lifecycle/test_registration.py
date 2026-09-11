from __future__ import annotations

from typing import Any, cast

import fakes
import pytest

from agents.lifecycle import (
    Lifecycle,
    LifecycleCapability,
    __all__,
)
from agents.lifecycle._runtime import _BUILTIN_CAPABILITY_IDS


def _capability(capability_id, label=None):
    capability_type = type(
        f"Capability_{label or capability_id}",
        (LifecycleCapability,),
        {"capability_id": capability_id, "label": label or capability_id},
    )
    return capability_type()


def _lifecycle() -> Lifecycle:
    return Lifecycle(fakes.FakeCtx(), host=object())


def test_lifecycle_module_exports_the_experimental_composition_surface():
    assert __all__ == [
        "CapabilityRequestContext",
        "CapabilityWebSocketCloseContext",
        "CapabilityWebSocketErrorContext",
        "CapabilityWebSocketMessageContext",
        "CapabilityWebSocketUpgradeContext",
        "CurrentLifecycleContext",
        "Lifecycle",
        "LifecycleCapability",
        "LifecycleEvent",
        "LifecycleEvents",
        "LifecycleHostContextScope",
        "LifecycleJob",
        "LifecycleJobContext",
        "LifecycleJobOutcome",
        "LifecycleJobPushOptions",
        "LifecycleJobReschedule",
        "LifecycleJobs",
        "LifecycleMemoryLimitContext",
        "LifecycleRetainedWork",
        "LifecycleRouteAddress",
        "LifecycleRouteContext",
        "LifecycleRouteEnvelope",
        "LifecycleRouteTransport",
        "LifecycleRoutes",
        "LifecycleServices",
        "LifecycleSockets",
        "LifecycleSql",
        "LifecycleStorage",
        "get_current_lifecycle_context",
    ]


def test_builtin_capability_ids_match_the_shared_contract():
    assert _BUILTIN_CAPABILITY_IDS == (
        "websockets",
        "scheduler",
        "tasks",
        "sessions",
        "mcp",
        "fibers",
    )


def test_host_capability_id_is_reserved():
    lifecycle = _lifecycle()
    lifecycle.use(_capability("host"))

    with pytest.raises(ValueError, match="reserved"):
        lifecycle._seal_registrations()


def test_use_binds_capability_to_one_lifecycle():
    lifecycle = _lifecycle()
    capability = _capability("tasks")

    with pytest.raises(RuntimeError, match="not installed"):
        _ = capability.lifecycle

    lifecycle.use(capability)

    assert capability.lifecycle is not lifecycle
    assert capability.lifecycle is capability.lifecycle
    assert lifecycle._seal_registrations() == (capability,)


def test_normal_capabilities_run_before_fallbacks_in_their_own_order():
    lifecycle = _lifecycle()
    fallback_one = _capability("fallback-one")
    normal_one = _capability("normal-one")
    fallback_two = _capability("fallback-two")
    normal_two = _capability("normal-two")

    lifecycle.use(fallback_one, fallback=True)
    lifecycle.use(normal_one)
    lifecycle.use(fallback_two, fallback=True)
    lifecycle.use(normal_two)

    assert lifecycle._seal_registrations() == (
        normal_one,
        normal_two,
        fallback_one,
        fallback_two,
    )


@pytest.mark.parametrize("capability_id", ["", "   ", None, 7])
def test_invalid_ids_are_delayed_until_registration_seals(capability_id):
    lifecycle = _lifecycle()
    capability = _capability(capability_id, "invalid")

    lifecycle.use(capability)

    with pytest.raises(ValueError, match="non-empty string"):
        lifecycle._seal_registrations()

    with pytest.raises(RuntimeError, match="closed"):
        lifecycle.use(_capability("late"))


def test_capability_id_descriptors_are_not_evaluated_during_registration():
    evaluated = []

    def read_id(_self):
        evaluated.append(True)
        raise RuntimeError("descriptor evaluated")

    descriptor_type = type(
        "DescriptorCapability",
        (LifecycleCapability,),
        {"capability_id": property(read_id)},
    )
    lifecycle = _lifecycle()
    lifecycle.use(descriptor_type())

    with pytest.raises(ValueError, match="non-empty string"):
        lifecycle._seal_registrations()
    assert evaluated == []


def test_capability_binding_does_not_read_an_instance_dict_descriptor():
    evaluated = []

    def read_dict(_self):
        evaluated.append(True)
        raise RuntimeError("instance dictionary evaluated")

    capability_type = type(
        "DescriptorCapability",
        (LifecycleCapability,),
        {
            "__slots__": (),
            "__dict__": property(read_dict),
            "capability_id": "safe",
        },
    )
    lifecycle = _lifecycle()
    capability = capability_type()

    lifecycle.use(capability)

    assert capability.lifecycle is capability.lifecycle
    assert lifecycle._seal_registrations() == (capability,)
    assert evaluated == []


@pytest.mark.parametrize("unsupported", [object(), LifecycleCapability])
def test_unsupported_capability_types_are_delayed_until_sealing(unsupported):
    lifecycle = _lifecycle()

    lifecycle.use(cast(Any, unsupported))

    with pytest.raises(ValueError, match="must inherit LifecycleCapability"):
        lifecycle._seal_registrations()


def test_unsupported_hostile_capability_is_not_inspected_by_use():
    evaluated = []

    class HostileCapability:
        @property
        def __class__(self):
            evaluated.append("class")
            raise RuntimeError("class evaluated")

    lifecycle = _lifecycle()
    lifecycle.use(cast(Any, HostileCapability()))

    with pytest.raises(ValueError, match="must inherit LifecycleCapability"):
        lifecycle._seal_registrations()
    assert evaluated == []


def test_hostile_capability_id_is_rejected_without_inspection():
    evaluated = []

    class HostileId:
        @property
        def __class__(self):
            evaluated.append("class")
            raise RuntimeError("class evaluated")

        def strip(self):
            evaluated.append("strip")
            raise RuntimeError("strip evaluated")

    capability_type = type(
        "HostileIdCapability",
        (LifecycleCapability,),
        {"capability_id": HostileId()},
    )
    lifecycle = _lifecycle()
    lifecycle.use(capability_type())

    with pytest.raises(ValueError, match="non-empty string"):
        lifecycle._seal_registrations()
    assert evaluated == []


def test_duplicate_id_is_rejected_when_registration_seals():
    lifecycle = _lifecycle()
    lifecycle.use(_capability("same", "first"))
    lifecycle.use(_capability("same", "second"), fallback=True)

    with pytest.raises(ValueError, match="duplicate capability_id: same"):
        lifecycle._seal_registrations()


def test_duplicate_instance_is_rejected_when_registration_seals():
    lifecycle = _lifecycle()
    capability = _capability("tasks")
    lifecycle.use(capability)
    lifecycle.use(capability)

    with pytest.raises(ValueError, match="instance is already installed"):
        lifecycle._seal_registrations()


def test_cross_host_reuse_keeps_the_first_binding():
    first = _lifecycle()
    second = _lifecycle()
    capability = _capability("tasks")

    first.use(capability)
    second.use(capability)

    first_services = capability.lifecycle
    assert first._seal_registrations() == (capability,)
    with pytest.raises(ValueError, match="another Lifecycle"):
        second._seal_registrations()
    assert capability.lifecycle is first_services


def test_capability_id_is_snapshotted_when_installed():
    lifecycle = _lifecycle()
    capability = _capability("stable")
    lifecycle.use(capability)
    type(capability).capability_id = ""

    assert lifecycle._seal_registrations() == (capability,)


def test_successful_seal_is_idempotent_and_registries_are_instance_local():
    first = _lifecycle()
    second = _lifecycle()
    first_capability = _capability("first")
    second_capability = _capability("second")
    first.use(first_capability)
    second.use(second_capability)

    first_order = first._seal_registrations()

    assert first._seal_registrations() is first_order
    assert first_order == (first_capability,)
    assert second._seal_registrations() == (second_capability,)


def test_failed_seal_remains_locked_and_repeatable():
    lifecycle = _lifecycle()
    lifecycle.use(_capability(""))

    for _ in range(2):
        with pytest.raises(ValueError, match="non-empty string"):
            lifecycle._seal_registrations()

    with pytest.raises(RuntimeError, match="closed"):
        lifecycle.use(_capability("late"))
