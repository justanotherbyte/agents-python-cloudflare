# TODO: move to utils.py
from __future__ import annotations

from collections.abc import Callable
from types import FunctionType, GetSetDescriptorType
from typing import Any, cast

_TYPE_NAMESPACE = cast(GetSetDescriptorType, type.__dict__["__dict__"])
_TYPE_MRO = cast(GetSetDescriptorType, type.__dict__["__mro__"])


def static_mro_members(instance: object) -> dict[str, object]:
    """Return effective class members without evaluating descriptors."""
    members: dict[str, object] = {}
    instance_type = type(instance)
    mro = GetSetDescriptorType.__get__(_TYPE_MRO, instance_type, type)
    for cls in mro:
        namespace = GetSetDescriptorType.__get__(_TYPE_NAMESPACE, cls, type)
        for name, value in namespace.items():
            if type(name) is not str:
                continue
            if name not in members:
                members[name] = value
    return members


def static_definition_function(member: object) -> FunctionType | None:
    """Return the function behind a supported method without binding it."""
    if type(member) is FunctionType:
        return member
    if type(member) is classmethod or type(member) is staticmethod:
        function = object.__getattribute__(member, "__func__")
        return function if type(function) is FunctionType else None
    return None


def deferred_method(member: object, instance: object) -> Callable[..., Any]:
    """Bind a statically discovered method only when it is invoked."""

    def invoke(*args: object, **kwargs: object) -> Any:
        instance_type = type(instance)
        if type(member) is FunctionType:
            method = FunctionType.__get__(member, instance, instance_type)
        elif type(member) is classmethod:
            method = classmethod.__get__(member, instance, instance_type)
        elif type(member) is staticmethod:
            method = staticmethod.__get__(member, instance, instance_type)
        else:
            raise TypeError("definition is not a supported method")
        return method(*args, **kwargs)

    return invoke
