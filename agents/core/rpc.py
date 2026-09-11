from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import FunctionType, MemberDescriptorType
from typing import Any, TypeVar, cast


@dataclass(frozen=True)
class _RPC_Func:
    func: Callable
    streaming: bool


@dataclass(frozen=True)
class _RPC_Metadata:
    streaming: bool


_RPC_METADATA_ATTR = "__agents_rpc_metadata__"
_F = TypeVar(
    "_F",
    bound=FunctionType | classmethod | staticmethod,
)
_CLASSMETHOD_FUNC = cast(
    MemberDescriptorType,
    classmethod.__dict__["__func__"],
)
_STATICMETHOD_FUNC = cast(
    MemberDescriptorType,
    staticmethod.__dict__["__func__"],
)


def _rpc_function(member: object) -> FunctionType | None:
    if type(member) is FunctionType:
        return member
    if type(member) is classmethod:
        function = MemberDescriptorType.__get__(_CLASSMETHOD_FUNC, member, classmethod)
        return function if type(function) is FunctionType else None
    if type(member) is staticmethod:
        function = MemberDescriptorType.__get__(
            _STATICMETHOD_FUNC, member, staticmethod
        )
        return function if type(function) is FunctionType else None
    return None


def rpc_callable(
    streaming: bool = False,
) -> Callable[[_F], _F]:
    def decorator(func: _F) -> _F:
        if _rpc_function(func) is None:
            raise TypeError("rpc_callable can only decorate methods")
        state = object.__getattribute__(func, "__dict__")
        state[_RPC_METADATA_ATTR] = _RPC_Metadata(streaming=streaming)
        return func

    return decorator


def _rpc_metadata(member: object) -> _RPC_Metadata | None:
    function = _rpc_function(member)
    if function is None:
        return None

    for candidate in (member, function):
        state = object.__getattribute__(candidate, "__dict__")
        if type(state) is dict:
            for name, metadata in state.items():
                if (
                    type(name) is str
                    and name == _RPC_METADATA_ATTR
                    and type(metadata) is _RPC_Metadata
                ):
                    return metadata
    return None


def _static_callable(member: object) -> bool:
    member_type = type(member)
    return member_type is classmethod or member_type is staticmethod or callable(member)


def _bind_rpc_member(member: object, instance: object) -> Callable[..., Any]:
    instance_type = type(instance)
    if type(member) is FunctionType:
        return FunctionType.__get__(member, instance, instance_type)
    if type(member) is classmethod:
        return classmethod.__get__(member, instance, instance_type)
    if type(member) is staticmethod:
        return staticmethod.__get__(member, instance, instance_type)
    raise TypeError("RPC definition is not a supported method")
