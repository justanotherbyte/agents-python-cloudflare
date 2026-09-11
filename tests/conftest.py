"""Shared fixtures. The module stubs are installed at import, before anything
pulls in `agents`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import _runtime_stubs

_runtime_stubs.install()

import fakes
import pytest
import workers


@pytest.fixture
def sql() -> Callable[..., list[dict[str, Any]]]:
    """A standalone `sql(query, *params)` over a fresh in-memory database."""
    return fakes.make_sql()


@pytest.fixture
def fake_connection() -> fakes.FakeConnection:
    return fakes.FakeConnection()


@pytest.fixture
def make_connection() -> Callable[..., fakes.FakeConnection]:
    return fakes.FakeConnection


@pytest.fixture
def fake_socket() -> Callable[..., fakes.FakeSocket]:
    return fakes.FakeSocket


@pytest.fixture
def make_agent() -> Callable[..., Any]:
    return fakes.build_agent


@pytest.fixture
def make_chat_agent() -> Callable[..., Any]:
    return fakes.build_chat_agent


@pytest.fixture
def wait_until():
    """Swap workers.waitUntil for a recorder for the duration of a test."""
    recorder = fakes.WaitUntilRecorder()
    previous = workers.waitUntil
    workers.waitUntil = recorder
    try:
        yield recorder
    finally:
        recorder.close()
        workers.waitUntil = previous
