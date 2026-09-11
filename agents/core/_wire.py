# TODO: move to utils.py
from __future__ import annotations

import json
from typing import Any


def strict_json_loads(value: object, label: str) -> Any:
    """Parse persisted or routed JSON text without non-standard constants."""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be JSON text")

    def reject_constant(constant: str) -> object:
        raise ValueError(f"invalid JSON constant: {constant}")

    try:
        return json.loads(value, parse_constant=reject_constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {label}") from error
    except ValueError:
        raise
    except RecursionError as error:
        raise ValueError("persisted JSON exceeds the nesting limit") from error
    except TypeError as error:
        raise ValueError(f"invalid {label}") from error
