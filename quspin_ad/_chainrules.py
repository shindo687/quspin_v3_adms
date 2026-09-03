"""Small bundled ChainRules-compatible fallback for offline sidecar installs.

The official ``chainrules`` package is used whenever it is installed.  This
module implements the same narrow v0.1 protocol so that a wheel remains
usable in an isolated environment where that dependency is not mirrored.
It intentionally contains no numerical differentiation fallback.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Iterable, Mapping
from numbers import Real


class _Zero:
    __slots__ = ()

    def __repr__(self) -> str:
        return "ZERO"


ZERO = _Zero()


def _name(function: Callable[..., object]) -> str:
    return getattr(function, "__qualname__", repr(function))


class RuleNotFound(LookupError):
    def __init__(self, function: Callable[..., object], mode: str) -> None:
        super().__init__(f"No {mode.upper()} rule is registered for {_name(function)}")


class UnsupportedWrt(ValueError):
    def __init__(
        self,
        function: Callable[..., object],
        requested: Iterable[str],
        *,
        supported: Iterable[str] | None = None,
    ) -> None:
        self.function = function
        self.requested = tuple(sorted(requested))
        self.supported = None if supported is None else tuple(sorted(supported))
        message = (
            f"{_name(function)} does not support differentiation with respect to "
            f"{self.requested!r}"
        )
        if self.supported is not None:
            message += f"; supported inputs are {self.supported!r}"
        super().__init__(message)


class NonDifferentiablePoint(RuntimeError):
    pass


class RuleRegistry:
    def __init__(self) -> None:
        self._jvp: dict[int, tuple[Callable[..., object], Callable[..., object]]] = {}
        self._vjp: dict[int, tuple[Callable[..., object], Callable[..., object]]] = {}

    def _register(self, table, function):
        key = id(function)

        def decorator(rule):
            if key in table:
                raise RuntimeError(
                    f"A rule is already registered for {_name(function)}"
                )
            table[key] = (function, rule)
            return rule

        return decorator

    def jvp_for(self, function):
        return self._register(self._jvp, function)

    def vjp_for(self, function):
        return self._register(self._vjp, function)

    def _get(self, table, function, mode):
        entry = table.get(id(function))
        if entry is None or entry[0] is not function:
            raise RuleNotFound(function, mode)
        return entry[1]

    def get_jvp(self, function):
        return self._get(self._jvp, function, "JVP")

    def get_vjp(self, function):
        return self._get(self._vjp, function, "VJP")


rules = RuleRegistry()


def _signature_bind(function, args, kwargs):
    signature = inspect.signature(function)
    signature.bind(*args, **kwargs).apply_defaults()
    return signature


def _names(names, signature, label):
    names = (names,) if isinstance(names, str) else tuple(names)
    if not names:
        raise ValueError("wrt must contain at least one parameter name")
    if any(not isinstance(name, str) for name in names):
        raise TypeError("every name must be a string parameter name")
    if len(set(names)) != len(names):
        raise ValueError("wrt must contain unique parameter names")
    unknown = set(names) - set(signature.parameters)
    if unknown:
        raise TypeError(f"Unknown {label} parameter names: {sorted(unknown)!r}")
    return names


def jvp(function, /, *args, tangents, **kwargs):
    if not isinstance(tangents, Mapping):
        raise TypeError("tangents must be a mapping from parameter names to values")
    signature = _signature_bind(function, args, kwargs)
    _names(tuple(tangents), signature, "tangent")
    if not tangents or all(value is ZERO for value in tangents.values()):
        return function(*args, **kwargs), ZERO
    result = rules.get_jvp(function)(dict(tangents), *args, **kwargs)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("A JVP rule must return a two-tuple")
    return result


def vjp(function, /, *args, wrt, **kwargs):
    signature = _signature_bind(function, args, kwargs)
    names = _names(wrt, signature, "wrt")
    result = rules.get_vjp(function)(names, *args, **kwargs)
    if not isinstance(result, tuple) or len(result) != 2 or not callable(result[1]):
        raise TypeError("A VJP rule must return (value, pullback)")
    value, raw = result

    def pullback(cotangent):
        if cotangent is ZERO:
            return dict.fromkeys(names, ZERO)
        output = raw(cotangent)
        if not isinstance(output, Mapping) or set(output) != set(names):
            raise TypeError("Pullback keys must exactly match wrt")
        return {name: output[name] for name in names}

    return value, pullback


def grad(function, /, *args, wrt, **kwargs):
    value, pullback = vjp(function, *args, wrt=wrt, **kwargs)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("grad requires a single real scalar output")
    return pullback(1.0)


def value_and_grad(function, /, *args, wrt, **kwargs):
    value, pullback = vjp(function, *args, wrt=wrt, **kwargs)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("value_and_grad requires a single real scalar output")
    return value, pullback(1.0)


__version__ = "0.1.0"
sys.modules.setdefault("chainrules", sys.modules[__name__])
