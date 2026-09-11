class HookError(Exception):
    """Wraps any exception raised inside a user lifecycle hook.

    The dispatch that invokes a hook catches whatever the hook raised, wraps it
    in this type, and reports it once through ``on_error`` before deciding to
    swallow or propagate. Carrying a single generic type lets a boundary tell
    "a user hook failed and was already reported" apart from a raw failure in
    the framework's own plumbing, so it can propagate the former without
    reporting it a second time. The original is preserved on ``original`` and
    chained as ``__cause__`` so a printed traceback shows both frames.
    """

    def __init__(self, hook: str, original: BaseException):
        super().__init__(f"error in hook {hook!r}: {original}")
        self.hook = hook
        self.original = original


class RPCError(Exception):
    """Raised when an RPC frame names a method the agent cannot dispatch.

    Distinct from an exception raised inside a user's RPC method body: this
    signals the RPC protocol layer itself could not resolve the call. The
    wire treatment is the same either way — a ``{success: false, error}``
    frame — but the type separates a protocol misuse from a bug in the
    method's own code.
    """


class RoutingException(Exception):
    """Raised when a request cannot be dispatched to a Durable Object namespace:
    a matched route names a binding the environment does not expose, or a
    sub-agent hop cannot resolve the capability, class handle, or child it names.
    """
