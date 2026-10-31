"""General-purpose Proxy wrapper for Node with all dunders."""

import operator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .node import FunctionCallNode, GetAttributeNode, MethodCallNode, Node

T = TypeVar("T")

_NOT_EVALUATED = object()


def _maybe_value_proxy(node, value):
    """Wrap a concretely-known `value` for `node`, unless it is itself a Proxy.

    A Proxy result already carries its own (more precise) node, so wrapping it
    would only hide its behavior behind ValueProxy, which does not forward
    calls or attribute semantics.
    """
    if isinstance(value, Proxy):
        return value
    return ValueProxy(node, value)


def _known_getattr(node, source_value, attr):
    """Attribute access on a proxy whose source value is concretely known.

    A fetched Proxy is returned as-is (it records its own node); a fetched
    method becomes an AttributeProxy so its calls evaluate on the source value;
    anything else is a known data attribute and stays value-aware.
    """
    try:
        fetched = getattr(source_value, attr)
    except Exception:
        return AttributeProxy(node, source_value)
    if isinstance(fetched, Proxy):
        return fetched
    if callable(fetched):
        return AttributeProxy(node, source_value)
    return ValueProxy(node, fetched)


def _try_evaluate(func, *args, **kwargs):
    """Evaluate `func` on the concrete values of `args`, if all are known.

    Returns _NOT_EVALUATED if any operand has no known value, or if the
    evaluation raises.
    """
    values = []
    for a in args:
        if isinstance(a, ValueProxy):
            values.append(a.value)
        elif isinstance(a, Proxy):
            return _NOT_EVALUATED
        else:
            values.append(a)
    kwvalues = {}
    for k, v in kwargs.items():
        if isinstance(v, ValueProxy):
            kwvalues[k] = v.value
        elif isinstance(v, Proxy):
            return _NOT_EVALUATED
        else:
            kwvalues[k] = v
    try:
        return func(*values, **kwvalues)
    except Exception:
        return _NOT_EVALUATED


@dataclass
class Proxy(Generic[T]):
    """General-purpose wrapper for Node that provides all dunder methods."""

    node: Node

    # Tell numpy to defer (ndarray + Proxy -> Proxy.__radd__) instead of probing the
    # Proxy as an array, which would fabricate a bogus __array_struct__ via __getattr__.
    __array_ufunc__ = None

    def __repr__(self):
        return f"Proxy({self.node!r})"

    def __getattr__(self, attr: str) -> "AttributeProxy":
        node = GetAttributeNode(source=self.node, attribute_name=attr)
        # __dict__ rather than getattr, to avoid recursing through __getattr__
        source_value = self.__dict__.get("value", _NOT_EVALUATED)
        if isinstance(self, ValueProxy) and source_value is not _NOT_EVALUATED:
            return _known_getattr(node, source_value, attr)
        return AttributeProxy(node, source_value)

    def __call__(self, *args, **kwargs) -> "Proxy":
        raise NotImplementedError("Proxy.__call__ is not implemented")

    def __len__(self) -> int:
        raise ValueError(
            "Tracing does not allow __len__ since real values are not evaluated"
        )

    def __iter__(self):
        raise ValueError(
            "Proxy does not support __iter__. Use explicit indexing instead."
        )

    def __getitem__(self, idx) -> "Proxy":
        idx_node = idx.node if isinstance(idx, Proxy) else idx
        getitem_node = FunctionCallNode(
            func=operator.getitem,
            args=(self.node, idx_node),
            kwargs={},
        )
        value = _try_evaluate(operator.getitem, self, idx)
        if value is not _NOT_EVALUATED:
            return _maybe_value_proxy(getitem_node, value)
        return Proxy(getitem_node)

    def __bool__(self):
        raise ValueError(
            "Base Proxy does not allow __bool__ during tracing since real values are unknown"
        )


NODE_DUNDER_METHODS = {
    "__add__": operator.add,
    "__sub__": operator.sub,
    "__mul__": operator.mul,
    "__truediv__": operator.truediv,
    "__floordiv__": operator.floordiv,
    "__mod__": operator.mod,
    "__pow__": operator.pow,
    "__lshift__": operator.lshift,
    "__rshift__": operator.rshift,
    "__and__": operator.and_,
    "__xor__": operator.xor,
    "__or__": operator.or_,
    "__neg__": operator.neg,
    "__pos__": operator.pos,
    "__abs__": operator.abs,
    "__invert__": operator.invert,
    "__eq__": operator.eq,
    "__ne__": operator.ne,
    "__lt__": operator.lt,
    "__le__": operator.le,
    "__gt__": operator.gt,
    "__ge__": operator.ge,
    "__matmul__": operator.matmul,
}

NODE_REFLECTABLE_METHODS = [
    "add",
    "sub",
    "mul",
    "floordiv",
    "truediv",
    "div",
    "mod",
    "pow",
    "lshift",
    "rshift",
    "and_",
    "or_",
    "xor",
    "getitem",
    "matmul",
]


def _add_proxy_operator(cls, name, operator_func):
    def proxy_method(self, *args, **kwargs):
        # Convert any Proxy args to their underlying nodes
        node_args = tuple(arg.node if isinstance(arg, Proxy) else arg for arg in args)
        node_kwargs = {
            k: v.node if isinstance(v, Proxy) else v for k, v in kwargs.items()
        }
        node = FunctionCallNode(
            func=operator_func,
            args=(self.node, *node_args),
            kwargs=node_kwargs,
        )
        value = _try_evaluate(operator_func, self, *args, **kwargs)
        if value is not _NOT_EVALUATED:
            return _maybe_value_proxy(node, value)
        return Proxy(node)

    setattr(cls, name, proxy_method)


def _add_proxy_reflection(cls, name: str):
    # __rmul__(self, rhs) means rhs * self — use the same operator but swap arg order
    fwd_dunder = f"__{name.rstrip('_')}__"
    operator_func = NODE_DUNDER_METHODS.get(fwd_dunder)
    if operator_func is None:
        return  # no matching forward op, skip

    def proxy_method(self, rhs):
        rhs_node = rhs.node if isinstance(rhs, Proxy) else rhs
        node = FunctionCallNode(
            func=operator_func,
            args=(rhs_node, self.node),
            kwargs={},
        )
        value = _try_evaluate(operator_func, rhs, self)
        if value is not _NOT_EVALUATED:
            return _maybe_value_proxy(node, value)
        return Proxy(node)

    setattr(cls, f"__r{name}__", proxy_method)


# Add all dunder methods to Proxy
for name, operator_func in NODE_DUNDER_METHODS.items():
    _add_proxy_operator(Proxy, name, operator_func)

# Add reflected methods
for name in NODE_REFLECTABLE_METHODS:
    _add_proxy_reflection(Proxy, name)


@dataclass(eq=False)
class ValueProxy(Proxy):
    """A Proxy that also carries the concrete value it stands for.

    The tracer can evaluate some calls for real - most importantly draws from
    a real np.random.Generator, whose only dynamic input is the generator
    itself. When it does, it returns this proxy so that the call is still
    recorded as a node in the graph, while the traced program can also use
    the value: to index a list, to slice, to convert to float, or to branch.

    Keeping the value available matters for two reasons:

    1. Fidelity. pf.control.choice always consumes the real generator, so if
       the tracer skipped the draws in between, the choice the tracer records
       would not be the choice the renderer takes.
    2. Expressiveness. Scene generators that branch on a sampled value cannot
       be traced at all if the tracer refuses to look at that value.

    Arithmetic on a ValueProxy stays value-aware: if every operand has a
    known value, the result carries the computed value too.
    """

    value: Any = None

    def __index__(self) -> int:
        return int(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __float__(self) -> float:
        return float(self.value)

    def __bool__(self) -> bool:
        return bool(self.value)

    def __len__(self) -> int:
        return len(self.value)

    def __iter__(self):
        return iter(self.value)


@dataclass
class AttributeProxy(Proxy):
    """Special proxy for attribute access that supports peekthrough optimization"""

    source_value: Any = _NOT_EVALUATED

    def __init__(self, node: Node, source_value: Any = _NOT_EVALUATED):
        super().__init__(node)
        assert isinstance(node, GetAttributeNode), node
        self.source_value = source_value

    def __getattr__(self, attr: str) -> "AttributeProxy":
        node = GetAttributeNode(source=self.node, attribute_name=attr)
        source_value = self.__dict__.get("source_value", _NOT_EVALUATED)
        if source_value is not _NOT_EVALUATED:
            return _known_getattr(node, source_value, attr)
        return AttributeProxy(node)

    def __call__(self, *args, **kwargs) -> Proxy:
        """
        Someone did func = proxy.xyz, then func(), or equivelantly thats just proxy.xyz().
        We can convert that to just a single node which is a method call on the obj node.

        torch.fx.symbolic_trace calls this a _peekthrough optimization_

        TODO: this means that `self` is very often dropped from the graph. need to account for this if we check for drops.
        """
        assert isinstance(self.node, GetAttributeNode), self.node

        # Convert any Proxy args to their underlying nodes
        node_args = tuple(arg.node if isinstance(arg, Proxy) else arg for arg in args)
        node_kwargs = {
            k: v.node if isinstance(v, Proxy) else v for k, v in kwargs.items()
        }
        # self.node.args[0] is the source node that we're calling the method on
        call_node = MethodCallNode(
            callee=self.node.args[0],
            method_name=self.node.attribute_name,
            args=node_args,
            kwargs=node_kwargs,
        )
        if self.source_value is not _NOT_EVALUATED:
            try:
                method = getattr(self.source_value, self.node.attribute_name)
                value = method(
                    *(a.value if isinstance(a, ValueProxy) else a for a in args),
                    **{k: v.value if isinstance(v, ValueProxy) else v for k, v in kwargs.items()},
                )
            except Exception:
                value = _NOT_EVALUATED
            if value is not _NOT_EVALUATED:
                return _maybe_value_proxy(call_node, value)
        return Proxy(call_node)
