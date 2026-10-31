"""Round-3 engine instrumentation: memoise realised subtree results by node hash.

The stage cache (DIRTY.md) proved an edit's dirty closure at *graph* level but
could not collect the reusable work: the venv boundary re-traces a stage as a
unit. This module is the missing engine-side piece — a memo over the *realised*
results of the functions a scene is composed from, keyed by a content hash of
each call (the function's identity plus value-hashes of its arguments,
including the bit-generator state of every rng argument). Because v2 draws
every grammar decision from ``spawn``ed child generators, a builder called
with the same spawned rng and the same argument values produces the same
bytes; the memo turns that into reuse.

Two modes, one store, one process:

* ``populate`` — every wrapped call executes and is recorded: its result (a
  live Python object graph, bpy datablocks included — this memo is in-process
  by design, the resident-engine model of DIRTY.md §6), the post-call state of
  every rng argument, and (for functions flagged ``track_created``) the bpy
  objects the call created.
* ``splice`` — a call whose hash is present is *spliced*: the recorded rng
  post-states are restored (so the caller's stream position after the call is
  exactly the base run's, even when the caller shares the generator) and the
  recorded result is returned without executing. A call whose hash is absent
  executes; if a base entry for that call existed it is *superseded* — its
  bpy objects are deleted before re-execution so the rebuilt subtree re-takes
  the same names in the same order.

Calls with the same hash consume queue entries in call order, so a builder
executed twice in the base run yields two distinct objects in the splice run,
never one object aliased into two placements.

Hashing rules (``value_hash``): primitives by value; numpy arrays by
shape/dtype/bytes and numpy scalars by their bytes; ``np.random.Generator`` by
its bit-generator state; functions/classes by module+qualname; tuples, lists,
dicts and sets recursively; other Python objects by their ``__dict__``/slots
recursively; C extension types without Python fields (mathutils vectors,
euler) via pickle; live ``bpy`` datablocks by pointer+name+type (content-
hashing a datablock would follow the entire modifier graph every call;
in-process pointer identity plus the end-to-end byte-equality check is the
contract here). Anything that cannot be hashed skips the memo entirely and
executes — an unhashable argument can only cost time, never correctness.

Mutating helpers must not be wrapped: a wrapped function's whole effect has to
be its return value (plus rng consumption, which is recorded). The default
target set (``infinigen2.util.splice``) wraps builders only.
"""

from __future__ import annotations

import copy as _copy
import functools
import hashlib
import json
import logging
import pickle
import threading
import time
import os

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

_MAX_WALK = 20_000  # nodes visited while content-hashing one call


class Unhashable(Exception):
    """Raised internally when a value cannot enter a memo key."""


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tok(payload: str) -> str:
    """A deterministic token for one value. Leaves are plain strings (no
    hashing per leaf — a scene's arguments walk to millions of leaves and
    per-leaf sha256 dominated everything); call_hash folds the whole token
    tree into one digest."""
    return payload


def _walk_bpy():
    try:
        import bpy  # noqa: PLC0415
    except ImportError:
        return None
    return bpy


# ---------------------------------------------------------------------------
# content hashing


def value_hash(value: Any, budget: list[int] | None = None) -> str:
    """A content hash of a concrete value; Unhashable if it cannot have one."""
    if budget is None:
        budget = [_MAX_WALK]
    budget[0] -= 1
    if budget[0] < 0:
        # past the walk budget: fall back to in-process identity rather than
        # giving up on the call — args are alive in cached entries, so an id
        # cannot be reused while it can still be served
        return f"budget-id:{type(value).__name__}:{id(value)}"
    v = value
    if v is None or isinstance(v, (bool, int, float, complex, str, bytes)):
        return _tok(f"{type(v).__name__}:{v!r}")
    if isinstance(v, np.ndarray):
        if v.dtype.hasobject:
            raise Unhashable("object-dtype array")
        try:
            arr = np.ascontiguousarray(v)
            head = f"ndarray:{arr.dtype.str}:{arr.shape}:".encode()
            return _tok(f"{head}{hashlib.sha256(arr.tobytes()).hexdigest()}")
        except Exception as err:
            raise Unhashable(f"array: {err}") from err
    if isinstance(v, np.generic):  # np.float64(3.0), np.int32(7), ...
        try:
            return _tok(f"np:{v.dtype.str}:{hashlib.sha256(v.tobytes()).hexdigest()}")
        except Exception as err:
            raise Unhashable(f"numpy scalar: {err}") from err
    if isinstance(v, np.random.Generator):
        try:
            state = json.dumps(v.bit_generator.state, sort_keys=True)
        except Exception as err:
            raise Unhashable(f"rng state: {err}") from err
        return _tok(f"rng:{state}")
    if isinstance(v, (tuple, list)):
        parts = [value_hash(x, budget) for x in v]
        return _tok(f"{type(v).__name__}:" + ",".join(parts))
    if isinstance(v, dict):
        try:
            items = sorted((repr(k), value_hash(val, budget)) for k, val in v.items())
        except TypeError as err:
            raise Unhashable(f"dict keys: {err}") from err
        joined = ";".join(f"{k}={h}" for k, h in items)
        return _tok(f"dict:{joined}")
    if isinstance(v, (set, frozenset)):
        parts = sorted(value_hash(x, budget) for x in v)
        return _tok(f"{type(v).__name__}:" + ",".join(parts))
    bpy = _walk_bpy()
    if bpy is not None and isinstance(v, bpy.types.ID):
        # live datablock: stable pointer for the process, plus a cheap
        # fingerprint (name/type) that catches pointer reuse after deletion
        return _tok(f"bpy:{v.as_pointer()}:{type(v).__name__}:{v.name}")
    if bpy is not None and not isinstance(v, (tuple, list, dict, set, frozenset)):
        tmod = type(v).__module__
        item = getattr(v, "_item", None)
        if isinstance(item, bpy.types.ID):
            # a procfunc object wrapper (MeshObject, ColliderObject, ...) is
            # its realized datablock: hashing by pointer collapses the whole
            # node-graph walk, which is what blew the budget on the scene's
            # big list arguments (colliders, objects) before
            return _tok(
                f"pfwrap:{tmod}.{type(v).__qualname__}:"
                f"{item.as_pointer()}:{item.name}"
            )
        if tmod == "procfunc.types" or tmod.startswith("procfunc.nodes.types"):
            # the pf Object protocol without a realized _item yet, or another
            # engine-internal handle: identity within the process
            return _tok(f"pfid:{tmod}.{type(v).__qualname__}:{id(v)}")
        if tmod.startswith("trimesh"):
            # cached collision geometry derived from a bpy object's mesh —
            # identity carries the derivation
            return _tok(f"trimesh:{tmod}.{type(v).__qualname__}:{id(v)}")
        if hasattr(v, "objs") and hasattr(v, "collision_manager"):
            # infinigen2's CollisionSet: the collider *objects* decide; the
            # manager, the per-object trimesh/fcl caches and the transform
            # cache are all derived from them
            try:
                elems = tuple(value_hash(o, budget) for o in v.objs)
            except Unhashable:
                raise
            names = ""
            if hasattr(v, "object_names"):
                names = ",".join(sorted(repr(k) for k in v.object_names))
            return _tok(
                f"collset:{tmod}.{type(v).__qualname__}"
                f":{';'.join(elems)}:{names}"
            )
    if isinstance(v, (type, functools.partial)) or (
        callable(v) and not isinstance(v, np.ndarray)
    ):
        mod = getattr(v, "__module__", None)
        qual = getattr(v, "__qualname__", None)
        if mod and qual and qual not in ("<lambda>",):
            return _tok(f"callable:{mod}.{qual}")
        self_ref = getattr(v, "__self__", None)
        if self_ref is not None:
            try:
                return _tok(
                    f"method:{type(self_ref).__qualname__}.{getattr(v, '__name__', '?')}"
                    f":{value_hash(self_ref, budget)}"
                )
            except Unhashable:
                pass
        # lambdas / partials / local closures: identity within this process
        return _tok(f"callable-id:{id(v)}")
    # generic Python object: hash its type and fields (dataclasses, pf
    # results holding bpy items via the ID branch above)
    try:
        if hasattr(v, "__dict__") and vars(v):
            fields = sorted(vars(v).items(), key=lambda kv: repr(kv[0]))
        else:
            fields = [
                (attr, getattr(v, attr))
                for attr in getattr(type(v), "__slots__", ()) or ()
            ]
    except Exception as err:
        raise Unhashable(f"fields: {err}") from err
    if not fields and not getattr(type(v), "__slots__", ()):
        # C extension type without Python fields (mathutils.Vector, ...):
        # stable via pickle, else via iteration, else via pointer
        try:
            return _tok(f"pickle:{hashlib.sha256(pickle.dumps(v, protocol=5)).hexdigest()}")
        except Exception:
            pass
        try:
            return value_hash(tuple(v), budget)
        except (Unhashable, TypeError):
            pass
        ptr = getattr(v, "as_pointer", None)
        if callable(ptr):
            return _tok(f"ptr:{ptr()}:{type(v).__name__}")
        return _tok(f"opaque:{type(v).__name__}:{id(v)}")
    parts = [f"{type(v).__module__}.{type(v).__qualname__}"]
    for name, field_val in fields:
        try:
            fh = value_hash(field_val, budget)
        except Unhashable as why:
            if str(why).startswith("unstructured"):
                # an opaque C object in a field position — e.g. python-fcl's
                # DynamicAABBTreeCollisionManager inside a ColliderSet, which
                # is an index *derived from* the objects beside it. Elide it:
                # hashing its identity would defeat content-addressing (the
                # index is rebuilt per run), hashing its contents is not
                # possible from Python.
                fh = f"#opaque:{type(field_val).__name__}"
            else:
                raise
        parts.append(f"{name}={fh}")
    return _tok("obj:" + ";".join(parts))


def call_hash(qualname: str, args: tuple, kwargs: dict) -> str:
    """The content hash of one call: the function plus its argument values."""
    h = value_hash((args, kwargs))
    return _hash_bytes(f"{qualname}\x00{h}".encode())


# ---------------------------------------------------------------------------


def _rng_leaves(value: Any, out: list) -> None:
    if isinstance(value, np.random.Generator):
        out.append(value)
    elif isinstance(value, (tuple, list)):
        for x in value:
            _rng_leaves(x, out)
    elif isinstance(value, dict):
        for x in value.values():
            _rng_leaves(x, out)


@dataclass
class Entry:
    key: str
    qualname: str
    result: Any
    rng_post: list[dict]
    created: list = field(default_factory=list)  # (pointer, name) bpy objects
    exec_seconds: float = 0.0
    children_seconds: float = 0.0  # direct nested memoised calls
    consumed: bool = False  # served to a splice call
    superseded: bool = False
    servable: bool = True  # False when its result references dead datablocks

    @property
    def self_seconds(self) -> float:
        return max(self.exec_seconds - self.children_seconds, 0.0)


def _result_ptrs(value, out: set, seen: set, budget: list) -> None:
    """Every bpy datablock pointer reachable from `value`, bounded."""
    if budget[0] <= 0:
        return
    budget[0] -= 1
    if id(value) in seen:
        return
    try:
        ptr = getattr(value, "as_pointer", None)
    except Exception:
        out.add(("dead", id(value)))
        return
    if callable(ptr):
        try:
            out.add(ptr())
        except Exception:
            out.add(("dead", id(value)))
        return
    seen.add(id(value))
    if isinstance(value, (tuple, list, set, frozenset)):
        for x in value:
            _result_ptrs(x, out, seen, budget)
    elif isinstance(value, dict):
        for x in value.values():
            _result_ptrs(x, out, seen, budget)
    else:
        try:
            fields = list(vars(value).items()) if hasattr(value, "__dict__") else []
        except Exception:
            fields = []
        for _k, v in fields:
            _result_ptrs(v, out, seen, budget)


def _entry_stats() -> dict:
    return {
        "calls": 0,
        "hits": 0,
        "misses": 0,
        "skipped": 0,
        "exec_seconds": 0.0,
        "hash_seconds": 0.0,
        "hit_seconds": 0.0,
        "created_objects": 0,
        "superseded_objects": 0,
    }


@dataclass
class _Frame:
    qualname: str
    snapshot: set[int] | None
    created_buffer: list = field(default_factory=list)
    children_seconds: float = 0.0


class SubtreeMemo:
    """The in-process memo. Inert until a :meth:`session` is entered."""

    def __init__(self) -> None:
        self.mode: str | None = None
        self.store: dict[str, list[Entry]] = {}
        self.by_func: dict[str, list[Entry]] = {}
        self.stats: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._local = threading.local()
        self._claimed: set[int] = set()  # pointers already attributed
        self.collected_self_seconds = 0.0
        self.executed_self_seconds = 0.0
        self._protected_cache = None

    # -- session ------------------------------------------------------------

    @contextmanager
    def session(self, mode: str):
        assert mode in ("populate", "splice"), mode
        assert self.mode is None, "a session is already active"
        _install_creation_hooks()
        # deep linked structures (mutated-node chains) walk far past the
        # default recursion limit once opaque fields are elided instead of
        # aborting the hash
        import sys as _sys

        if _sys.getrecursionlimit() < 100_000:
            _sys.setrecursionlimit(100_000)
        self.mode = mode
        self._claimed = set()
        self.collected_self_seconds = 0.0
        self.executed_self_seconds = 0.0
        self._protected_cache = None
        try:
            yield self
        finally:
            self.mode = None

    # -- the wrapper ----------------------------------------------------------

    def call(self, qualname: str, func: Callable, args, kwargs):
        """One wrapped call. Splices on hit, executes (recording) on miss."""
        assert self.mode is not None
        stats = self.stats.setdefault(qualname, _entry_stats())
        stats["calls"] += 1

        t_hash = time.perf_counter()
        try:
            key = call_hash(qualname, args, kwargs)
        except RecursionError:
            stats["skipped"] += 1
            return func(*args, **kwargs)
        except Unhashable as why:
            stats["skipped"] += 1
            reason = str(why)
            stats.setdefault("skip_reasons", {})
            stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
            return func(*args, **kwargs)
        rngs: list[np.random.Generator] = []
        _rng_leaves(args, rngs)
        _rng_leaves(kwargs, rngs)
        stats["hash_seconds"] += time.perf_counter() - t_hash

        with self._lock:
            entries = self.store.get(key)
            if self.mode == "splice" and entries:
                hit_entry = entries.pop(0)
                hit_entry.consumed = True
                self._serve(qualname, stats, hit_entry, rngs)
                return hit_entry.result

        if self.mode == "splice":
            # A tracked function re-executing in the splice run supersedes
            # every base entry of the SAME function not yet consumed: within
            # the unchanged call prefix, an entry that would be hit has been
            # hit already (call order is preserved), so what remains is stale
            # — the base run's objects for call sites whose inputs changed.
            # Deleting them before executing lets the rebuild retake the same
            # names in the same order. Function-granularity splice-safety: a
            # function whose call sequence is not prefix-stable (an unchanged
            # call after a changed one) over-deletes and re-executes — correct
            # (execution is deterministic) but less collected.
            pending = [
                e
                for e in self.by_func.get(qualname, ())
                if not e.consumed and not e.superseded
            ]
            # unservable entries (validation dropped their result) still own
            # live objects: superseding them is what removes the base run's
            # stale instances from the resident scene
            if pending:
                doomed = [o for entry in pending for o in entry.created]
                protected = self._protected_ptrs(exclude=pending)
                doomed = [p for p in doomed if p not in protected]
                if os.environ.get("SPLICE_DEBUG_SUPERSEDE"):
                    import bpy as _bpy

                    alive = {o.as_pointer(): o.name for o in _bpy.data.objects}
                    blockers = {}
                    for _pes in self.store.values():
                        for _e in _pes:
                            if _e in pending:
                                continue
                            refs = set()
                            _result_ptrs(_e.result, refs, set(), [50_000])
                            for ptr in doomed:
                                if ptr in refs and ptr in alive:
                                    blockers.setdefault(
                                        _e.qualname.split(".")[-1], []
                                    ).append(alive[ptr])
                    print(
                        f"[supersede] {qualname.split('.')[-1]}: pending={len(pending)} "
                        f"created={sum(len(e.created) for e in pending)} doomed={len(doomed)} "
                        f"protected={len(protected)} blockers={ {k: len(v) for k, v in blockers.items()} }",
                        flush=True,
                    )
                    for k, v in list(blockers.items())[:3]:
                        print(f"    {k} protects e.g. {v[:4]}", flush=True)
                self._delete_objects(qualname, stats, doomed)
                for entry in pending:
                    entry.superseded = True
                    bucket = self.store.get(entry.key)
                    if bucket and entry in bucket:
                        bucket.remove(entry)

        snapshot = self._snapshot()
        frame = _Frame(qualname, snapshot)
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        stack.append(frame)
        t0 = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            stack.pop()
            stats["exec_seconds"] += elapsed
            if stack:
                stack[-1].children_seconds += elapsed
        created = self._created_since(frame)
        rng_post = [_copy.deepcopy(r.bit_generator.state) for r in rngs]
        entry = Entry(
            key=key,
            qualname=qualname,
            result=result,
            rng_post=rng_post,
            created=created,
            exec_seconds=elapsed,
            children_seconds=frame.children_seconds,
        )
        with self._lock:
            self.store.setdefault(key, []).append(entry)
            self.by_func.setdefault(qualname, []).append(entry)
            if self._protected_cache is not None:
                self._protected_cache.update(entry.created)
                _result_ptrs(
                    entry.result, self._protected_cache, set(), [20_000]
                )
        stats["misses"] += 1
        stats["created_objects"] += len(created)
        self.executed_self_seconds += entry.self_seconds
        return result

    def _serve(self, qualname: str, stats: dict, entry: Entry, rngs) -> None:
        t0 = time.perf_counter()
        # the base run consumed the rngs inside this call; restore the exact
        # post-call stream positions so shared generators stay aligned
        for rng, post in zip(rngs, entry.rng_post):
            if post is not None:
                try:
                    rng.bit_generator.state = _copy.deepcopy(post)
                except Exception:
                    pass
        stats["hits"] += 1
        stats["hit_seconds"] += time.perf_counter() - t0
        self.collected_self_seconds += entry.self_seconds
        logger.debug(f"subtree-memo hit: {qualname}")

    # -- bpy bookkeeping ------------------------------------------------------

    def _snapshot(self) -> set[int] | None:
        bpy = _walk_bpy()
        if bpy is None:
            return None
        return {o.as_pointer() for o in bpy.data.objects}

    def _created_since(self, frame: _Frame) -> list:
        """bpy objects created while `frame` was active and not claimed by an
        inner (nested) memoised call — attributed to `frame`'s key."""
        bpy = _walk_bpy()
        if bpy is None or frame.snapshot is None:
            return []
        # merge creation-hook pointers (exact, ints) with the exit diff
        created: list[int] = list(dict.fromkeys(frame.created_buffer))
        seen = set(created)
        for obj in bpy.data.objects:
            ptr = obj.as_pointer()
            if ptr in frame.snapshot or ptr in self._claimed or ptr in seen:
                continue
            created.append(ptr)
            self._claimed.add(ptr)
            seen.add(ptr)
        return created

    def _protected_ptrs(self, exclude: list[Entry]) -> set:
        """Pointers no supersede may delete: everything the still-live entries
        own or reference (a hit may already have served them) — lazily
        realized geometry attributes objects to the frame that *realized*
        them, which is not always the frame that owns them. Computed over the
        store once, then extended as splice-run entries are created."""
        if self._protected_cache is None:
            protected: set = set()
            budget = [400_000]
            for entries in self.store.values():
                for e in entries:
                    protected.update(e.created)
                    _result_ptrs(e.result, protected, set(), budget)
            self._protected_cache = protected
        return self._protected_cache

    def _delete_objects(self, qualname: str, stats: dict, doomed: list) -> None:
        if not doomed:
            return
        bpy = _walk_bpy()
        if bpy is None:
            return
        alive = {o.as_pointer(): o for o in bpy.data.objects}
        objs = [alive[ptr] for ptr in doomed if ptr in alive]
        if not objs:
            return
        stats["superseded_objects"] += len(objs)
        logger.debug(
            f"subtree-memo splice: superseding {len(objs)} objects of {qualname}"
        )
        try:
            bpy.data.batch_remove(objs)
            # no orphan-data sweep here, deliberately: aliases share mesh data,
            # and a mesh whose every alias is superseded may still be the
            # cached result of a *hit* entry (a pool the splice run keeps
            # serving). Orphaned data costs memory, never correctness.
        except Exception as err:
            logger.warning(f"subtree-memo: failed to supersede objects: {err}")


MEMO = SubtreeMemo()


def wrap(qualname: str | None = None):
    """Decorator: wrap a builder for subtree memoisation.

    Inert outside a session — the wrapped function is called through with no
    hashing and no recording, so un-instrumented runs are unchanged (the
    round-2 convention for every instrumentation patch).
    """

    def deco(func: Callable) -> Callable:
        name = qualname or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if MEMO.mode is None:
                return func(*args, **kwargs)
            return MEMO.call(name, func, args, kwargs)

        wrapper._subtree_memo_orig = func  # type: ignore[attr-defined]
        return wrapper

    return deco


def report() -> dict:
    """The session's counters, per function and in total."""
    out = {
        "functions": {
            name: {k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()}
            for name, st in MEMO.stats.items()
        },
    }
    tot = _entry_stats()
    for st in MEMO.stats.values():
        for k, v in st.items():
            if isinstance(v, (int, float)):
                tot[k] = tot.get(k, 0) + v
    out["total"] = {
        k: round(v, 4) if isinstance(v, float) else v for k, v in tot.items()
    }
    out["store_entries"] = sum(len(v) for v in MEMO.store.values())
    out["collected_self_seconds"] = round(MEMO.collected_self_seconds, 4)
    out["executed_self_seconds"] = round(MEMO.executed_self_seconds, 4)
    out["store_self_seconds"] = round(
        sum(e.self_seconds for lst in MEMO.store.values() for e in lst), 4
    )
    return out


# ---------------------------------------------------------------------------
# creation hooks — precise attribution of bpy objects to memoised frames
#
# Object creation in this stack does not happen only inside the builder that
# logically owns the object: procfunc's node graphs realize lazily, and the
# scatter realizes instances through the depsgraph (to_aliases). Frame
# entry/exit snapshots misattribute all of that. But every creation path
# RETURNS the objects it creates (to_mesh_object, to_aliases, the primitives,
# alias/duplicate/join), so attribution by return value is exact and cheap:
# each hook appends the returned objects' pointers to the innermost active
# memo frame. Creation no creation-hook covers still lands in the enclosing
# frame via the exit snapshot diff.

_CREATION_SITES = [
    ("procfunc.nodes.execute.execute", "to_mesh_object"),
    ("procfunc.nodes.execute.execute", "to_mesh_object_with_attributes"),
    ("procfunc.nodes.execute.execute", "to_objects_multi"),
    ("procfunc.nodes.execute.execute", "to_aliases"),
    ("procfunc.nodes.execute.execute", "to_curve_object"),
    ("procfunc.ops.object", "alias"),
    ("procfunc.ops.object", "duplicate"),
    ("procfunc.ops.object", "joined"),
    ("procfunc.ops.object", "join"),
    ("procfunc.ops.primitives.mesh", "mesh_single_vertex"),
    ("procfunc.ops.primitives.mesh", "point_cloud"),
    ("procfunc.ops.primitives.mesh", "mesh_from_numpy"),
    ("procfunc.ops.primitives.mesh", "mesh_line"),
    ("procfunc.ops.primitives.mesh", "empty"),
    ("procfunc.ops.primitives.curve", "curve_line"),
    ("procfunc.ops.primitives.curve", "curve_splines"),
]

_creation_hooks_installed = False


def _collect_return_ptrs(value, out: list, depth: int = 0) -> None:
    if depth > 6:
        return
    ptr = getattr(value, "as_pointer", None)
    if callable(ptr):
        try:
            out.append(ptr())
        except Exception:
            pass
        return
    if isinstance(value, (tuple, list)):
        for x in value:
            _collect_return_ptrs(x, out, depth + 1)
    elif isinstance(value, dict):
        for x in value.values():
            _collect_return_ptrs(x, out, depth + 1)


def _install_creation_hooks() -> None:
    global _creation_hooks_installed
    if _creation_hooks_installed:
        return
    import importlib

    for mod_name, fn_name in _CREATION_SITES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as err:
            logger.debug(f"creation hook: cannot import {mod_name}: {err}")
            continue
        orig = getattr(mod, fn_name, None)
        if orig is None or getattr(orig, "_memo_creation_hook", False):
            continue

        def make_hook(original):
            def hook(*args, **kwargs):
                result = original(*args, **kwargs)
                stack = getattr(MEMO._local, "stack", None)
                if MEMO.mode is not None and stack:
                    ptrs: list = []
                    _collect_return_ptrs(result, ptrs)
                    stack[-1].created_buffer.extend(ptrs)
                return result

            hook._memo_creation_hook = True  # type: ignore[attr-defined]
            return functools.wraps(original)(hook)

        hook = make_hook(orig)
        setattr(mod, fn_name, hook)
        import sys as _sys

        for other in list(_sys.modules.values()):
            ns = getattr(other, "__dict__", None)
            if ns and ns.get(fn_name) is orig:
                ns[fn_name] = hook
        _creation_hooks_installed = True
        logger.debug(f"creation hook installed: {mod_name}.{fn_name}")
