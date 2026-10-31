#!/usr/bin/env python3
"""
closed_loop.py - MEASURE the loop  R = A o G  on real Infinigen ground truth.

    R(y) = A(G(y))          the round trip
    drift = ||R(y) - y||    the residual

Two residuals, because the toy proved one is not enough:
  CONSISTENCY  ||A(G(y)) - y||        - does the generated world still say what
                                        the perception stack saw?
  PLAUSIBILITY  score residual        - is the generated world a world at all?
  A world can be perfectly consistent and physically impossible. The toy
  (toy_coin.py) exhibits exactly that: the deterministic inverse has a
  round-trip error of ~0 and lands 10x farther from the data manifold than a
  sampled inverse. So a critic that only checks consistency is not a critic.

USAGE
    # self-test: no model, no data, runs anywhere, proves the harness itself
    python3 closed_loop.py --selftest

    # real measurement on the Thor over an Infinigen render directory
    python3 closed_loop.py --source /path/to/render_out \
        --perception mypkg.perception:build --generator mypkg.generator:build \
        --out runs/closed-loop-01

    # controls only (always run, always reported)
    python3 closed_loop.py --source ... --controls-only

PRE-REGISTERED PREDICTIONS (state them BEFORE measuring; the receipt says
whether they held)
  E1  Under autoregressive conditioning (G sees its own previous output), the
      per-frame residual GROWS with frame index. Growth is superlinear if the
      generator conditions on its own output, linear if it conditions only on
      the measurement.
  E2  A generator trained on the CLOSED-LOOP objective
      (consistency + plausibility) has lower plausibility residual than one
      trained on pixel error against a teacher, at equal compute.
  E3  Consistency residual and plausibility residual are NOT the same signal:
      there exist frames with low consistency error and high implausibility.
      A consistency-only gate passes them.
  E4  The magnitude of drift on a sequence is predicted by the information the
      perception stack destroyed (I(x;y)), measured on the same distribution.

CONTROLS (a harness that cannot fail is not a measurement)
  POSITIVE  G = identity. The residual MUST be exactly 0. If it is not, the
            harness is broken and every number it produces is void.
  NEGATIVE  G = identity + eps. The residual MUST grow. If it does not, the
            metric cannot see a bad generator and is void.
"""
import argparse, csv, glob, json, os, re, sys, time
import numpy as np

# ---------------------------------------------------------------------------
# modalities and their metrics
# ---------------------------------------------------------------------------
MODALITIES = ("rgb", "depth", "flow", "obj_index", "semantic_seg")

# The REAL Infinigen output layout, from
# src/infinigen2/exporters/util/format.py:
#
#     ExportType.IMAGE          Path("%c/%f.png")              uint8
#     ExportType.DEPTH          Path("%c/depth_%f.npy")        float32
#     ExportType.OPTICAL_FLOW   Path("%c/optical-flow_%f.npy") float32
#     ExportType.OBJECT_INDEX   Path("%c/object_%f.npy")       uint32
#     ExportType.SURFACE_NORMAL Path("%c/surface-normal_%f.npy") float32
#
# with  %c -> the camera name (a SUBDIRECTORY)  and  %f -> f"{frame:04d}".
# So a render lands as:
#
#     <output>/<camera>/0001.png
#     <output>/<camera>/depth_0001.npy
#     <output>/<camera>/optical-flow_0001.npy
#     <output>/<camera>/object_0001.npy
#
# A loader that globs the top level for "rgb_*.png" and "*obj_index*.npy"
# finds NOTHING on a real output and reports "no frames found" - which reads
# like a data problem and is actually a loader problem.

#: filename stem -> modality key
_PASS_STEMS = {
    "depth": "depth",
    "optical-flow": "flow",
    "flow": "flow",
    "object": "obj_index",
    "obj-index": "obj_index",
    "surface-normal": "normal",
    "surface-normal-world": "normal_world",
    "material-index": "material_index",
}
_IMAGE_STEMS = {"image-denoised": "rgb_denoised"}


def _frame_number(stem):
    """The frame number out of a filename STEM ("depth_0001" -> 1).

    Infinigen writes %f as f"{frame:04d}", so the number is the trailing
    digits of the stem.  It may be followed by a viewlayer suffix on some
    passes, so take the LAST digit group."""
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else None


def load_frames(source, limit=None):
    """Read an Infinigen render directory's passes, indexed by (camera, frame).

    Walks the tree, because the passes live one directory down, under the
    camera name."""
    index = {}
    for root, _dirs, files in os.walk(source):
        camera = os.path.basename(os.path.normpath(root))
        for fn in files:
            stem, ext = os.path.splitext(fn)
            ext = ext.lower()
            if ext not in (".npy", ".png", ".exr"):
                continue
            frame = _frame_number(stem)
            if frame is None:
                continue
            key = None
            if ext == ".npy":
                # "depth_0001" -> "depth";  "object_0001" -> "object"
                base = re.sub(r"_\d+$", "", stem)
                key = _PASS_STEMS.get(base)
            else:
                if stem.isdigit():
                    key = "rgb"                 # "%c/%f.png" is the main image
                else:
                    base = re.sub(r"_\d+$", "", stem)
                    key = _IMAGE_STEMS.get(base)
            if key is None:
                continue
            index.setdefault((camera, frame), {})[key] = os.path.join(root, fn)

    if not index:
        return []
    frames = []
    for k in sorted(index):
        if limit and len(frames) >= limit:
            break
        d = {}
        for key, path in index[k].items():
            arr = read_any(path)
            if arr is not None:
                d[key] = arr
        if d:
            frames.append(d)
    return frames


def read_any(path):
    """Load a pass WITHOUT flattening its dtype.  object index is uint32 and
    rgb is uint8; casting them to float32 here loses the type information a
    caller may need, and the metrics cast for themselves."""
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".exr"):
        try:
            import imageio.v3 as iio
            return np.asarray(iio.imread(path), dtype=np.float32)
        except Exception:
            return None
    if path.endswith(".png"):
        try:
            from PIL import Image
            return np.asarray(Image.open(path)).astype(np.float32)
        except Exception:
            return None
    return None

def metric(name, a, b):
    """The ERROR between generated `a` and target `b` for one modality.

    EVERY metric here is an error, so LOWER IS BETTER, without exception.
    (A PSNR would be the opposite direction, and a residual dict that mixes
    the two cannot carry a control like "identity must give exactly 0".)"""
    if a is None or b is None: return None
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        n = min(a.size, b.size)
        a = a.reshape(-1)[:n]; b = b.reshape(-1)[:n]
    if name in ("depth",):
        m = b > 1e-6
        if m.sum() == 0: return None
        return float(np.abs(a[m] - b[m]).mean() / max(np.abs(b[m]).mean(), 1e-6))
    if name in ("flow",):
        return float(np.linalg.norm(a - b, axis=-1).mean())
    if name in ("obj_index", "semantic_seg", "material_index"):
        # Infinigen writes these as LABEL MAPS (H,W) uint32, not one-hot.
        # argmax(-1) on a label map is meaningless, and on a 1-channel array
        # it silently compares channel 0 against itself.
        if a.ndim == 3 and a.shape[-1] > 1:
            a = a.argmax(-1)
            b = b.argmax(-1)
        return float((a != b).mean())
    if name in ("rgb", "rgb_denoised"):
        # RMSE in 0-255 units.  An error, like every other metric here.
        return float(np.sqrt(((a - b) ** 2).mean()))
    return float(np.abs(a - b).mean())

# ---------------------------------------------------------------------------
# stub backends: the controls.  These must work with NO model at all.
# ---------------------------------------------------------------------------
class IdentityPerception:
    name = "stub:identity-perception"
    def perceive(self, frames): return [dict(f) for f in frames]

class IdentityGenerator:
    name = "stub:identity-generator"
    eps = 0.0
    def generate(self, obs, history=None):
        return [dict(o) for o in obs]

class NoisyGenerator(IdentityGenerator):
    """A generator that corrupts EVERYTHING, including the integer passes.

    (A bad generator corrupts the object index too.  Perturbing only the
    floating arrays would leave the label maps bit-exact, and the control
    would then report "the noise did not change anything" for a generator
    that is plainly broken.)"""
    name = "stub:noisy-generator"
    eps = 0.05
    def generate(self, obs, history=None):
        out = []
        for o in obs:
            d = dict(o)
            for k in list(d):
                if d[k] is not None and np.issubdtype(
                        np.asarray(d[k]).dtype, np.number):
                    d[k] = (np.asarray(d[k], dtype=np.float32)
                            + self.eps * np.random.randn(*np.shape(d[k])))
            out.append(d)
        return out

def load_factory(spec):
    """'pkg.mod:factory' -> callable.  'stub' -> None."""
    if spec in (None, "", "stub"): return None
    mod, _, fn = spec.partition(":")
    __import__(mod)
    return getattr(sys.modules[mod], fn)

# ---------------------------------------------------------------------------
def run_roundtrip(frames, perception, generator, plausibility=None, autoregressive=False):
    """one pass: observe -> generate -> re-observe -> residual."""
    obs = perception.perceive(frames)
    hist = None
    gen = []
    for o in obs:
        g = generator.generate([o], history=hist)
        g = g[0] if isinstance(g, list) else g
        gen.append(g)
        hist = g if autoregressive else None
    re_obs = perception.perceive(gen)
    per_frame = []
    for i, (o, r) in enumerate(zip(obs, re_obs)):
        rec = {"frame": i}
        for k in MODALITIES:
            if k in o and k in r:
                m = metric(k, r[k], o[k])
                if m is not None: rec[k] = round(m, 6)
        if plausibility is not None:
            try:
                rec["plausibility"] = round(float(plausibility([g])[0]), 6)
            except Exception as e:
                rec["plausibility"] = None
                rec["plausibility_error"] = str(e)[:200]
        per_frame.append(rec)
    return per_frame

def summarize(per_frame):
    keys = [k for k in (per_frame[0] if per_frame else {}) if k != "frame"]
    agg = {k: float(np.mean([f[k] for f in per_frame if k in f and f[k] is not None]))
           for k in keys}
    # drift growth: slope of log(residual) vs frame index, per metric
    growth = {}
    n = len(per_frame)
    if n > 2:
        idx = np.arange(n)
        for k in keys:
            vals = np.array([f.get(k) for f in per_frame], dtype=float)
            if np.any(np.isnan(vals)): continue
            m = np.isfinite(vals) & (np.abs(vals) > 0)
            if m.sum() > 2:
                growth[k] = float(np.polyfit(idx[m], np.log(np.abs(vals[m])), 1)[0])
    return agg, growth

# ---------------------------------------------------------------------------
def selftest():
    """the harness proves itself.  a measurement tool that cannot fail is void."""
    print("=" * 78); print("closed_loop.py --selftest : the harness proves itself"); print("=" * 78)
    rng = np.random.default_rng(0)
    frames = [{"depth": rng.random((8, 8)).astype(np.float32),
               "semantic_seg": rng.integers(0, 4, (8, 8, 5)).astype(np.float32),
               "flow": rng.normal(0, 1, (8, 8, 2)).astype(np.float32)} for _ in range(6)]
    P = IdentityPerception()

    per = run_roundtrip(frames, P, IdentityGenerator())
    agg, _ = summarize(per)
    pos_ok = all(abs(v) < 1e-12 for v in agg.values())
    print(f"\nPOSITIVE control (G = identity): residual {agg}")
    print(f"  -> must be exactly 0 .......... {'PASS' if pos_ok else 'FAIL'}")

    per2 = run_roundtrip(frames, P, NoisyGenerator())
    agg2, _ = summarize(per2)
    neg_ok = all(agg2[k] > agg[k] for k in agg)
    print(f"\nNEGATIVE control (G = identity + eps): residual {agg2}")
    print(f"  -> must exceed positive ........ {'PASS' if neg_ok else 'FAIL'}")

    print(f"\nSELFTEST {'PASS' if (pos_ok and neg_ok) else 'FAIL'}")
    return 0 if (pos_ok and neg_ok) else 1

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="Infinigen render directory with GT passes")
    ap.add_argument("--perception", default="stub", help="pkg.mod:factory")
    ap.add_argument("--generator", default="stub", help="pkg.mod:factory")
    ap.add_argument("--plausibility", default=None, help="pkg.mod:factory, optional score residual")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="runs/closed-loop")
    ap.add_argument("--autoregressive", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--controls-only", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    os.makedirs(args.out, exist_ok=True)
    frames = load_frames(args.source, args.limit) if args.source else []
    if not frames:
        print("no frames found. pass --source <render dir> or run --selftest",
              file=sys.stderr)
        return 2

    P = load_factory(args.perception)
    G = load_factory(args.generator)
    P = P() if P else IdentityPerception()
    G = G() if G else IdentityGenerator()
    S = load_factory(args.plausibility)
    S = S() if S else None

    results = {}
    # --- controls -----------------------------------------------------------
    pos = run_roundtrip(frames, IdentityPerception(), IdentityGenerator())
    neg = run_roundtrip(frames, IdentityPerception(), NoisyGenerator())
    agg_pos, _ = summarize(pos); agg_neg, _ = summarize(neg)
    pos_ok = all(abs(v) < 1e-12 for v in agg_pos.values())
    neg_ok = all(agg_neg[k] > agg_pos[k] for k in agg_pos)
    results["controls"] = dict(positive=dict(residual=agg_pos, must_be_zero=bool(pos_ok)),
                                negative=dict(residual=agg_neg, must_exceed=bool(neg_ok)))
    print(f"\nCONTROLS  positive(identity) residual={agg_pos} zero? {pos_ok}")
    print(f"           negative(identity+eps) residual={agg_neg} exceeds? {neg_ok}")
    if not (pos_ok and neg_ok):
        print("\nABORT: the harness failed its own controls. every number below is void.")
        return 3

    # --- the real measurement ----------------------------------------------
    if args.controls_only:
        per = None; agg = {}; growth = {}
    else:
        per = run_roundtrip(frames, P, G, plausibility=S, autoregressive=args.autoregressive)
        agg, growth = summarize(per)

    # --- E3: consistency and plausibility are different signals -------------
    e3 = None
    if per and "plausibility" in (per[0] if per else {}):
        cons = np.array([f.get("depth", np.nan) for f in per], dtype=float)
        plaus = np.array([f.get("plausibility", np.nan) for f in per], dtype=float)
        m = np.isfinite(cons) & np.isfinite(plaus)
        if m.sum() > 3:
            corr = float(np.corrcoef(cons[m], plaus[m])[0, 1])
            med_cons = np.median(cons[m]); med_pl = np.median(plaus[m])
            bad = int(((cons[m] < med_cons) & (plaus[m] > med_pl)).sum())
            e3 = dict(correlation=round(corr, 4),
                      frames_consistent_but_implausible=bad,
                      n_frames=int(m.sum()),
                      prediction_holds=bool(abs(corr) < 0.9))
            print(f"\nE3  consistency vs plausibility: corr={corr:+.4f}  "
                  f"consistent-but-impossible frames = {bad}/{int(m.sum())}")

    receipt = dict(
        source=args.source, perception=getattr(P, "name", str(P)),
        generator=getattr(G, "name", str(G)),
        plausibility=getattr(S, "name", None),
        autoregressive=args.autoregressive,
        n_frames=len(frames),
        residual=agg, drift_growth_log_per_frame=growth,
        controls=results["controls"],
        E3_consistency_is_not_plausibility=e3,
        predictions=dict(
            E1_growth_with_frame_index=(growth or {}),
            E3_consistency_vs_plausibility=e3,
        ),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    with open(os.path.join(args.out, "closed-loop-receipt.json"), "w") as f:
        json.dump(receipt, f, indent=2)
    if per:
        with open(os.path.join(args.out, "per-frame.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per[0].keys())); w.writeheader()
            for r in per: w.writerow(r)
    print(f"\nresidual per modality: {agg}")
    if growth: print(f"drift growth (log-residual per frame): {growth}")
    print(f"receipt: {os.path.join(args.out, 'closed-loop-receipt.json')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
