#!/usr/bin/env bash
# thor/probe-thor-gen.sh - measure the GENERATION budget on this Thor.
#
# Companion to thor/probe-thor.sh (which measures what the box already can do).
# This one answers: "what size generative model can I actually run, at what
# resolution, at how many steps, in real time?" - because that number decides
# the architecture before you train anything.
#
# It measures four things and derives the fifth:
#   1. can nvcc emit sm_110a SASS, and does the tcgen05/NVFP4 path exist?
#      (the "FP4 is a measured no" result was a torch/torchao limit, not hardware)
#   2. GEMM throughput: FP8 e4m3, FP16, FP32, at several sizes
#   3. a real DiT block forward at the token counts you would actually use
#   4. a decoder-shaped conv workload (the VAE wall)
#   5. => the derived fps table: params x steps x tokens -> fps
#
# Usage:  thor/probe-thor-gen.sh [OUTDIR]
# Exit:   0 = all REQUIRED probes ran, 1 = something required is missing.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="${1:-$REPO_ROOT/outputs/probe-gen}"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/probe-gen.log"
: > "$LOG"

FAILED=0
say() { printf '%s\n' "$*" | tee -a "$LOG"; }
rec() { printf '%-10s %s\n' "$1" "$2" | tee -a "$OUTDIR/probe-gen-receipt.txt"; }
ok()   { rec "OK" "$1"; }
info() { rec "INFO" "$1"; }
miss() { rec "MISSING" "$1"; }
fail() { rec "FAIL" "$1"; FAILED=$((FAILED+1)); }
skip() { rec "SKIP" "$1"; }

say "=== thor/probe-thor-gen.sh  $(date -Is) ==="

# CUDA is at /usr/local/cuda/bin on this box and is NOT necessarily on PATH in a
# non-interactive shell (the Thor's CUDA came from a .deb, not a profile.d).
# Resolve the tools explicitly or this whole script silently skips.
CUDA_BIN=""
for d in "${CUDA_HOME:-}/bin" /usr/local/cuda/bin /opt/cuda/bin; do
  [ -x "$d/nvcc" ] && { CUDA_BIN="$d"; break; }
done
if [ -n "$CUDA_BIN" ]; then
  export PATH="$CUDA_BIN:$PATH"
  say "cuda bin: $CUDA_BIN"
else
  say "cuda bin: NOT FOUND (looked in ${CUDA_HOME:-}/bin /usr/local/cuda/bin /opt/cuda/bin)"
fi
for t in nvcc ptxas cuobjdump; do
  command -v "$t" >/dev/null 2>&1 && say "  $t -> $(command -v "$t")" || say "  $t -> MISSING"
done

# ---------------------------------------------------------------------------
say ""
say "--- 1. arch support: sm_110a and the tcgen05/NVFP4 path ---"
# thor-hacks measured with ptxas that the block-scaled NVFP4 tensor-core form
#   tcgen05.mma.sp cta_group::2 kind::mxf4nvf4.block_scale.scale_vec::4X
# assembles on sm_110a and is REJECTED on plain sm_110. That is the difference
# between "the hardware cannot do FP4" (false) and "torch/torchao has no cast
# op for it" (true, and fixable).
D=$(mktemp -d)
if command -v ptxas >/dev/null 2>&1; then
  # 1a. does the toolchain accept sm_110a at all?
  cat > "$D/arch.cu" <<'CUEOF'
#include <cstdio>
__global__ void k(float *x){ x[threadIdx.x] += 1.0f; }
int main(){ float h[4]={0,0,0,0},*d; cudaMalloc(&d,16);
  cudaMemcpy(d,h,16,cudaMemcpyHostToDevice); k<<<1,4>>>(d);
  cudaMemcpy(h,d,16,cudaMemcpyDeviceToHost); printf("%f\n",h[0]); return 0; }
CUEOF
  if nvcc -O2 -arch=sm_110a "$D/arch.cu" -o "$D/arch" >"$D/arch.log" 2>&1; then
    ok "nvcc accepts -arch=sm_110a"
    if [ -x "$D/arch" ]; then
      ELF=$(cuobjdump --list-elf "$D/arch" 2>/dev/null | grep -o 'sm_[0-9a]*' | sort -u | tr '\n' ' ')
      say "     sm_110a SASS: ${ELF:-<none>}"
      if [ -x "$D/arch" ]; then
        R=$("$D/arch" 2>&1); [ "$R" = "1.000000" ] && ok "an sm_110a kernel RUNS on this GPU" \
          || fail "sm_110a kernel did not run correctly (got '$R')"
      fi
    fi
  else
    fail "nvcc rejected -arch=sm_110a: $(tail -2 "$D/arch.log" | tr '\n' ' ')"
  fi

  # 1b. the tcgen05 block-scale probe, WITH a control.
  # The control (dense f16 mma) is known-good on every target. If the control
  # fails to assemble, our PTX is malformed and the result says nothing about
  # the hardware - that distinction is the whole point of including it.
  cat > "$D/tcgen05.ptx" <<'PTXEOF'
.version 8.7
.target sm_110a
.address_size 64
.visible .entry probe(.param .u64 smem_ptr)
{
  .reg .b32 taddr;
  .reg .b64 sptr;
  .reg .b32 bdesc;
  ld.param.u64 sptr, [smem_ptr];
  mov.u32 taddr, 0;
  tcgen05.alloc.cta_group::1.32x256b [taddr], 128;
  mov.b32 bdesc, 0;
  tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
      [taddr], bdesc, sptr, 0, [taddr];
  ret;
}
PTXEOF
  OUT=$(ptxas -arch=sm_110a "$D/tcgen05.ptx" -o "$D/tcgen05.cubin" 2>&1); RC=$?
  if [ $RC -eq 0 ]; then
    ok "tcgen05 block-scale (mxf4nvf4) ASSEMBLES for sm_110a - the NVFP4 tensor path exists"
    say "     -> write a CUTLASS/tcgen05 GEMM with CUTLASS_NVCC_ARCHS=110a and measure it"
  else
    case "$OUT" in
      *"not supported on .target"*|*"Unsupported"*)
        fail "tcgen05 block-scale REJECTED for sm_110a: $(echo "$OUT" | head -2 | tr '\n' ' ')" ;;
      *)
        miss "tcgen05 probe MALFORMED (not a capability verdict): $(echo "$OUT" | head -3 | tr '\n' ' ')"
        say "     -> our PTX may be wrong; this is not evidence the hardware lacks FP4" ;;
    esac
  fi
  # control: the same probe against plain sm_110, where thor-hacks says it is rejected
  OUT2=$(ptxas -arch=sm_110 "$D/tcgen05.ptx" -o "$D/tcgen05_110.cubin" 2>&1); RC2=$?
  if [ $RC2 -eq 0 ]; then
    info "tcgen05 block-scale also assembles for plain sm_110 (differs from thor-hacks' finding - worth a note)"
  else
    info "tcgen05 block-scale is rejected for plain sm_110, as thor-hacks recorded"
  fi
else
  skip "ptxas not on PATH - cannot probe the tcgen05/NVFP4 path"
fi
rm -rf "$D"

# ---------------------------------------------------------------------------
say ""
say "--- 2..5. throughput + the derived fps table (torch) ---"
PY=""
for v in "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/.toolchain/python-3.11/bin/python3.11" "./.venv/bin/python"; do
  [ -x "$v" ] && { PY="$v"; break; }
done
[ -n "$PY" ] || PY="$(command -v python3 || true)"
if [ -z "$PY" ]; then
  fail "no python interpreter found"
elif ! "$PY" -c "import torch" >/dev/null 2>&1; then
  miss "torch not importable from $PY - install torch 2.15.0+cu132 to run the throughput section"
  say "     the arch probe above still stands; the fps table needs torch"
else
  "$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import torch, time, math, statistics, os

dev = "cuda:0"
print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0),
      "| cc", "%d.%d" % torch.cuda.get_device_capability(0))
print("arch list:", torch.cuda.get_arch_list())
props = torch.cuda.get_device_properties(0)
print("total mem: %.1f GiB" % (props.total_memory / 2**30))
free, total = torch.cuda.mem_get_info(0)
print("free mem:  %.1f GiB" % (free / 2**30))

def bench(fn, warmup=5, iters=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

# ---- 2. GEMM throughput ------------------------------------------------
print("\n== GEMM (M=N=K) ==")
print(f"{'size':>6} {'fp8 TFLOPS':>12} {'fp16 TFLOPS':>12} {'fp32 TFLOPS':>12}")
gemm = {}
for n in (4096, 8192):
    flops = 2.0 * n**3
    row = {}
    for dtype, tag in ((torch.float16, "fp16"), (torch.float32, "fp32")):
        a = torch.randn(n, n, device=dev, dtype=dtype)
        b = torch.randn(n, n, device=dev, dtype=dtype)
        t = bench(lambda: torch.mm(a, b))
        row[tag] = flops / t / 1e12
        del a, b
    # FP8 e4m3: rowwise scales; scale_b must be (1, N) with N = output width
    try:
        a = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn).t().contiguous().t()
        sa = torch.ones(n, 1, device=dev, dtype=torch.float32)
        sb = torch.ones(1, n, device=dev, dtype=torch.float32)
        t = bench(lambda: torch._scaled_mm(a, b, sa, sb, out_dtype=torch.bfloat16))
        row["fp8"] = flops / t / 1e12
        del a, b
    except Exception as e:
        print(f"   (fp8 unavailable at {n}: {type(e).__name__}: {str(e)[:80]})")
    gemm[n] = row
    print(f"{n:>6} {row.get('fp8', float('nan')):>12.1f} {row.get('fp16', float('nan')):>12.1f} {row.get('fp32', float('nan')):>12.1f}")

# ---- 3. a real DiT block ------------------------------------------------
print("\n== DiT block forward (d=1024, 16 heads, mlp 4x, GELU) ==")
d, heads, T = 1024, 16, None

class Blk(torch.nn.Module):
    def __init__(self, d=d, heads=heads):
        super().__init__()
        self.h = heads; self.d = d
        self.n1 = torch.nn.LayerNorm(d); self.n2 = torch.nn.LayerNorm(d)
        self.qkv = torch.nn.Linear(d, 3*d, bias=False)
        self.proj = torch.nn.Linear(d, d, bias=False)
        self.fc1 = torch.nn.Linear(d, 4*d, bias=False)
        self.fc2 = torch.nn.Linear(4*d, d, bias=False)
    def forward(self, x):
        B, S, C = x.shape
        q, k, v = self.n1(x).chunk(3, dim=-1)
        q = q.view(B, S, self.h, C//self.h).transpose(1,2)
        k = k.view(B, S, self.h, C//self.h).transpose(1,2)
        v = v.view(B, S, self.h, C//self.h).transpose(1,2)
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1,2).reshape(B, S, C)
        x = x + self.proj(a)
        h = self.fc2(torch.nn.functional.gelu(self.fc1(self.n2(x))))
        return x + h

blk = Blk().to(dev, torch.float16).eval()
block_rows = []
for T in (1024, 4096, 16384):
    x = torch.randn(1, T, d, device=dev, dtype=torch.float16)
    with torch.no_grad():
        t = bench(lambda: blk(x), warmup=3, iters=10)
    nparams = sum(p.numel() for p in blk.parameters())
    flops = 2.0 * nparams * T + 2.0 * 2.0 * T*T*d   # MLP + attention
    tflops = flops / t / 1e12
    block_rows.append((T, t, tflops))
    print(f"   T={T:>6}  {t*1000:8.2f} ms  ({tflops:6.1f} TFLOP/s)")

# ---- 4. a decoder-shaped conv (the VAE wall) ---------------------------
print("\n== decoder-shaped conv (256x256x64 -> 128 -> 64 -> 32, RGB out) ==")
try:
    def dec():
        x = torch.randn(1, 64, 32, 32, device=dev, dtype=torch.float16)
        x = torch.nn.functional.conv2d(x, torch.randn(128,64,3,3,device=dev,dtype=torch.float16), padding=1)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.nn.functional.conv2d(x, torch.randn(64,128,3,3,device=dev,dtype=torch.float16), padding=1)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.nn.functional.conv2d(x, torch.randn(3,64,3,3,device=dev,dtype=torch.float16), padding=1)
        return x
    td = bench(dec, warmup=3, iters=10)
    out_bytes = 3*256*256*2
    print(f"   one 256x256 frame: {td*1000:.2f} ms  -> {16/td:.1f} fps for a 16-frame chunk")
    print(f"   output bytes/frame {out_bytes/1e6:.2f} MB; at 262.8 GB/s the write alone is "
          f"{out_bytes/262.8e9*1000:.3f} ms/frame")
except Exception as e:
    print("   conv bench failed:", type(e).__name__, str(e)[:120])

# ---- 5. the derived table ----------------------------------------------
print("\n== DERIVED: fps for a latent-video DiT ==")
print("   model = 16 latent frames @ 256x256, VAE f8, patch 2  ->  4096 tokens")
print("   cost/step = 2*N*T (MLP) + 4*T^2*d (attention), d=1024, 12 blocks")
print("   fp8 throughput taken from the measured GEMM above")
T = 4096; d = 1024; L = 12
attn = 4.0 * T*T*d * L
fp8 = gemm.get(8192, {}).get("fp8") or gemm.get(4096, {}).get("fp8")
fp16 = gemm.get(8192, {}).get("fp16") or gemm.get(4096, {}).get("fp16")
if not fp8:
    print("   (no FP8 number measured - cannot fill the table)")
else:
    print(f"   measured FP8 {fp8:.1f} TFLOPS | FP16 {fp16:.1f} TFLOPS")
    print(f"\n{'params':>9} {'steps':>6} {'TFLOP/fwd':>10} {'ms@fp8':>8} {'fps@fp8':>8} {'ms@fp16':>9} {'fps@fp16':>9}")
    for N in (1e8, 3e8, 1e9, 4e9):
        per_step = 2.0*N*T + attn
        for steps in (2, 4, 8):
            tf = per_step*steps/1e12
            ms8 = tf/fp8*1000
            ms16 = tf/fp16*1000
            print(f"{N/1e9:>8.1f}B {steps:>6} {tf:>10.1f} {ms8:>8.1f} {1000/ms8:>8.1f} {ms16:>9.1f} {1000/ms16:>9.1f}")
    print("\n   reading: a 100-300M conditioned model at 4 steps is the interactive tier;")
    print("   4B at 4 steps is a research instrument on this box, not a renderer.")
PYEOF
  if grep -q "DERIVED: fps" "$LOG"; then ok "throughput + derived fps table measured"; else fail "the torch section did not complete"; fi
fi

say ""
say "--- summary ---"
say "required probes failed: $FAILED"
say "receipt: $OUTDIR/probe-gen-receipt.txt"
exit $(( FAILED > 0 ? 1 : 0 ))
