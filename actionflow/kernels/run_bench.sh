#!/bin/bash
# Run Action-Flow kernel benchmark and format output tables.
# Usage: bash run_bench.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
RAW=$(mktemp /tmp/af_bench_XXXXXX.tsv)
trap "rm -f $RAW" EXIT

echo "Running benchmark..."
python3 actionflow/kernels/bench_real.py > "$RAW" 2>&1

PEAK=238

echo ""
echo "================================================================================"
echo "  Action-Flow Kernel Benchmark — Real Shapes (openvla-7b, bf16, Thor)"
echo "  Peak DRAM bandwidth: ${PEAK} GB/s"
echo "================================================================================"

# ── helper: format a TSV table nicely ──
fmt_table() {
    awk -F'\t' -v peak="$PEAK" '
    function pad(s,w) { return sprintf("%*s",w,s) }
    BEGIN { getline; for(i=1;i<=NF;i++) h[i]=$i; nr=0 }
    /^[0-9]/ {
        nr++
        for(i=1;i<=NF;i++) d[nr,i]=$i
    }
    END {
        # compute col widths
        for(i=1;i<=NF;i++) {
            w[i]=length(h[i])
            for(r=1;r<=nr;r++) {
                l=length(d[r,i])
                if(l>w[i]) w[i]=l
            }
            if(w[i]<4) w[i]=4
        }
        # header
        for(i=1;i<=NF;i++) printf "%s%s", pad(h[i],w[i]+2), (i==NF?"\n":"")
        for(i=1;i<=NF;i++) printf "%s%s", str_repeat("-",w[i]+2), (i==NF?"\n":"")
        # data
        for(r=1;r<=nr;r++) {
            for(i=1;i<=NF;i++) printf "%s%s", pad(d[r,i],w[i]+2), (i==NF?"\n":"")
        }
    }
    function str_repeat(s,n, t) { t=""; for(i=1;i<=n;i++) t=t s; return t }
    '
}

echo ""
echo "── 1. RMSNorm  (N=4096) ──"
awk '/^TABLE RMSNorm/,/^$/' "$RAW" | grep -v '^TABLE\|^$' | fmt_table

echo ""
echo "── 2. FusedRoPE+KV  (H_q=32, H_kv=32, D=128) ──"
awk '/^TABLE FusedRoPE/,/^$/' "$RAW" | grep -v '^TABLE\|^$' | fmt_table

echo ""
echo "── 3. Shift KV Cache  (H_kv=32, D=128) ──"
awk '/^TABLE ShiftKV/,/^$/' "$RAW" | grep -v '^TABLE\|^$' | fmt_table

echo ""
echo "================================================================================"
echo "  Done."
echo "================================================================================"
