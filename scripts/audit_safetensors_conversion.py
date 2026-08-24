"""Checkpoint-level investigation of safetensors as a storage/loading
alternative for lingbot-map.pt — see CLAUDE.md, "Investigación: conversión a
safetensors", for context and results.

Scope (deliberately limited, per instruction — does NOT touch demo.py, does
NOT run the 10-image campaign, does NOT re-instantiate GCTStream):
  1. Convert a COPY of the checkpoint to .safetensors (original .pt untouched).
  2. Verify tensor count, names, shapes, dtypes, and exact values against the
     original (torch.equal on every tensor — cheap enough at 1342 tensors).
  3. Measure both file sizes.
  4. Probe two loading strategies safetensors exposes (eager load_file() and
     lazy per-tensor safe_open()) and report their feasibility/characteristics
     for a state_dict that already matches GCTStream 1:1 (missing=0,
     unexpected=0 already confirmed in every previous load — see baseline
     sections above — so key compatibility does not need to be re-derived by
     re-instantiating the model here).

Pre-check already done manually (see CLAUDE.md): 1342 tensors, all
torch.float32, 0 non-contiguous, 0 shared-storage groups — no known
safetensors blockers going in.
"""
import argparse
import os
import time

import torch
from safetensors.torch import save_file, load_file
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="lingbot-map.pt")
    ap.add_argument("--dst", default="lingbot-map_converted.safetensors")
    args = ap.parse_args()

    print(f"torch version: {torch.__version__}", flush=True)

    # --- 1. Load original checkpoint (read-only, mmap; original file untouched) ---
    t0 = time.time()
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False, mmap=True)
    print(f"[1] Loaded original checkpoint in {time.time() - t0:.2f}s: {len(ckpt)} tensors", flush=True)

    dtypes = sorted({str(t.dtype) for t in ckpt.values()})
    non_contig = [k for k, t in ckpt.items() if not t.is_contiguous()]
    print(f"    dtypes present: {dtypes}", flush=True)
    print(f"    non-contiguous tensors: {len(non_contig)}", flush=True)

    # safetensors requires each tensor to own distinct memory unless handled
    # specially — confirm no shared storage (already checked once manually,
    # re-verified here as part of the reproducible script).
    storage_groups = {}
    for name, t in ckpt.items():
        storage_groups.setdefault(t.untyped_storage().data_ptr(), []).append(name)
    shared = {ptr: names for ptr, names in storage_groups.items() if len(names) > 1}
    print(f"    shared-storage groups: {len(shared)}", flush=True)
    if shared:
        for ptr, names in list(shared.items())[:10]:
            print(f"      shared: {names}", flush=True)

    if non_contig or shared:
        print("BLOCKER: checkpoint has non-contiguous or shared-storage tensors — "
              "safetensors conversion needs special handling (.contiguous()/.clone()) "
              "not implemented in this script. Stopping.", flush=True)
        return

    # --- 2. Convert (copy) to safetensors ---
    # save_file needs plain CPU tensors; materialize each from the mmap-backed
    # dict (this is the same cost torch.load(mmap=True) already defers to
    # "whenever you touch the data" — here we touch all of it, once, to write
    # the new file).
    t0 = time.time()
    save_file(dict(ckpt), args.dst)
    conv_time = time.time() - t0
    print(f"[2] Converted to {args.dst} in {conv_time:.2f}s", flush=True)

    # --- 3. File size comparison ---
    src_bytes = os.path.getsize(args.src)
    dst_bytes = os.path.getsize(args.dst)
    print(
        f"[3] File sizes: {args.src}={src_bytes / 1e9:.3f}GB  "
        f"{args.dst}={dst_bytes / 1e9:.3f}GB  "
        f"diff={dst_bytes - src_bytes:+,d} bytes ({(dst_bytes - src_bytes) / 1e6:+.2f}MB)",
        flush=True,
    )

    # --- 4. Verification: count, names, shapes, dtypes, exact values ---
    t0 = time.time()
    reloaded = load_file(args.dst, device="cpu")
    reload_time = time.time() - t0
    print(f"[4] load_file() (eager) reloaded {len(reloaded)} tensors in {reload_time:.2f}s", flush=True)

    orig_keys = set(ckpt.keys())
    new_keys = set(reloaded.keys())
    print(f"    key sets identical: {orig_keys == new_keys}", flush=True)
    if orig_keys != new_keys:
        print(f"    only in original: {sorted(orig_keys - new_keys)[:10]}", flush=True)
        print(f"    only in converted: {sorted(new_keys - orig_keys)[:10]}", flush=True)

    shape_mismatches, dtype_mismatches, value_mismatches = [], [], []
    for name in orig_keys & new_keys:
        a, b = ckpt[name], reloaded[name]
        if tuple(a.shape) != tuple(b.shape):
            shape_mismatches.append(name)
        if a.dtype != b.dtype:
            dtype_mismatches.append(name)
        elif not torch.equal(a, b):
            value_mismatches.append(name)

    print(
        f"    shape mismatches: {len(shape_mismatches)} | "
        f"dtype mismatches: {len(dtype_mismatches)} | "
        f"value mismatches (bit-exact torch.equal): {len(value_mismatches)}",
        flush=True,
    )
    all_ok = (
        orig_keys == new_keys
        and not shape_mismatches
        and not dtype_mismatches
        and not value_mismatches
    )
    print(f"    VERIFICATION {'PASSED' if all_ok else 'FAILED'}", flush=True)

    # --- 5. Loading-strategy feasibility probe (lazy, per-tensor, mmap-native) ---
    print("\n[5] Loading-strategy probe: safe_open() lazy per-tensor access", flush=True)
    t0 = time.time()
    with safe_open(args.dst, framework="pt", device="cpu") as f:
        open_time = time.time() - t0
        keys = list(f.keys())
        print(f"    safe_open() opened file (header parsed) in {open_time:.3f}s, {len(keys)} keys visible", flush=True)
        t1 = time.time()
        sample_name = keys[0]
        sample = f.get_tensor(sample_name)
        one_tensor_time = time.time() - t1
        print(
            f"    get_tensor('{sample_name}') materialized ONE tensor "
            f"(shape={tuple(sample.shape)}) in {one_tensor_time:.4f}s "
            f"— confirms per-tensor lazy access works without touching the rest of the file",
            flush=True,
        )
        # Key-compatibility with GCTStream's expected state_dict is already
        # established (every prior load — baseline, mmap, verify_load_fix —
        # reported missing=0 unexpected=0 against this exact key set), so we
        # don't need to re-instantiate the model here to confirm names match.
        print(
            "    Note: state_dict key compatibility with GCTStream already confirmed "
            "(missing=0/unexpected=0 in every prior load of this checkpoint) — "
            "these are the SAME 1342 keys, so a safetensors-loaded dict would "
            "load into the model with identical strict=False semantics.",
            flush=True,
        )

    print("\nDONE - audit_safetensors_conversion complete.", flush=True)
    print(
        f"\nSUMMARY: conv_time={conv_time:.2f}s reload_time={reload_time:.2f}s "
        f"size_diff_MB={(dst_bytes - src_bytes) / 1e6:.2f} verification={'PASS' if all_ok else 'FAIL'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
