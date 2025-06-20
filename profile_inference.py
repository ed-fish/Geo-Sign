#!/usr/bin/env python3
"""
Profile Uni-Sign inference (FLOPs + latency).

Artefacts
---------
• console tables (DeepSpeed & PyTorch)
• profile_summary.txt   – full report
• profile_metrics.json  – headline numbers
• flops_trace.json      – Chrome/TB trace  (if DS ≥ 0.13)
• tb_logs/              – timeline for TensorBoard
"""
# --------------------------------------------------------------------------- #
import os, argparse, functools, io, json, contextlib, inspect
import torch

# ---- Uni-Sign modules ------------------------------------------------------ #
import utils
from config import train_label_paths
from datasets import S2T_Dataset
from models import Uni_Sign

# ---- Profiling toolkits ---------------------------------------------------- #
from deepspeed.profiling.flops_profiler import FlopsProfiler, get_model_profile
from torch.profiler import (
    profile, ProfilerActivity, schedule,
    tensorboard_trace_handler,
)

# --------------------------------------------------------------------------- #
def get_dummy_batch(args, device):
    ds = S2T_Dataset(path=train_label_paths[args.dataset],
                     args=args,
                     phase="train")
    batch = [ds[i] for i in range(args.batch_size)]
    src, tgt = ds.collate_fn(batch)
    for d in (src, tgt):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.to(device, non_blocking=True)
    return src, tgt
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Profile Uni-Sign forward pass (FLOPs & latency)",
        parents=[utils.get_args_parser()],
    )
    parser.set_defaults(batch_size=4, dataset="CSL_Daily", task="SLT")
    args = parser.parse_args()
    device = torch.device("cuda", 0)

    # ---------------- Build model + sample batch ----------------------------
    model = Uni_Sign(args=args).to(device)
    src, tgt = get_dummy_batch(args, device)

    # ---------------- DeepSpeed FLOP profiler -------------------------------
    flop_prof = FlopsProfiler(model)
    flop_prof.start_profile()

    # Wrap DeepSpeed’s patched einsum so tuple operands don’t break FLOP hook
    _einsum_ds = torch.einsum
    def _einsum_tuple_safe(eq, *ops, **kw):
        if len(ops) == 1 and isinstance(ops[0], (tuple, list)):
            ops = tuple(ops[0])
        return _einsum_ds(eq, *ops, **kw)
    torch.einsum = functools.wraps(_einsum_ds)(_einsum_tuple_safe)

    with torch.no_grad():
        _ = model(src, tgt)
    flop_prof.stop_profile()

    print("\n================ DeepSpeed FLOPs per module ================")
    flop_prof.print_model_profile(profile_step=0, module_depth=2, top_modules=20)

    if hasattr(flop_prof, "export_chrome_trace"):
        flop_prof.export_chrome_trace("flops_trace.json")
        print("[✓] flops_trace.json written")
    else:
        print("[i] DeepSpeed < 0.13 – skipping flops_trace.json")

    # ---------------- PyTorch profiler (quick) ------------------------------
    prof_sched = schedule(wait=1, warmup=1, active=2)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_sched,
        on_trace_ready=tensorboard_trace_handler("./tb_logs"),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True,
    ) as prof:
        for _ in range(4):
            with torch.no_grad():
                _ = model(src, tgt)
            prof.step()

    print("\n================ PyTorch Profiler (top ops) ================")
    torch_table_txt = prof.key_averages(group_by_input_shape=True)\
                          .table(sort_by="flops", row_limit=30)
    print(torch_table_txt)

    # ---------------- Robust headline numbers -------------------------------
    flops_total = macs_total = params_total = None
    try:
        # DS ≥ 0.13
        flops_total, macs_total, params_total = flop_prof.get_total_flops_params()
    except AttributeError:
        # older DS: try get_model_profile
        try:
            sig = inspect.signature(get_model_profile)
            if "inputs" in sig.parameters:
                flops_total, macs_total, params_total = get_model_profile(
                    model, inputs=(src, tgt), print_profile=False)
            else:
                flops_total, macs_total, params_total = get_model_profile(
                    model, (src, tgt), print_profile=False)
        except Exception:
            # absolute fallback: sum the per-module MACs accumulated by DS
            try:
                # flop_prof.original_modules contains tuples (module, macs, flops, params)
                macs_total = sum(m[1] for m in flop_prof.original_modules)
                flops_total = sum(m[2] for m in flop_prof.original_modules)
                params_total = sum(m[3] for m in flop_prof.original_modules)
            except Exception:
                pass   # leave as None

    # make them printable
    flop_str   = f"{flops_total/1e12:.3f} TFLOPs" if flops_total else "N/A"
    macs_str   = f"{macs_total/1e12:.3f} TMACs"  if macs_total else "N/A"
    params_str = f"{params_total/1e6:.2f} M"     if params_total else "N/A"

    # capture DS table for summary
    ds_buf = io.StringIO()
    with contextlib.redirect_stdout(ds_buf):
        flop_prof.print_model_profile(profile_step=0, module_depth=2, top_modules=20)
    ds_table_txt = ds_buf.getvalue()

    # ---------------- Summary text ------------------------------------------
    summary_txt = (
        "\n================ Uni-Sign profiling summary ================\n"
        f"Batch size         : {args.batch_size}\n"
        f"Dataset            : {args.dataset}\n"
        f"Task               : {args.task}\n"
        f"Total forward FLOPs: {flop_str}\n"
        f"Total MACs         : {macs_str}\n"
        f"Total parameters   : {params_str}\n"
        "\n--- DeepSpeed per-module table (top 20) ---\n"
        f"{ds_table_txt}"
        "\n--- PyTorch per-op table (top 30) ---\n"
        f"{torch_table_txt}\n"
    )

    print(summary_txt)
    with open("profile_summary.txt", "w") as f:
        f.write(summary_txt)
    print("[✓] profile_summary.txt written")

    with open("profile_metrics.json", "w") as f:
        json.dump({
            "batch_size": args.batch_size,
            "dataset": args.dataset,
            "task": args.task,
            "flops_total": flops_total,
            "macs_total": macs_total,
            "params_total": params_total,
        }, f, indent=2)
    print("[✓] profile_metrics.json written")
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
