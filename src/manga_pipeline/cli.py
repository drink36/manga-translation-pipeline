"""Explicit commands: planning/doctor never invoke models or submit jobs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .config import STAGES, load_config


def server_command(cfg):
    server, model = cfg.get("server", {}), cfg["vlm"]
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model["model"], "--served-model-name", model["model"], "--host", str(server.get("host", "127.0.0.1")), "--port", str(server.get("port", 8000))]
    if model.get("revision"):
        command += ["--revision", model["revision"]]
    for key, default in (("tensor_parallel_size", 1), ("max_model_len", 16384), ("max_num_seqs", 1), ("max_num_batched_tokens", 4096), ("gpu_memory_utilization", 0.85), ("dtype", "bfloat16")):
        command += ["--" + key.replace("_", "-"), str(server.get(key, default))]
    command += ["--limit-mm-per-prompt", '{"image":2,"video":0}']
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resumable manga translation. No jobs are submitted by this tool.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "doctor", "run", "serve"):
        p = commands.add_parser(name)
        p.add_argument("--config", default="configs/pipeline.yaml")
        if name == "run":
            p.add_argument("--stage", choices=("all", *STAGES), default="all")
            p.add_argument("--force", action="store_true", help="Recompute requested stage, preserving original input files")
        if name == "serve":
            p.add_argument("--print-command", action="store_true", help="Print only; no model load/download")
    demo = commands.add_parser("demo", help="Generate and process synthetic fixtures with scripted responses, CPU only")
    demo.add_argument("--output", default="output/demo", help="Dedicated demo project directory")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import create_demo
            from .pipeline import Pipeline
            config_path = create_demo(Path(args.output))
            Pipeline(load_config(config_path)).run()
            print("Synthetic demo only: no OCR/translation model was run.")
            return 0
        cfg = load_config(args.config)
        if args.command == "plan":
            print(json.dumps({"input": cfg["input"], "output": cfg["output"], "mode": cfg.get("mode", "real"), "stages": list(STAGES), "detector": cfg.get("detector"), "ocr_routes": cfg.get("ocr", {}).get("routes"), "ocr_readers": cfg.get("ocr", {}).get("readers"), "vlm": cfg.get("vlm"), "audit": cfg.get("audit"), "retries": cfg.get("retries"), "server_command": shlex.join(server_command(cfg))}, indent=2, ensure_ascii=False))
            return 0
        if args.command == "doctor":
            modules = ("PIL", "numpy", "yaml", "torch", "rfdetr", "transformers", "manga_ocr", "paddleocr", "vllm")
            font = cfg.get("render", {}).get("font_path")
            print(json.dumps({"python": sys.version, "packages_present": {name: importlib.util.find_spec(name) is not None for name in modules}, "font_exists": bool(font and Path(font).is_file()), "input_exists": Path(cfg["input"]).exists(), "endpoint": cfg["vlm"]["endpoint"], "note": "Read-only local check. No weights, API requests, or GPU checks."}, indent=2))
            return 0
        if args.command == "serve":
            command = server_command(cfg)
            print(shlex.join(command), flush=True)
            if args.print_command:
                return 0
            os.execv(command[0], command)
        from .pipeline import Pipeline
        Pipeline(cfg).run(args.stage, args.force)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
