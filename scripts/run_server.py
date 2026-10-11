#!/usr/bin/env python
"""Local development server runner for the Flood Damage Mapper FastAPI backend.

Usage:
    python scripts/run_server.py
    python scripts/run_server.py --port 8080 --reload
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import uvicorn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    ap.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    args = ap.parse_args()

    print(f"\n==============================================================")
    print(f" Starting Flood Damage Mapper Backend (FastAPI)")
    print(f" Dashboard:    http://{args.host}:{args.port}/floodintel/")
    print(f" API Base:     http://{args.host}:{args.port}/api")
    print(f" Swagger UI:   http://{args.host}:{args.port}/docs")
    print(f" Health Check: http://{args.host}:{args.port}/api/v1/health")
    print(f" Note:         127.0.0.1 refers to this local computer running the browser.")
    print(f"               Keep this terminal window open while using the dashboard.")
    print(f"==============================================================\n")

    uvicorn.run("floodmap.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
