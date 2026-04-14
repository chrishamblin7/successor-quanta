#!/usr/bin/env python3
"""Entry point for successor function experiments."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig
from src.train import train


def main():
    parser = argparse.ArgumentParser(description="Run a successor function experiment")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Key=value overrides (e.g. lr=0.001 seed=123)",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()

    for ov in args.overrides:
        key, val = ov.split("=", 1)
        if key not in ExperimentConfig.__dataclass_fields__:
            parser.error(f"Unknown config key: {key}")
        setattr(cfg, key, yaml.safe_load(val))

    train(cfg)


if __name__ == "__main__":
    main()
