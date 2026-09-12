"""Unified command-line entry point for the model identity pipeline."""
from __future__ import annotations

import argparse


COMMANDS = ("resolver", "fetch", "search", "verify", "compare", "export")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    args, remainder = parser.parse_known_args(argv)
    module = __import__(f"identity.{args.command}", fromlist=["main"])
    return module.main(remainder)
