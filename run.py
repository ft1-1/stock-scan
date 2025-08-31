#!/usr/bin/env python
"""Launcher script for the options screening application."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run main
from src.main import cli

if __name__ == "__main__":
    cli()