#!/usr/bin/env python3
"""
CLI runner for the Cross-Encoder Re-ranking Engine (Phase 6).
"""

import sys
import os

# Add root folder to sys path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.reranker.cli import main

if __name__ == "__main__":
    main()
