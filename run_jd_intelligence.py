#!/usr/bin/env python3
"""
CLI runner for the Job Description Intelligence module.
"""

import sys
import os

# Add root folder to sys path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.jd_intelligence.cli import main

if __name__ == "__main__":
    main()
