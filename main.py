#!/usr/bin/env python3
"""
Momentum vs Vanilla SGD Convergence Visualizer
================================================
A pure-Python desktop application for comparing optimisation algorithms.

Usage
-----
    python main.py

Dependencies (all commonly pre-installed)
------------------------------------------
    numpy, matplotlib, tkinter (stdlib)
"""

import sys
import os

# Ensure the package directory is on the path when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from app import SGDVisualizerApp


def main():
    root = tk.Tk()
    root.protocol(
        "WM_DELETE_WINDOW",
        lambda: (plt.close("all"), root.destroy()),
    )
    SGDVisualizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
