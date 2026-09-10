"""Convenience entry point for LUMI Robot Motion Visual Simulator.

Usage:
    python simulate_movement.py          # Opens Desktop GUI (Tkinter)
    python simulate_movement.py --web    # Opens Web Browser Dashboard (port 8080)
"""

from execution.simulate_movement import main

if __name__ == "__main__":
    main()
