"""
Dedicated Dashboard Server Launcher.
Instantly launches the interactive web dashboard for visualizing results.
"""

import sys
import argparse
from pathlib import Path
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

def main():
    parser = argparse.ArgumentParser(description="Launch Smart Campus Energy Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("FEDERATED SMART CAMPUS ENERGY ANOMALY MONITOR DASHBOARD")
    print(f"Access UI in browser at: http://{args.host}:{args.port}")
    print("="*70 + "\n")

    uvicorn.run("dashboard.app:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
