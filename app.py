"""Legacy entry point. Use `kicks serve` instead."""
from kicks.cli import app

if __name__ == "__main__":
    app(["serve"])
