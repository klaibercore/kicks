"""Legacy entry point. Use `kicks cluster` instead."""
from kicks.cli import app

if __name__ == "__main__":
    app(["cluster"])
