"""Legacy entry point. Use `kicks train` instead."""
from kicks.cli import app

if __name__ == "__main__":
    app(["train"])
