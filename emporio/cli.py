"""Interactive CLI: ``python -m emporio.cli [--session NAME]``."""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Chat with Mel, the Empório da Música assistant")
    parser.add_argument("--session", default="default", help="session id (history is saved per session)")
    parser.add_argument("--reset", action="store_true", help="start the session from scratch")
    args = parser.parse_args(argv)

    from .agent import Agent  # after load_dotenv so env vars are visible
    agent = Agent()
    if args.reset:
        agent.history.save(args.session, [])
    print(f"Mel ({agent.model}) — type 'exit' to quit.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text.lower() in {"exit", "quit", "sair"}:
            return 0
        if text:
            print(f"Mel: {agent.reply(args.session, text)}\n")


if __name__ == "__main__":
    sys.exit(main())
