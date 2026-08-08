from __future__ import annotations

import argparse
import importlib


COMMANDS = {
    "data": "tears_training.data",
    "summaries": "tears_training.summaries",
    "train": "tears_training.train",
    "evaluate": "tears_training.evaluate",
    "orchestrate": "tears_training.orchestrate",
    "report": "tears_training.report",
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m tears_training")
    parser.add_argument("command", choices=COMMANDS)
    args, remainder = parser.parse_known_args()
    module = importlib.import_module(COMMANDS[args.command])
    module.main(remainder)


if __name__ == "__main__":
    main()
