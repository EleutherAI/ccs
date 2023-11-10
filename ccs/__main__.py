"""Main entry point for `ccs`."""

from dataclasses import dataclass

from simple_parsing import ArgumentParser

from ccs.evaluation.evaluate import Eval
from ccs.plotting.command import Plot
from ccs.training.sweep import Sweep
from ccs.training.train import Elicit


@dataclass
class Command:
    """Some top-level command"""

    command: Elicit | Eval | Sweep | Plot

    def execute(self):
        return self.command.execute()


def run():
    parser = ArgumentParser(add_help=False)
    parser.add_arguments(Command, dest="run")
    args = parser.parse_args()
    run: Command = args.run
    run.execute()


if __name__ == "__main__":
    run()
