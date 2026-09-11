# Typer CLI: flp ask, flp ingest, flp eval, flp calibrate. Spec Appendix D.
"""Command line entry point. Subcommands are stubs until their milestone lands."""

import typer

app = typer.Typer(no_args_is_help=True, help="Front Line PHP RAG.")


def _not_yet(issue: int) -> None:
    typer.echo(f"not implemented yet, see issue #{issue}", err=True)
    raise typer.Exit(code=2)


@app.command()
def ask(question: str) -> None:
    """Ask one question. Milestone 2 (retrieval only), Milestone 3 (full answer)."""
    _not_yet(15)


@app.command()
def ingest() -> None:
    """Run Stages 1-8. Milestone 1 and 2."""
    _not_yet(14)


@app.command(name="eval")
def eval_() -> None:
    """Run the evaluation harness. Milestone 6."""
    _not_yet(34)


@app.command()
def calibrate() -> None:
    """Threshold sweep of spec Section 8.4. Milestone 6."""
    _not_yet(35)
