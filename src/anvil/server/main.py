"""``anvil-inspect`` console script entry point.

Read-only localhost inspector for ``.anvil/`` pin directories. Single
command, no config: opens your browser at the index page and serves
HTML rendered from whatever's on disk.
"""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

import click
import uvicorn

from anvil.manifest import DEFAULT_PIN_ROOT
from anvil.server.app import create_app


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--root",
    "pin_root",
    type=click.Path(path_type=Path),
    default=None,
    help="Pin storage root to inspect. Defaults to ./.anvil in the cwd.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Network interface to bind to. Stays localhost-only by default.",
)
@click.option(
    "--port",
    type=int,
    default=7777,
    show_default=True,
    help="Port to serve on.",
)
@click.option(
    "--no-browser",
    is_flag=True,
    default=False,
    help="Don't auto-open a browser tab.",
)
def main(
    pin_root: Path | None,
    host: str,
    port: int,
    no_browser: bool,
) -> None:
    """Browse Anvil pins in a local web UI."""
    root = (pin_root or Path.cwd() / DEFAULT_PIN_ROOT).resolve()
    if not root.exists():
        click.echo(
            f"No pin directory at {root}. Run `anvil pin --name ...` first, "
            "or pass --root to point at an existing .anvil/ tree.",
            err=True,
        )
        sys.exit(1)

    app = create_app(root)
    url = f"http://{host}:{port}"
    click.echo(f"Anvil inspector serving from {root}\n  → {url}")
    if not no_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
