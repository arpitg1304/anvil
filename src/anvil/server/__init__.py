"""Read-only inspector UI for Anvil pins.

Serves a localhost HTML view over a ``.anvil/`` directory: pin grid on
the home page, per-pin detail with the reference image, manifest, named
objects, and the full diff history. Launched via the ``anvil-inspect``
console script (see ``anvil.server.main``).

The UI is deliberately read-only and stateless — no auth, no database,
no actions. Everything it shows is straight from the filesystem.
"""

from anvil.server.app import create_app

__all__ = ["create_app"]
