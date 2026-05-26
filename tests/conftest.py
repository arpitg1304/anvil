"""Shared test fixtures.

The CLI tests exercise ``pin`` and ``check`` end-to-end via Click's runner.
When the ``[full]`` extra is installed, ``load_embedder`` would otherwise
load a real HF model — slow, requires network, and prints a tqdm progress
bar to stdout which corrupts the ``--json`` output the tests parse.

This conftest forces ``anvil.cli.load_embedder`` to return ``None`` for
those tests, exercising the histogram fallback path deterministically. The
real embedder is covered by ``test_embedder.py`` via stubs and by manual
``anvil pin/check`` runs against a live camera.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_heavy_models_in_cli(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if "test_cli" not in request.node.nodeid:
        return
    monkeypatch.setattr("anvil.cli.load_embedder", lambda *a, **kw: None)
    monkeypatch.setattr("anvil.cli.load_keypoint_pipeline", lambda *a, **kw: None)
