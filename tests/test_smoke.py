from __future__ import annotations

import argus_puppet


def test_version_is_exposed() -> None:
    assert isinstance(argus_puppet.__version__, str)
    assert argus_puppet.__version__
