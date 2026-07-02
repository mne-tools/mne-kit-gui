# -*- coding: utf-8 -*-
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import gc
from collections.abc import Iterator
from typing import Protocol, TypeVar

import pytest

from qtpy.QtCore import QObject

_T = TypeVar("_T", bound=QObject)


class FindChild(Protocol):
    """Callable returned by the :func:`find_child` fixture."""

    def __call__(self, parent: QObject, kind: type[_T], name: str) -> _T:
        """Return ``parent``'s named child of ``kind``, asserting it exists."""
        ...


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest options."""
    warning_lines = r"""
    error::
    ignore:.*Call to deprecated .* vtk.*:DeprecationWarning
    ignore:SelectableGroups dict interface.*:DeprecationWarning
    ignore:.*imp module is deprecated in favour of.*:DeprecationWarning
    ignore:.*np\.loads is deprecated.*:DeprecationWarning
    ignore:^numpy\.ufunc size changed.*:RuntimeWarning
    ignore:.*invalid escape sequence.*:
    ignore:.*an integer is required \(got type.*:DeprecationWarning
    ignore:.*distutils Version classes are deprecated.*:DeprecationWarning
    ignore:.*to a dtype is deprecated.*:DeprecationWarning
    ignore:.*is a deprecated alias for the builtin.*:DeprecationWarning
    ignore:.*deprecated method GetIsPicking.*:DeprecationWarning
    ignore:module 'sre_.*' is deprecated:DeprecationWarning
    ignore:Implementing implicit namespace packages.*:DeprecationWarning
    ignore:Deprecated call to `pkg_resources.*:DeprecationWarning
    ignore:pkg_resources is deprecated as an API.*:DeprecationWarning
    ignore:numpy\.ndarray size changed.*:RuntimeWarning
    ignore:events_as_annotations defaults to False.*:FutureWarning
    ignore:Setting the shape on a NumPy array[\s\S]*:DeprecationWarning
    always::ResourceWarning
    """  # noqa: E501
    for warning_line in warning_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)


@pytest.fixture(autouse=True)
def _qapp(qtbot) -> Iterator[None]:
    """Ensure a QApplication exists for every test.

    Many objects create Qt widgets (e.g. a QProgressDialog) even when no GUI is
    shown, which aborts the interpreter if no QApplication has been created.
    Depending on ``qtbot`` here guarantees one exists for the whole test run.
    """
    yield


@pytest.fixture
def check_gc(qtbot) -> Iterator[None]:
    """Check that things are garbage collected after closing a GUI."""
    yield
    qtbot.wait(200)  # wait for the close to finish
    gc.collect()


@pytest.fixture
def find_child() -> FindChild:
    """Return a helper that looks up a named child widget.

    ``QObject.findChild`` is typed as returning ``kind | None``; this wraps it
    with an assertion so tests fail fast (and stay type-clean) when a widget
    with the given object name is missing.
    """

    def _find_child(parent: QObject, kind: type[_T], name: str) -> _T:
        child = parent.findChild(kind, name)
        assert child is not None, f"no {kind.__name__} named {name!r}"
        return child

    return _find_child
