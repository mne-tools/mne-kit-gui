# -*- coding: utf-8 -*-
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import gc

import pytest


def pytest_configure(config):
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
def _qapp(qtbot):
    """Ensure a QApplication exists for every test.

    Many objects create Qt widgets (e.g. a QProgressDialog) even when no GUI is
    shown, which aborts the interpreter if no QApplication has been created.
    Depending on ``qtbot`` here guarantees one exists for the whole test run.
    """
    yield


@pytest.fixture
def check_gc(qtbot):
    """Check that things are garbage collected after closing a GUI."""
    yield
    qtbot.wait(200)  # wait for the close to finish
    gc.collect()
