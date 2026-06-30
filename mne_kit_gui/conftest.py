# -*- coding: utf-8 -*-
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import shutil

import pytest

from mne.datasets import testing
data_path = testing.data_path(download=False)
subjects_dir = data_path / 'subjects'


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
    always::ResourceWarning
    """  # noqa: E501
    for warning_line in warning_lines.split('\n'):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith('#'):
            config.addinivalue_line('filterwarnings', warning_line)


@pytest.fixture(scope='function', params=[testing._pytest_param()])
def subjects_dir_tmp(tmp_path):
    """Copy MNE-testing-data subjects_dir to a temp dir for manipulation."""
    for key in ('sample', 'fsaverage'):
        shutil.copytree(subjects_dir / key, tmp_path / key)
    return tmp_path
