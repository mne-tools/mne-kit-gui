# -*- coding: utf-8 -*-
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import os.path as op
import shutil

import pytest

from mne.datasets import testing
data_path = testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')


def pytest_configure(config):
    """Configure pytest options."""
    warning_lines = r"""
    error::
    ignore:.*in an Any trait will be shared.*:DeprecationWarning
    ignore:.*Call to deprecated .* vtk.*:DeprecationWarning
    always::ResourceWarning
    """  # noqa: E501
    for warning_line in warning_lines.split('\n'):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith('#'):
            config.addinivalue_line('filterwarnings', warning_line)


@pytest.fixture(scope='function', params=[testing._pytest_param()])
def subjects_dir_tmp(tmpdir):
    """Copy MNE-testing-data subjects_dir to a temp dir for manipulation."""
    for key in ('sample', 'fsaverage'):
        shutil.copytree(op.join(subjects_dir, key), str(tmpdir.join(key)))
    return str(tmpdir)
