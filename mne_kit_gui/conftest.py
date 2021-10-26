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
    # Fixtures
    config.addinivalue_line('usefixtures', 'traits_test')
    warning_lines = r"""
    error::
    ignore:.*in an Any trait will be shared.*:DeprecationWarning
    ignore:.*Call to deprecated .* vtk.*:DeprecationWarning
    ignore:SelectableGroups dict interface.*:DeprecationWarning
    ignore:.*use "HasTraits\.trait_set".*:DeprecationWarning
    ignore:.*imp module is deprecated in favour of.*:DeprecationWarning
    ignore:.*trait handler has been deprecated.*:DeprecationWarning
    ignore:.*np\.loads is deprecated.*:DeprecationWarning
    ignore:.*metadata has been deprecated.*:DeprecationWarning
    ignore:^numpy\.ufunc size changed.*:RuntimeWarning
    ignore:.*invalid escape sequence.*:
    ignore:.*an integer is required \(got type.*:DeprecationWarning
    always::ResourceWarning
    """  # noqa: E501
    for warning_line in warning_lines.split('\n'):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith('#'):
            config.addinivalue_line('filterwarnings', warning_line)


@pytest.fixture(scope='session')
def traits_test():
    """Context to raise errors in trait handlers."""
    from traits.api import push_exception_handler
    push_exception_handler(reraise_exceptions=True)
    yield
    push_exception_handler(reraise_exceptions=False)


@pytest.fixture(scope='function', params=[testing._pytest_param()])
def subjects_dir_tmp(tmpdir):
    """Copy MNE-testing-data subjects_dir to a temp dir for manipulation."""
    for key in ('sample', 'fsaverage'):
        shutil.copytree(op.join(subjects_dir, key), str(tmpdir.join(key)))
    return str(tmpdir)
