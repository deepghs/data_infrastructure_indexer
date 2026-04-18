import subprocess
import sys

import pytest

_WORKFLOW_ENTRYPOINT_MODULES = [
    'inf.danbooru.index_n',
    'inf.danbooru.dbsquash_n',
    'inf.danbooru.tags_versioned',
    'inf.e621.index',
    'inf.e621.eidx',
    'inf.e621.dbsquash',
    'inf.e621.tags',
    'inf.gelbooru.index_prev',
    'inf.gelbooru.dbsquash',
    'inf.gelbooru.tags',
    'inf.konachan.index',
    'inf.konachan.dbsquash',
    'inf.konachan.tags',
    'inf.rule34.index',
    'inf.rule34.tags',
    'inf.yande.index',
    'inf.yande.dbsquash',
    'inf.yande.tags',
    'inf.zerochan.index',
    'inf.zerochan.dbsquash',
]


@pytest.mark.unittest
@pytest.mark.parametrize('module_name', _WORKFLOW_ENTRYPOINT_MODULES)
def test_workflow_entrypoints_support_cli_help(module_name):
    result = subprocess.run(
        [sys.executable, '-m', module_name, '--help'],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert 'usage:' in result.stdout.lower()
    assert 'default:' in result.stdout.lower()
