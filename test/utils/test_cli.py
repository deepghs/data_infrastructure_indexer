from typing import Optional

import pytest

from inf.utils.cli import build_cli_parser, env_default, resolve_cli_arguments, run_callable_from_cli
from inf.utils.duration import parse_duration_to_seconds


@pytest.mark.unittest
def test_parse_duration_to_seconds_supports_humanized_text():
    assert parse_duration_to_seconds('5h') == 5 * 60 * 60
    assert parse_duration_to_seconds('48min') == 48 * 60
    assert parse_duration_to_seconds('2d') == 2 * 24 * 60 * 60


@pytest.mark.unittest
def test_parse_duration_to_seconds_supports_none_and_unlimited_tokens():
    assert parse_duration_to_seconds('none', allow_none=True) is None
    assert parse_duration_to_seconds('unlimited', allow_none=True) is None


@pytest.mark.unittest
def test_resolve_cli_arguments_uses_env_defaults_and_duration_parsing(monkeypatch):
    def sync(repository: str, max_time_limit: Optional[float] = 300.0,
             sync_mode: bool = False, site_username: Optional[str] = None,
             start_from_id: Optional[int] = None):
        raise NotImplementedError

    monkeypatch.setenv('TEST_REPOSITORY', 'demo/repository')

    values = resolve_cli_arguments(
        sync,
        defaults={
            'repository': env_default('TEST_REPOSITORY'),
            'site_username': env_default('TEST_USERNAME', default=None),
        },
        argv=[
            '--max-time-limit', '5h',
            '--sync-mode', 'true',
            '--site-username', 'none',
            '--start-from-id', '42',
        ],
    )

    assert values == {
        'repository': 'demo/repository',
        'max_time_limit': 5 * 60 * 60,
        'sync_mode': True,
        'site_username': None,
        'start_from_id': 42,
    }


@pytest.mark.unittest
def test_run_callable_from_cli_supports_unlimited_optional_durations():
    captured = {}

    def sync(access_interval: Optional[float] = 60.0):
        captured['access_interval'] = access_interval
        return access_interval

    retval = run_callable_from_cli(sync, argv=['--access-interval', 'unlimited'])

    assert retval is None
    assert captured == {
        'access_interval': None,
    }


@pytest.mark.unittest
def test_non_optional_durations_do_not_accept_none_tokens():
    def sync(upload_time_span: float = 30.0):
        raise NotImplementedError

    parser = build_cli_parser(sync)
    help_text = ' '.join(parser.format_help().lower().split())
    assert 'use none to clear the value' not in help_text

    with pytest.raises(SystemExit):
        resolve_cli_arguments(sync, argv=['--upload-time-span', 'none'])


@pytest.mark.unittest
def test_build_cli_parser_help_contains_usage_type_and_defaults():
    def sync(repository: str, max_time_limit: Optional[float] = 300.0,
             sync_mode: bool = False, site_username: Optional[str] = None):
        raise NotImplementedError

    parser = build_cli_parser(sync, defaults={
        'repository': env_default('TEST_REPOSITORY'),
        'site_username': env_default('TEST_USERNAME', default=None),
    })
    help_text = ' '.join(parser.format_help().lower().split())

    assert '--repository' in help_text
    assert 'target hugging face dataset repository to read from and write to' in help_text
    assert 'stop the sync after this total runtime' in help_text
    assert 'duration in seconds or text like 5h, 48min, 2d' in help_text
    assert 'boolean: true/false/yes/no/on/off/1/0' in help_text
    assert 'default: env test_repository' in help_text
    assert 'default: env test_username or none' in help_text
