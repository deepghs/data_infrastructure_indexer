import click
import pytest

from inf.utils.duration import duration_type, parse_duration_to_seconds


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
def test_duration_type_parses_humanized_text():
    @click.command()
    @click.option('--value', type=duration_type())
    def cli(value):
        return value

    assert cli.main(args=['--value', '5h'], standalone_mode=False) == 5 * 60 * 60


@pytest.mark.unittest
def test_duration_type_supports_unlimited_when_allow_none():
    @click.command()
    @click.option('--value', type=duration_type(allow_none=True))
    def cli(value):
        return value

    assert cli.main(args=['--value', 'unlimited'], standalone_mode=False) is None


@pytest.mark.unittest
def test_duration_type_rejects_invalid_values():
    @click.command()
    @click.option('--value', type=duration_type())
    def cli(value):
        return value

    with pytest.raises(click.BadParameter):
        cli.main(args=['--value', 'bad-duration'], standalone_mode=False)
