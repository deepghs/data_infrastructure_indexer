from typing import Optional, Union

import click
from pytimeparse import parse as _parse_duration_text

_NONE_TOKENS = {'', 'none', 'null', 'nil'}
_UNLIMITED_TOKENS = {'unlimited', 'infinite', 'infinity', 'inf'}


def parse_duration_to_seconds(value: Union[str, int, float, None], allow_none: bool = False) -> Optional[float]:
    if value is None:
        if allow_none:
            return None
        raise ValueError('Duration value should not be empty.')

    if isinstance(value, (int, float)):
        return value

    text = str(value).strip()
    lower_text = text.lower()
    if allow_none and lower_text in _NONE_TOKENS | _UNLIMITED_TOKENS:
        return None

    seconds = _parse_duration_text(text)
    if seconds is None:
        raise ValueError(f'Invalid duration value - {value!r}.')
    return seconds


class DurationType(click.ParamType):
    name = 'duration'

    def __init__(self, allow_none: bool = False, cast=None):
        self.allow_none = allow_none
        self.cast = cast

    def convert(self, value, param, ctx):
        try:
            seconds = parse_duration_to_seconds(value, allow_none=self.allow_none)
        except ValueError as err:
            self.fail(str(err), param, ctx)

        if seconds is None:
            return None
        if self.cast is not None:
            return self.cast(seconds)
        return seconds


def duration_type(allow_none: bool = False, cast=None) -> DurationType:
    return DurationType(allow_none=allow_none, cast=cast)


__all__ = ['DurationType', 'duration_type', 'parse_duration_to_seconds']
