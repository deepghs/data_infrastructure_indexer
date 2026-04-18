from typing import Optional, Union

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


__all__ = ['parse_duration_to_seconds']
