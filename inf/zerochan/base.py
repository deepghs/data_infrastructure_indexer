import os
from functools import lru_cache
from typing import Optional, Tuple

import requests
from hfutils.utils import get_random_ua
from waifuc.source import ZerochanSource

_ROOT = 'https://www.zerochan.net'


class ZerochanFuckedUp(Exception):
    pass


def _get_session_for_zerochan(auth: Optional[Tuple[str, str]] = None) -> requests.Session:
    if auth:
        username, password = auth
    else:
        username, password = None, None
    source = ZerochanSource(
        '1girk',
        user_agent=get_random_ua(),
        username=username,
        password=password
    )
    source.session.headers.update({
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
    })
    if auth:
        source._auth()

    session = source.session
    return session


@lru_cache()
def get_session():
    return _get_session_for_zerochan((
        os.environ['ZEROCHAN_USERNAME'],
        os.environ['ZEROCHAN_PASSWORD'],
    ))
