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
    resp = source.session.get('https://www.zerochan.net/xbotcheck-image.svg')
    resp.raise_for_status()
    assert source._check_session(), 'Session check failed'
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

# if __name__ == '__main__':
#     s = get_session()
#     resp = s.get('https://www.zerochan.net/?json=1')
#     print(resp)
