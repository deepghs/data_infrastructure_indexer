import os

from ditk import logging

from .eidx import sync

if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository=os.environ['REMOTE_REPOSITORY_E621_PUBLIC_4M'],
        exist_repo=os.environ['REMOTE_REPOSITORY_E621_4M_IDX'],
    )
