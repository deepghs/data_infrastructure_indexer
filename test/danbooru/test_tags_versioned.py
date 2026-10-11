import importlib
import io
import sys

import pytest

from inf.danbooru.tags_versioned import generate_calendar_markdown


@pytest.mark.unittest
class TestGenerateCalendarMarkdown:
    def test_renders_without_dtype_errors(self):
        # pandas 3 refuses to write a string into an int64 column, which is what the empty and
        # linked cells are, so this used to raise TypeError on every run.
        buf = io.StringIO()
        generate_calendar_markdown(buf, (2020, 1), lambda ymd: f'https://x/{ymd[0]}-{ymd[1]}-{ymd[2]}',
                                   last_month_count=3)
        text = buf.getvalue()
        assert text.count('## ') == 3
        assert '| Mon' in text
        assert '[1](https://x/' in text

    def test_days_without_a_url_are_plain_numbers(self):
        buf = io.StringIO()
        generate_calendar_markdown(buf, (2020, 1), lambda ymd: None, last_month_count=1)
        text = buf.getvalue()
        assert '](' not in text
        assert ' 1 ' in text

    def test_importing_the_module_does_not_load_waifuc(self):
        # waifuc installs onnxruntime with pip on first import, which races when several test
        # workers import it at once, so the module must only load it when a sync actually runs.
        sys.modules.pop('inf.danbooru.tags_versioned', None)
        sys.modules.pop('waifuc', None)
        importlib.import_module('inf.danbooru.tags_versioned')
        assert 'waifuc' not in sys.modules
