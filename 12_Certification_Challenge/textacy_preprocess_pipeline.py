"""
Textacy preprocessing pipeline: Markdown/HTML → plain text.
Keeps emojis; strips headings, links, URLs, HTML; normalizes bullets, quotes, unicode, whitespace.
"""

import re
from functools import partial
from textacy import preprocessing


def strip_markdown(text: str) -> str:
    """Remove common Markdown syntax; keep inner text and emojis."""
    # Headings: ## Title -> Title
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Inline code: `code` -> code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Bold/italic: **bold** or *italic* -> bold / italic
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
    # Link markup: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Collapse many newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def make_plain_text_pipeline(
    *,
    url_repl: str = "",
    unicode_form: str = "NFC",
):
    """
    Build a textacy pipeline: Markdown/HTML → plain text (emojis kept).

    Args:
        url_repl: Replacement for URLs (default ""). Use "_URL_" to keep a placeholder.
        unicode_form: Unicode normalization form ("NFC", "NFD", "NFKC", "NFKD"). Default "NFC".

    Returns:
        A callable that takes a string and returns preprocessed plain text.
    """
    return preprocessing.make_pipeline(
        strip_markdown,
        preprocessing.remove.html_tags,
        partial(preprocessing.replace.urls, repl=url_repl),
        preprocessing.normalize.bullet_points,
        preprocessing.normalize.quotation_marks,
        partial(preprocessing.normalize.unicode, form=unicode_form),
        preprocessing.normalize.whitespace,
    )


# Default pipeline instance (URLs removed)
preprocess_to_plain_text = make_plain_text_pipeline()


if __name__ == "__main__":
    example = '''## textacy: NLP, before and after spaCy

`textacy` is a Python library for NLP tasks, built on spaCy.

[![build](https://img.shields.io/travis/chartbeat-labs/textacy.svg)](https://travis-ci.org/chartbeat-labs/textacy)

### features

- Access and extend spaCy's core functionality
- Clean, normalize, and explore raw text

### maintainer

Howdy, y'all. 👋

- Burton DeWilde (<burtdewilde@gmail.com>)
'''
    out = preprocess_to_plain_text(example)
    print("--- preprocessed ---")
    print(out)
    print("--- end ---")
