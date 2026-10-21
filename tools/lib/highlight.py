#!/usr/bin/env python3
"""Batch syntax highlighting for the paper build.

Reads a JSON array of {code, lang} objects from stdin and writes a JSON array
of Pygments HTML fragments (without the surrounding <pre>).
"""
import json
import sys

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound

ALIASES = {
    "py": "python",
    "python3": "python",
    "sh": "bash",
    "shell": "bash",
    "console": "bash",
    "text": "text",
    "plaintext": "text",
    "json": "json",
    "js": "javascript",
    "ts": "typescript",
    "yml": "yaml",
    "md": "markdown",
}


def lexer_for(lang: str):
    lang = (lang or "").strip().lower()
    lang = ALIASES.get(lang, lang)
    if not lang:
        return get_lexer_by_name("text")
    try:
        return get_lexer_by_name(lang, stripnl=False)
    except ClassNotFound:
        return get_lexer_by_name("text")


def main() -> None:
    items = json.load(sys.stdin)
    formatter = HtmlFormatter(nowrap=True)
    out = []
    for item in items:
        code = item.get("code", "")
        try:
            out.append(highlight(code, lexer_for(item.get("lang", "")), formatter).rstrip("\n"))
        except Exception:
            # Never fail the build because one snippet cannot be highlighted.
            from html import escape

            out.append(escape(code))
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
