"""LeetCode page HTML -> the plain-text problem description the model sees.

newfacade/LeetCodeDataset's descriptions were extracted without markup, so
10<sup>9</sup> became "109", l<sub>i</sub> became "li" and list numbering was
lost. This converter keeps that meaning in plain text:

    10<sup>9</sup>      -> 10^9          2<sup>n-1</sup> -> 2^{n-1}
    l<sub>i</sub>       -> l_i           x<sub>i+1</sub> -> x_{i+1}
    <ol><li>            -> "1. "         <ul><li>        -> "- "
    <table>             -> "a | b" rows  <br>            -> newline

Images and videos are dropped (the prompt is text only) and counted, so a
problem whose text depends on a figure can be reviewed. <style> blocks and
browser-extension residue pasted into some pages are removed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

BLOCK_TAGS = {"p", "div", "pre", "ul", "ol", "li", "table", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}
# Subtrees whose text is never part of the problem.
SKIP_TAGS = {"style", "script", "video", "source"}
SKIP_CLASS_PREFIXES = ("simple-translate", "gtx-trans", "darkreader")
VOID_TAGS = {"br", "img", "meta", "source", "hr", "input", "wbr"}
SIMPLE_SCRIPT = re.compile(r"-?[A-Za-z0-9]+(\.\d+)?|\(.*\)|\{.*\}")
# Punctuation or spaces caught inside the tag ("x<sub>i,</sub>"): script the token, keep the rest as text.
SCRIPT_THEN_TEXT = re.compile(r"(-?[A-Za-z0-9]+)([,;:.]?\s*)")
INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))


@dataclass
class Description:
    text: str
    images: list[str] = field(default_factory=list)
    videos: int = 0
    tables: int = 0


class _Converter(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.stack: list[str] = []          # open tags, for skipping subtrees
        self.skip_depth = 0                 # >0 while inside a skipped subtree
        self.pre = 0
        self.lists: list[list] = []         # [kind, next number]
        self.script: list[tuple[str, int]] = []  # open sup/sub with their out position
        self.row: list[str] | None = None
        self.cell_start: int | None = None
        self.example_blocks: list[bool] = []  # per open div: is it an example-block?
        self.images: list[str] = []
        self.videos = 0
        self.tables = 0

    # -- helpers --
    def _newline(self, blank: bool = False) -> None:
        text = "".join(self.out)
        stripped = text.rstrip(" \t")
        if not stripped:
            self.out = []
            return
        want = "\n\n" if blank else "\n"
        tail = len(stripped) - len(stripped.rstrip("\n"))
        self.out = [stripped + want[min(tail, len(want)):]] if tail < len(want) else [stripped]

    def _skipping(self) -> bool:
        return self.skip_depth > 0

    # -- parser callbacks --
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if self._skipping():
            if tag not in VOID_TAGS:
                self.skip_depth += 1
            return
        cls = attr.get("class") or ""
        if tag in SKIP_TAGS or any(c.startswith(SKIP_CLASS_PREFIXES) for c in cls.split()):
            if tag == "video":
                self.videos += 1
            if tag not in VOID_TAGS:
                self.skip_depth = 1
            return
        if tag == "img":
            self.images.append(attr.get("src") or "")
            return
        if tag == "br":
            self._newline()
            return
        if tag in ("ul", "ol"):
            self._newline()
            self.lists.append([tag, 1])
        elif tag == "li":
            self._newline()
            indent = "  " * (len(self.lists) - 1)
            if self.lists and self.lists[-1][0] == "ol":
                self.out.append(f"{indent}{self.lists[-1][1]}. ")
                self.lists[-1][1] += 1
            else:
                self.out.append(f"{indent}- ")
        elif tag == "pre":
            self._newline(blank=True)
            self.pre += 1
        elif tag == "table":
            self._newline(blank=True)
            self.tables += 1
        elif tag == "tr":
            self._newline()
            self.row = []
        elif tag in ("td", "th"):
            self.cell_start = len("".join(self.out))
        elif tag in ("sup", "sub"):
            self.script.append((tag, len("".join(self.out))))
        elif tag in BLOCK_TAGS:
            if tag == "div":
                self.example_blocks.append("example-block" in cls.split())
            self._newline(blank=self._paragraph(tag))

    def _paragraph(self, tag: str) -> bool:
        """Whether this block is separated by a blank line (paragraphs, not list items or example lines)."""
        return tag in ("p", "div") and not self.lists and not any(self.example_blocks)

    def handle_endtag(self, tag: str) -> None:
        if self._skipping():
            if tag not in VOID_TAGS:
                self.skip_depth -= 1
            return
        if tag in ("sup", "sub") and self.script and self.script[-1][0] == tag:
            _, start = self.script.pop()
            text = "".join(self.out)
            inner = " ".join(text[start:].split())
            mark = "^" if tag == "sup" else "_"
            split = SCRIPT_THEN_TEXT.fullmatch(inner)
            if not inner:
                wrapped = ""
            elif SIMPLE_SCRIPT.fullmatch(inner):
                wrapped = mark + inner
            elif split:
                wrapped = mark + split.group(1) + split.group(2)
            else:
                wrapped = mark + "{" + inner + "}"
            self.out = [text[:start] + wrapped]
        elif tag in ("td", "th") and self.row is not None and self.cell_start is not None:
            text = "".join(self.out)
            self.row.append(" ".join(text[self.cell_start:].split()))
            self.out = [text[: self.cell_start]]
            self.cell_start = None
        elif tag == "tr" and self.row is not None:
            self.out.append(" | ".join(self.row))
            self.row = None
            self._newline()
        elif tag in ("ul", "ol"):
            if self.lists:
                self.lists.pop()
            self._newline(blank=not self.lists)
        elif tag == "pre":
            self.pre = max(0, self.pre - 1)
            self._newline(blank=True)
        elif tag == "table":
            self._newline(blank=True)
        elif tag in BLOCK_TAGS and tag != "li":
            if tag == "div" and self.example_blocks:
                self.example_blocks.pop()
            self._newline(blank=self._paragraph(tag) or tag == "div")

    def handle_data(self, data: str) -> None:
        if self._skipping():
            return
        data = data.translate(INVISIBLE).replace("\xa0", " ").replace("−", "-")
        if self.pre:
            self.out.append(data)
            return
        if self.row is not None and self.cell_start is None:
            return  # whitespace between table cells
        data = re.sub(r"\s+", " ", data)
        if not self.out or "".join(self.out).endswith("\n"):
            data = data.lstrip()
        self.out.append(data)


def _tidy(text: str) -> str:
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # "&nbsp;" paragraphs leave lines holding only spaces; the split above emptied them.
    return text.strip() + "\n"


def html_to_text(content: str) -> Description:
    """Convert a LeetCode page's content HTML to plain text (see module docstring)."""
    converter = _Converter()
    converter.feed(content)
    converter.close()
    return Description(
        text=_tidy("".join(converter.out)),
        images=converter.images,
        videos=converter.videos,
        tables=converter.tables,
    )
