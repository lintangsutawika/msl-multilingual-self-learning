from __future__ import annotations

from html.parser import HTMLParser
import re
from typing import Any


class _StatementExampleParser(HTMLParser):
    """Read outputs from both current and legacy LeetCode statement markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._pre_depth = 0
        self._pre_text: list[str] = []
        self.pre_blocks: list[str] = []

        self.outputs: list[str] = []
        self._strong_depth = 0
        self._strong_text: list[str] = []
        self._output_text: list[str] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        tag = tag.lower()

        if tag == "pre":
            if self._pre_depth == 0:
                self._pre_text = []
            self._pre_depth += 1
        elif self._pre_depth and tag == "br":
            self._pre_text.append("\n")

        if tag in {"strong", "b"}:
            if self._strong_depth == 0:
                self._strong_text = []
            self._strong_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in {"strong", "b"} and self._strong_depth:
            self._strong_depth -= 1
            if self._strong_depth == 0:
                label = "".join(self._strong_text).strip().rstrip(":").lower()
                if label == "output" and not self._pre_depth:
                    self._output_text = []
                self._strong_text = []

        if tag == "p" and self._output_text is not None:
            output = "".join(self._output_text).strip()
            if output:
                self.outputs.append(output)
            self._output_text = None

        if tag != "pre" or not self._pre_depth:
            return

        self._pre_depth -= 1
        if self._pre_depth == 0:
            self.pre_blocks.append("".join(self._pre_text))
            self._pre_text = []

    def handle_data(self, data: str) -> None:
        if self._pre_depth:
            self._pre_text.append(data)
        if self._strong_depth:
            self._strong_text.append(data)
        if self._output_text is not None:
            self._output_text.append(data)


_OUTPUT_RE = re.compile(
    r"(?:^|\n)\s*Output\s*:\s*(?P<output>.*?)"
    r"(?=\n\s*(?:Explanation|Note)\s*:|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_example_outputs(content_html: str) -> list[str]:
    """Extract the displayed Output value from each statement example block."""
    parser = _StatementExampleParser()
    parser.feed(content_html)
    parser.close()

    if parser.outputs:
        return parser.outputs

    outputs: list[str] = []

    for block in parser.pre_blocks:
        match = _OUTPUT_RE.search(block)
        if match is None:
            continue

        output = match.group("output").strip()
        if output:
            outputs.append(output)

    return outputs


def build_public_test_cases(
    example_testcase_list: list[str] | None,
    content_html: str | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pair official raw example inputs with outputs shown in the statement.

    LeetCode exposes these in two different GraphQL fields. Keeping unmatched
    values makes schema drift auditable instead of silently shifting pairs.
    """
    inputs = example_testcase_list or []
    outputs = extract_example_outputs(content_html or "")
    count = max(len(inputs), len(outputs))

    cases = [
        {
            "input": inputs[index] if index < len(inputs) else None,
            "output": outputs[index] if index < len(outputs) else None,
        }
        for index in range(count)
    ]

    stats = {
        "input_count": len(inputs),
        "output_count": len(outputs),
        "paired_count": min(len(inputs), len(outputs)),
    }

    return cases, stats
