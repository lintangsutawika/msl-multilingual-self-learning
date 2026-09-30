"""LeetCode page HTML -> the plain-text description the model sees."""
import unittest

from src.benchmarks.leetcode.hf_dataset.description import html_to_text


class DescriptionTests(unittest.TestCase):
    def text(self, html):
        return html_to_text(html).text.strip()

    def test_exponents_and_subscripts_are_kept(self):
        self.assertEqual(self.text("<code>-10<sup>9</sup> &lt;= board[i][j] &lt;= 10<sup>9</sup></code>"),
                         "-10^9 <= board[i][j] <= 10^9")
        self.assertEqual(self.text("2<sup>n - 1</sup> and 10<sup>-5</sup>"), "2^{n - 1} and 10^-5")
        self.assertEqual(self.text("queries[i] = [l<sub>i</sub>, r<sub>i</sub>], x<sub>i+1</sub>, y<sub>i,</sub> z"),
                         "queries[i] = [l_i, r_i], x_{i+1}, y_i, z")

    def test_lists_tables_and_examples(self):
        self.assertEqual(self.text("<ol><li>a<ul><li>b</li></ul></li><li>c</li></ol>"), "1. a\n  - b\n2. c")
        self.assertEqual(self.text("<table>\n<tr>\n <th>Symbol</th> <th>Value</th></tr><tr><td>I</td><td>1</td></tr></table>"),
                         "Symbol | Value\nI | 1")
        block = ('<div class="example-block"><p><strong>Input:</strong> <span class="example-io">n = 1</span></p>'
                 '<p><strong>Output:</strong> <span class="example-io">2</span></p></div>')
        self.assertEqual(self.text(block), "Input: n = 1\nOutput: 2")

    def test_media_and_page_residue_are_removed(self):
        page = html_to_text('<p>See<img src="a.png" /> this​.</p><style>p{}</style>'
                            '<div class="simple-translate-panel"><p>junk</p></div><video><source src="v.mp4"></video>')
        self.assertEqual(page.text.strip(), "See this.")
        self.assertEqual((page.images, page.videos), (["a.png"], 1))


if __name__ == "__main__":
    unittest.main()
