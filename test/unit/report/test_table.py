from ..util import BaseCase

from pygsti.report import table


# Refactored from test.test_packages.reportb.testEBFormatters
# XXX what unit does this cover?
class EBFormatterBase:
    def _test_render(self, n, expected_fmt, col_header, precision):
        expected = expected_fmt.format(n)
        ebtable = table.ReportTable([col_header], self.formatters)
        rendered = ebtable.render(self.target, precision=precision)
        # TODO Instead of strict comparison, only assert neccessary elements
        self.assertEqual(rendered[self.target], expected)

    def test_render_EB_formatter(self):
        n = 1.819743  # Six digits after decimal
        self._test_render(n, self.expected_eb_fmt, (n, n), 6)

        n = 1.82  # Two digits after decimal
        self._test_render(n, self.expected_eb_fmt, (n, n), 2)

    def test_render_EB_formatter_none(self):
        n = 1.819743  # Six digits after decimal
        self._test_render(n, self.expected_no_eb_fmt, (n, None), 6)

        n = 1.82  # Two digits after decimal
        self._test_render(n, self.expected_no_eb_fmt, (n, None), 2)


class NormalEBFormatterTester(EBFormatterBase, BaseCase):
    formatters = ['Normal']
    target = 'latex'
    expected_eb_fmt = (
        '\\begin{{tabular}}[l]{{|c|}}\n\\hline\n$ \\begin{{array}}{{c}} {0} \\\\ '
        '\\pm {0} \\end{{array}} $ \\\\ \\hline\n\\end{{tabular}}\n'
    )
    expected_no_eb_fmt = '\\begin{{tabular}}[l]{{|c|}}\n\\hline\n{0} \\\\ \\hline\n\\end{{tabular}}\n'


class PiLaTeXEBFormatterTester(EBFormatterBase, BaseCase):
    formatters = ['Pi']
    target = 'latex'
    expected_eb_fmt = (
        '\\begin{{tabular}}[l]{{|c|}}\n\\hline\n$ \\begin{{array}}{{c}}({0} \\\\ '
        '\\pm {0})\\pi \\end{{array}} $ \\\\ \\hline\n\\end{{tabular}}\n'
    )
    expected_no_eb_fmt = '\\begin{{tabular}}[l]{{|c|}}\n\\hline\n{0}$\\pi$ \\\\ \\hline\n\\end{{tabular}}\n'


class PiHTMLEBFormatterTester(EBFormatterBase, BaseCase):
    formatters = ['Pi']
    target = 'html'
    expected_eb_fmt = (
        '<table><thead><tr><th> <span title="({0}, {0})">{0}&pi; <span class="errorbar">&plusmn; '
        '{0}</span>&pi;</span> </th></tr></thead><tbody></tbody></table>'
    )
    expected_no_eb_fmt = (
        '<table><thead><tr><th> <span title="({0}, None)">{0}&pi;</span> </th></tr></thead>'
        '<tbody></tbody></table>'
    )
