import pygsti.report.formatters as fmt
from pygsti.report.latex import value as latex_value
from pygsti.report.table import ReportTable
from ..util import BaseCase


def render_pair(heading, formatter, formattype='latex', **kwargs):
    # TODO render directly instead of through ReportTable
    table = ReportTable([heading], [formatter])
    return table.render(formattype, **kwargs)[formattype]


class FormatterTester(BaseCase):
    def setUp(self):
        super(FormatterTester, self).setUp()

    def test_none_formatter(self):
        fmt.format_dict['BadFormat'] = {
            'latex': lambda l, s: None,
            'html': lambda l, s: None,
            'text': lambda l, s: None,
            'ppt': lambda l, s: None,
        }

        with self.assertRaises(AssertionError):
            render_pair('some_heading', 'BadFormat')

    def test_unformatted_none(self):
        with self.assertRaises(ValueError):
            render_pair(None, None)

    def test_unformatted(self):
        render_pair('some heading', None)
        # TODO assert correctness

    def test_string_return(self):
        self.assertEqual(
            render_pair('Ec', 'Effect', 'html'),
            ('<table><thead><tr><th> <span title="Ec">E<sub>C</sub></span> </th></tr></thead>'
             '<tbody></tbody></table>')
        )

    def test_string_replace(self):
        self.assertEqual(
            render_pair('rho0', 'Rho', 'html'),
            ('<table><thead><tr><th> <span title="rho0">&rho;<sub>0</sub></span> </th></tr></thead>'
             '<tbody></tbody></table>')
        )

    def test_conversion_formatters(self):
        self.assertEqual(fmt.convert_html('|<STAR>', {}), ' &#9733;')
        self.assertEqual(
            fmt.convert_latex('%% # half-width 1/2 Diamond Check <STAR>', {}),
            '$\\%\\% \# $\\nicefrac{1}{2}$-width $\\nicefrac{1}{2}$ $\\Diamond$ \\checkmark \\bigstar$'
        )
        self.assertEqual(fmt.convert_latex('x|y', {}), '\\begin{tabular}{c}x\\\\y\\end{tabular}')

    def test_value_fns(self):
        specs = {'precision': 2, 'sciprecision': 2, 'polarprecision': 2, 'complex_as_polar': True}
        self.assertEqual(latex_value("Hello", specs), "Hello")
        latex_value({"Weird type": "to get value of!"}, specs)
        #More variants?


# Refactored from test.test_packages.reportb.testEBFormatters
# TODO assert correct elements instead of directly comparing rendered strings
class EBFormatterBase(object):
    def test_render_EB_formatter(self):
        n = 1.819743  # Six digits after decimal
        render = render_pair((n, n), self.formatter, formattype=self.target, precision=6)
        self.assertEqual(render, self.expected_eb_fmt.format(n))

        n = 1.82  # Two digits after decimal
        render = render_pair((n, n), self.formatter, formattype=self.target, precision=2)
        self.assertEqual(render, self.expected_eb_fmt.format(n))

    def test_render_EB_formatter_none(self):
        n = 1.819743  # Six digits after decimal
        render = render_pair((n, None), self.formatter, formattype=self.target, precision=6)
        self.assertEqual(render, self.expected_no_eb_fmt.format(n))

        n = 1.82  # Two digits after decimal
        render = render_pair((n, None), self.formatter, formattype=self.target, precision=2)
        self.assertEqual(render, self.expected_no_eb_fmt.format(n))


class NormalEBFormatterTester(EBFormatterBase, BaseCase):
    formatter = 'Normal'
    target = 'latex'
    expected_eb_fmt = (
        '\\begin{{tabular}}[l]{{|c|}}\n\\hline\n$ \\begin{{array}}{{c}} {0} \\\\ '
        '\\pm {0} \\end{{array}} $ \\\\ \\hline\n\\end{{tabular}}\n'
    )
    expected_no_eb_fmt = '\\begin{{tabular}}[l]{{|c|}}\n\\hline\n{0} \\\\ \\hline\n\\end{{tabular}}\n'


class PiLaTeXEBFormatterTester(EBFormatterBase, BaseCase):
    formatter = 'Pi'
    target = 'latex'
    expected_eb_fmt = (
        '\\begin{{tabular}}[l]{{|c|}}\n\\hline\n$ \\begin{{array}}{{c}}({0} \\\\ '
        '\\pm {0})\\pi \\end{{array}} $ \\\\ \\hline\n\\end{{tabular}}\n'
    )
    expected_no_eb_fmt = '\\begin{{tabular}}[l]{{|c|}}\n\\hline\n{0}$\\pi$ \\\\ \\hline\n\\end{{tabular}}\n'


class PiHTMLEBFormatterTester(EBFormatterBase, BaseCase):
    formatter = 'Pi'
    target = 'html'
    expected_eb_fmt = (
        '<table><thead><tr><th> <span title="({0}, {0})">{0}&pi; <span class="errorbar">&plusmn; '
        '{0}</span>&pi;</span> </th></tr></thead><tbody></tbody></table>'
    )
    expected_no_eb_fmt = (
        '<table><thead><tr><th> <span title="({0}, None)">{0}&pi;</span> </th></tr></thead>'
        '<tbody></tbody></table>'
    )


class PrecisionFormatterBase(object):
    expected_LaTeX_fmt = '\\begin{{tabular}}[l]{{|c|}}\n\\hline\n{0} \\\\ \\hline\n\\end{{tabular}}\n'
    expected_HTML_fmt = '<table><thead><tr><th> <span title="{0}">{1}</span> </th></tr></thead><tbody></tbody></table>'

    def setUp(self):
        super(PrecisionFormatterBase, self).setUp()
        n = 1.819743  # Six digits after decimal
        self.table = ReportTable([n], ['Normal'])
        self.options = dict(
            precision=self.precision,
            polarprecision=3
        )
        format_n = "{{:.{}f}}".format(self.precision).format(n)
        self.expected_LaTeX = self.expected_LaTeX_fmt.format(format_n)
        self.expected_HTML = self.expected_HTML_fmt.format(n, format_n)

    def test_render_precision_LaTeX(self):
        self.assertEqual(self.table.render('latex', **self.options)['latex'], self.expected_LaTeX)

    def test_render_precision_HTML(self):
        self.assertEqual(self.table.render('html', **self.options)['html'], self.expected_HTML)


class HighPrecisionFormatterTester(PrecisionFormatterBase, BaseCase):
    precision = 6


class LowPrecisionFormatterTester(PrecisionFormatterBase, BaseCase):
    precision = 2
