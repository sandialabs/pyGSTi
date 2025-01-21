from pygsti.report.textblock import ReportText
from ..util import BaseCase


class TextBlockTester(BaseCase):
    def test_render(self):
        raw = "Hello"
        text = ReportText(raw)
        render = text.render('html')
        self.assertEqual(render['html'], raw)

    def test_to_string(self):
        raw = "Hello"
        text = ReportText(raw)
        s = str(text)
        # TODO assert correctness

    def test_render_raises_on_unknown_form(self):
        with self.assertRaises(ValueError):
            # XXX shouldn't this be validated on initialization?  EGN: yeah, that would make sense to me.
            text = ReportText("Hello", 'foobar')
            text.render('html')
