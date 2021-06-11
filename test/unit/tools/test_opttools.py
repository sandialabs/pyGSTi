import numbers
from collections import defaultdict
from time import sleep
from unittest import mock

from pygsti.tools import opttools as opt
from ..util import BaseCase


class TestTimedBlock(BaseCase):
    def test_stdout_output(self):
        with mock.patch('sys.stdout') as mock_out:
            with opt.timed_block('time'):
                pass
            call_count = mock_out.write.call_count

        self.assertEqual(call_count, 2, "unexpected number of lines written")

    def test_pre_message(self):
        preMessage = "this is a pre-message!"
        with mock.patch('sys.stdout') as mock_out:
            with opt.timed_block('time', pre_message=preMessage):
                pass
            call_count = mock_out.write.call_count
            call_args_list = mock_out.write.call_args_list

        self.assertEqual(call_count, 4, "unexpected number of lines written")
        self.assertIn(preMessage, call_args_list[0][0], "pre-message not written first")

    def test_time_dict_storage(self):
        timeDict = {}
        with opt.timed_block('time', timeDict):
            pass

        self.assertIn('time', timeDict)
        self.assertTrue(isinstance(timeDict['time'], numbers.Number), "time is not numeric")

        timeDict = defaultdict(list)
        with opt.timed_block('time', timeDict):
            pass

        self.assertIn('time', timeDict)
        self.assertTrue(isinstance(timeDict['time'], list), "time was not appended to list")
        self.assertTrue(isinstance(timeDict['time'][0], numbers.Number), "time is not numeric")

    def test_timer(self):

        duration = 0.03
        timeDict = {}
        with opt.timed_block('time', timeDict):
            sleep(duration)

        self.assertGreaterEqual(timeDict['time'], duration)
        tolerance = 0.015  # this should deliberately be large, for repeatability
        self.assertLessEqual(timeDict['time'], duration + tolerance, "timed block result is greater than {} seconds off".format(tolerance))


class TestTimeHash(BaseCase):
    def test_properties(self):
        value1 = opt.time_hash()
        self.assertTrue(isinstance(value1, str), "time hash is not a string")

        value2 = opt.time_hash()
        self.assertNotEqual(value1, value2, "different time hashes are equal")
