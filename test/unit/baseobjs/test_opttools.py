from time import sleep
import numbers
from collections import defaultdict

from ..util import BaseCase, mock

from pygsti.baseobjs import opttools as opt


class CacheByHashedArgsTester(BaseCase):
    def test_args_are_cached(self):
        mock_call = mock.MagicMock(return_value=42)

        @opt.cache_by_hashed_args
        def cached_call(*args):
            return mock_call(*args)

        value = cached_call(8, 6, 7, 5)
        self.assertEqual(value, mock_call.return_value, "function returns unexpected value")
        self.assertEqual(cached_call(8, 6, 7, 5), mock_call.return_value, "invalid value fetched from cache")
        mock_call.assert_called_once_with(8, 6, 7, 5)

        mock_call.return_value = 777
        value = cached_call(3, 0, 9)
        self.assertEqual(value, mock_call.return_value, "function returns unexpected value on subsequent call")
        self.assertEqual(cached_call(3, 0, 9), mock_call.return_value,
                         "invalid value fetched from cache on subsequent call")
        self.assertEqual(cached_call(8, 6, 7, 5), 42, "invalid value fetched from cache on subsequent call")
        self.assertEqual(mock_call.call_count, 2, "return value was not cached on subsequent call")
        mock_call.assert_any_call(3, 0, 9)

    def test_kwargs_not_cached(self):
        mock_call = mock.MagicMock(return_value=42)

        @opt.cache_by_hashed_args
        def cached_call(a, b=1):
            return mock_call(a, b)

        value = cached_call(2, b=2)
        self.assertEqual(value, mock_call.return_value, "function returns unexpected value with kwargs")
        mock_call.assert_called_once_with(2, 2)
        mock_call.return_value = 777
        value = cached_call(2, b=2)
        self.assertEqual(value, mock_call.return_value,
                         "function returns unexpected value on subsequent call with kwargs")
        self.assertEqual(mock_call.call_count, 2, "function with kwargs was cached")

    def test_unhashable_args_not_cached(self):
        mock_call = mock.MagicMock(return_value=42)

        @opt.cache_by_hashed_args
        def cached_call(lst):
            return mock_call(lst)

        arg = [2, 2]
        value = cached_call(arg)
        self.assertEqual(value, mock_call.return_value, "function returns unexpected value with unhashable args")
        mock_call.assert_called_once_with(arg)
        mock_call.return_value = 777
        value = cached_call(arg)
        self.assertEqual(value, mock_call.return_value,
                         "function returns unexpected value on subsequent call with unhashable args")
        self.assertEqual(mock_call.call_count, 2, "function with unhashable args was cached")


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
            with opt.timed_block('time', preMessage=preMessage):
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
        tolerance = 0.01  # this should deliberately be large, for repeatability
        self.assertLessEqual(timeDict['time'], duration + tolerance, "timed block result is greater than {} seconds off".format(tolerance))


class TestTimeHash(BaseCase):
    def test_properties(self):
        value1 = opt.time_hash()
        self.assertTrue(isinstance(value1, str), "time hash is not a string")

        value2 = opt.time_hash()
        self.assertNotEqual(value1, value2, "different time hashes are equal")
