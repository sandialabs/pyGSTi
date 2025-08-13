import os
import re
from tqdm import tqdm


def check_file(fname: str) -> bool:

    error = False
    if os.path.exists(fname):
        with open(fname, "r") as my_file:
            data = my_file.readlines()

        for i, line in enumerate(data):
            # line numbers in editors start at 1.
            error = error or check_line(line, i + 1, fname)

    return error


def check_line(line: str, ind: int, fname: str) -> bool:

    error_on_line = False
    # Note this is just working the code / comment string as a raw string.
    # It is not parsed.
    error_on_line = error_on_line or check_for_implied_zeros(line, ind, fname)

    # Add more checks here for items which are not already present in the other linters.

    return error_on_line


def check_for_implied_zeros(line: str, ind: int, fname: str) -> bool:

    missing_implied_zero_pattern = re.compile("(?P<LeadingPunctuation>[ ,({])"
                                              + "(?P<LeadingTerm>[^a-zA-Z0-9%]*)"
                                              + "[.](?P<PostDecimal>[0-9][0-9]*)"
                                              + "(?P<Punctuation>[ ,)}])")
    ans = missing_implied_zero_pattern.match(line)

    if ans:
        print(f" There is an implied zero on line number {ind} for file {fname}. The offending line is \n {line}")

        return True
    return False


def test_repo_against_internal_styling():
    direct, _ = os.path.split(__file__)

    direct, _ = os.path.split(direct)  # repo/test/unit
    direct, _ = os.path.split(direct)  # repo/test
    direct, _ = os.path.split(direct)  # repo/

    error = False
    for path, _, files in tqdm(os.walk(direct)):
        for file in files:
            if os.path.splitext(file)[1] == ".py":
                # We have a python file that we may want to correct.
                error = error or check_file(os.path.join(path, file))

    if error:
        raise ValueError(" There has been an error. Check the printed statements for the location.")
