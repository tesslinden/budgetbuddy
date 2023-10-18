import os

import pandas as pd
import re

from .. import config
from ..classes.folder import Folder


def get_matching_row_of_budget(
    budget: pd.DataFrame,
    name: str,
    amount: float,
):
    #TODO: reformat to use string_to_list_of_strings. 
    #TODO: maybe make this more efficient by using something other than for loops
    name_test_cols = ['name_test'+x for x in [str(i) for i in range(1,100)]]
    name_test_cols = [col for col in name_test_cols if col in budget.columns]
    amount_test_cols = ['amount_test'+x for x in [str(i) for i in range(1,100)]]
    amount_test_cols = [col for col in amount_test_cols if col in budget.columns]
    for row in budget.index:        
        name_tests = [budget.loc[row, col] for col in name_test_cols]
        name_tests = [str(name_test).lower() for name_test in name_tests if not pd.isna(name_test)]
        pass_name_test = any([name_test in name.lower() for name_test in name_tests]) or len(name_tests) == 0
        if pass_name_test:
            amount_tests = [budget.loc[row, col] for col in amount_test_cols]
            amount_tests = [float(amount_test) for amount_test in amount_tests if not pd.isna(amount_test)]
            pass_amount_test = any([amount_test == amount for amount_test in amount_tests]) or len(amount_tests) == 0
            if pass_amount_test:
                return row
    return None


def get_max_filename(
    folder: Folder, 
    prefix: str = None, 
    suffix: str = None
) -> str:
    """Returns the max filename with a given prefix & suffix from a given folder within path_to_transactions."""

    assert os.path.exists(folder.path), f"Invalid path: '{folder.path}'"
    assert folder.name in config.FOLDERS_DICT.keys(), f"Invalid folder: '{folder.name}'"
    prefix = prefix or folder.prefix
    suffix = suffix or folder.suffix
    
    filenames = [
        filename 
        for filename in os.listdir(folder.path) 
        if (not filename.startswith('~')) and filename.startswith(prefix) and filename.endswith(suffix)
    ]
    assert len(filenames) != 0, f"No files matching '{prefix}'...'{suffix}' were found in '{folder.path}'."

    if folder.filename_format is not None:
        reformatted_date = r"\g<year>\g<month>\g<day> \g<hour>\g<minute>\g<seconds>"
        filenames_reformatted = [
            re.sub(folder.filename_format, reformatted_date, filename) 
            if re.search(folder.filename_format, filename) 
            else filename 
            for filename in filenames
        ]
    else:
        filenames_reformatted = filenames
    filenames_dict = dict(zip(filenames_reformatted, filenames))

    max_filename_reformatted = max(filenames_reformatted)
    max_filename = filenames_dict[max_filename_reformatted]

    assert filenames_reformatted.count(max_filename_reformatted) <= 1, (
        f"Multiple files in folder '{folder}' are contenders for max filename because they have the same date:\n" + 
        f"{max_filename_reformatted.replace(suffix,'')}" 
    )

    return max_filename