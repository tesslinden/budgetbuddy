import os

import pandas as pd
import re

from .. import config
from ..classes.folder import Folder
from ..classes.transactions_df import TransactionsDF


def FromCsvTransactionsDF(**kwargs) -> TransactionsDF:
    """Imports a csv file as a TransactionsDF by calling FromSpreadsheetTransactionsDF."""
    return FromSpreadsheetTransactionsDF(filetype='csv', **kwargs)
  

def FromExcelTransactionsDF(**kwargs) -> TransactionsDF:
    """Imports an excel file as a TransactionsDF by calling FromSpreadsheetTransactionsDF."""
    return FromSpreadsheetTransactionsDF(filetype='excel', **kwargs)


def FromSpreadsheetTransactionsDF(
    filetype: str,
    folder: Folder, 
    filename: str = None, 
    prefix: str = None,
    prune_columns: bool = None,
    fill_false: bool = None,
) -> TransactionsDF:
    """Imports a csv file as a TransactionsDF.
    If filename is None, the max filename with the given prefix and the suffix '.csv'
    from the directory 'path_to_transactions+'/'+folder' is used.
    If prefix is None, the default prefix for the given folder is used.
    """

    assert folder.name in config.FOLDERS_DICT.keys(), f"Invalid folder: '{folder.name}'"
    assert filetype in ['csv','excel'], f"Invalid filetype: '{filetype}'. Filetype must be 'csv' or 'excel'."
    suffix = '.csv' if filetype == 'csv' else '.xlsx'
    if prune_columns is None: prune_columns = folder.prune_columns
    if fill_false is None: fill_false = folder.fill_false

    if filename is None and prefix is None: prefix = ''
    elif prefix is None: prefix = folder.prefix 
    filename = filename or get_max_filename(folder)

    path_to_file = f"{folder.path}/{filename}"
    if filetype == 'csv': 
        df = pd.read_csv(path_to_file)
    else: 
        df = pd.read_excel(path_to_file)
    assert list(df.index) == list(range(len(df))), f"Invalid index for file '{path_to_file}': {df.index=}"

    df.columns = [col.lower() for col in df.columns]
    if 'account' not in df.columns: 
        assert folder.account is not None, "folder.account is None"
        df['account'] = folder.account
    if folder.columns_to_rename is not None: df = df.rename(columns=folder.columns_to_rename)
    if folder.reformat_amounts_alliant:
        df['amount'] = [
            float(amount.replace('$','').replace(',','').replace('(','-').replace(')','')) for amount in df['amount']
        ]
    if folder.flip_sign_of_amounts: df['amount'] = -df['amount']

    if 'filename' not in df.columns: df['filename'] = filename
    
    if folder.name == 'manual':
        assert all([category in TransactionsDF.CATEGORIES_SUBCATEGORIES_SEARCHTERMS.keys() for category in df['category']]), (
            "Invalid categories in 'manual' folder: {invalid_categories}.".format(
                invalid_categories = (set(df['category']) - set(TransactionsDF.CATEGORIES_SUBCATEGORIES_SEARCHTERMS.keys()))
            )
        )

    return TransactionsDF(data=df, folder=folder, filename=filename, prune_columns=prune_columns, fill_false=fill_false)


def FromTxtTransactionsDF(
    folder: Folder, 
    filename: str = None, 
    prefix: str = None,
) -> TransactionsDF:
    """Imports a relay txt file as a TransactionsDF.
    If filename is None, the max filename with the prefix 'relay' and the suffix '.txt'
    from the directory 'path_to_transactions+'/'+folder' is used; default folder is 'relay'.
    """
    assert folder.name in config.FOLDERS_DICT.keys(), f"Invalid folder: '{folder.name}'"
    
    filename = filename or get_max_filename(folder)
    
    path_to_file = f"{folder.path}/{filename}"
    with open(path_to_file, 'r') as f:
        relay_txt = f.read()
        relay_txt = '\n'+relay_txt # this fixes a bug where the first day is omitted if the file doesn't start with \n

    date_query = r'Pending|\n\d{2}/\d{2}/\d{4}'
    date_matches = re.findall(date_query, relay_txt)
    transactions_by_date = dict(zip(date_matches,re.split(date_query, relay_txt)[1:])) 

    for k, v in transactions_by_date.items():
        transactions_by_date[k] = v.strip('\n').replace('\n\n', '\n')
    
    if 'Pending' in transactions_by_date.keys(): del transactions_by_date['Pending']
    while 'Pending' in date_matches: date_matches.remove('Pending')

    dfs = []
    for date in date_matches:
        split = re.split(r'\n', transactions_by_date[date])    
        df = pd.DataFrame({
            'date': date.replace('\n', ''),
            'transaction': split[::4],
            'account': split[1::4],
            'category': split[2::4],
            'amount': split[3::4],
        })
        df['amount'] = [float(amount.replace('$','').replace(',','')) for amount in df['amount']]
        dfs.append(df)
    if len(dfs) == 0:
        print(f'\nWARNING: No transactions were found in file {path_to_file}.\n')
        df = pd.DataFrame(columns=TransactionsDF.BASE_COLUMNS_NAMES)
    else:
        df = pd.concat(dfs, ignore_index=True)
        df.columns = [col.lower() for col in df.columns]
        df['category'] = [category.lower() for category in df['category']]
    df['filename'] = filename

    return TransactionsDF(data=df, folder=folder, filename=filename)


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