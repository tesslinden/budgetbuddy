import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import re

from .. import config
from ..classes.folder import Folder
from ..classes.transactions_df import TransactionsDF
from ..classes.transactions_df import assert_compatible_dates, assert_no_duplicate_rows


def FromCsvTransactionsDF(**kwargs) -> TransactionsDF:
    """Imports a csv file as a TransactionsDF by calling FromSpreadsheetTransactionsDF."""
    return FromSpreadsheetTransactionsDF(filetype='csv', **kwargs)
  

def FromExcelTransactionsDF(**kwargs) -> TransactionsDF:
    """Imports an excel file as a TransactionsDF by calling FromSpreadsheetTransactionsDF."""
    return FromSpreadsheetTransactionsDF(filetype='excel', **kwargs)


def FromSpreadsheetTransactionsDF(
    filetype: str,
    folder: Folder, 
    filename: Optional[str] = None, 
    prefix: Optional[str] = None,
    prune_columns: Optional[bool] = None,
    fill_false: Optional[bool] = None,
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
    filename: Optional[str] = None, 
    prefix: Optional[str] = None,
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
    #TODO: make this more efficient by using something other than for loops (df.merge perhaps)
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
    prefix: Optional[str] = None, 
    suffix: Optional[str] = None
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


def concat_all_unannotated(
    prune_tdf: Optional[TransactionsDF] = None,
) -> Tuple[TransactionsDF, List[str]]: 
    """Imports all unannotated transactions from all folders in folders.keys() (except 'merged') and 
    concatenates them into a single TransactionsDF.
    If prune_tdf is not None, prune_tdf is passed to the prune_external_duplicates method of each TransactionsDF.
    Returns the concatenated TransactionsDF and a list of the filenames of the imported transactions.
    """

    merged_filename = 'merged'
    transaction_dfs = []
    num_dfs = 0
    for folder in config.FOLDERS_LIST:
        if folder.name == 'merged': continue
        filenames = [filename for filename in os.listdir(folder.path) if filename.startswith(folder.prefix)]
        if len(filenames) == 0: continue
        if folder.suffix == '.txt':
            new_dfs = [FromTxtTransactionsDF(folder=folder,filename=filename) for filename in filenames]
        elif folder.suffix == '.xlsx':
            new_dfs = [FromExcelTransactionsDF(folder=folder,filename=filename) for filename in filenames]
        else:
            new_dfs = [FromCsvTransactionsDF(folder=folder,filename=filename) for filename in filenames]
        for new_df in new_dfs:
            new_df.df['filename'] = new_df.filename
            if prune_tdf: 
                if folder=='manual': 
                    new_df.prune_external_duplicates(prune_tdf, subset=['date_orig','date','transaction','account','amount'])
                else: 
                    new_df.prune_external_duplicates(prune_tdf)
        transaction_dfs.extend([new_df for new_df in new_dfs if len(new_df) > 0])
        new_num_dfs = len(transaction_dfs)
        if new_num_dfs > num_dfs: 
            merged_filename = merged_filename + '_' + get_max_filename(folder,folder.prefix,folder.suffix)[:-4]
        num_dfs = len(transaction_dfs)
        
    if len(transaction_dfs) == 0:
        return TransactionsDF(data=pd.DataFrame()), []

    concat_df = pd.concat([transaction_df.df for transaction_df in transaction_dfs], ignore_index=True)
    concat_filenames = concat_df['filename'].unique().tolist()
    concat_filenames.sort()
    concat_transactions = TransactionsDF(data=concat_df, folder=config.FOLDERS_DICT['merged'], filename=merged_filename+'.csv') 
    
    return concat_transactions, concat_filenames


def merge_all(write: bool = True) -> TransactionsDF:
    """Imports all unannotated transactions from all non-'merged' folders in folders.keys(), prunes them 
    against the latest merged transactions file, and concatenates them into a single unannotated TransactionsDF.
    Prunes internal duplicates from the unannotated transactions, then calls annotate_all to annotate the unannotated 
    transactions. Concatenates the newly annotated and previously annotated transactions into a new merged 
    TransactionsDF and writes it to a csv file in the 'merged' folder.
    """

    # 0. import latest file of previous merged transactions
    print("\nImporting latest merged transactions file...")
    annotated = FromCsvTransactionsDF(folder=config.FOLDERS_DICT['merged']) 
    annotated.sort_transactions()
    print(f"Imported {annotated.filename}.")

    # 1. if there are transactions in merged that have been deleted from manual.xlsx, delete them from annotated df
    print("\nChecking for deleted manual transactions...")
    manual = FromExcelTransactionsDF(folder=config.FOLDERS_DICT['manual'])
    assert_compatible_dates(manual)
    assert_no_duplicate_rows(manual, subset=[col for col in manual.df.columns if col not in ['date_override','date_orig']])
    manual.sort_transactions()

    annotated_pruned = pd.merge(
        left=annotated.df,
        right=manual.df, 
        on=['date','date_orig','date_override','transaction','category','amount','account','budgeted','exclude'],
        how='left', 
        suffixes=[None, '_drop'],
        indicator=True,
    )
    deleted_transactions_index = annotated_pruned.query("filename == 'manual.xlsx' and _merge == 'left_only'").index
    deleted_transactions = TransactionsDF(data=annotated_pruned.loc[deleted_transactions_index])
    if len(deleted_transactions) > 0:
        print(f"\nThe following {len(deleted_transactions)} transactions were not found in manual.xlsx and will be deleted:")
        print(deleted_transactions.copy(cols=TransactionsDF.DISPLAY_COLUMNS_NAMES))
        columns_to_drop = [col for col in annotated_pruned.columns if '_drop' in col or col=='_merge']
        annotated.df = annotated_pruned.drop(deleted_transactions_index).drop(columns=columns_to_drop)
    else:
        print("No deleted manual transactions were found.")

    # 2. import ALL raw files: relay, avis, asav, manual; prune transactions that are already in annotated df
    print("\nImporting unannotated transactions files & pruning them against the annotated transactions...")
    unannotated, unannotated_filenames = concat_all_unannotated(prune_tdf=annotated if len(annotated) > 0 else None)
    unannotated.sort_transactions()
    if len(unannotated) == 0: 
        print("No new transactions were found.")
        if len(deleted_transactions) > 0: 
            print("\nSaving file with deleted manual transactions...")
            annotated.to_csv()
        return annotated
    else:
        print(f"Imported {len(unannotated)} new transaction(s) from {len(unannotated_filenames)} file(s): " + 
              f"{unannotated_filenames}")
        print("\nNew transactions:")
        print(unannotated.copy(cols=TransactionsDF.DISPLAY_COLUMNS_NAMES).category_to_bottom('transfers'))

    # 3. prune internal duplicates
    unannotated.prune_internal_duplicates()

    # 4. add category, exclude, date_override, budgeted, and subcategory annotation columns
    unannotated.annotate_all()
    unannotated.sort_transactions()
    
    print("\nRows to be deleted:")
    if len(deleted_transactions) > 0:
        print(deleted_transactions.copy(cols=TransactionsDF.DISPLAY_COLUMNS_NAMES).column_to_last('filename').category_to_bottom('transfers'))
    else:
        print("None")
    print("\nRows to be added:")
    print(unannotated.copy(cols=TransactionsDF.DISPLAY_COLUMNS_NAMES).column_to_last('filename').category_to_bottom('transfers'))

    response = ''
    while response not in ['y','n']:
        response = input("\nSave these changes? (y/n) ").lower() # TODO: add option for user to ask for more columns to be displayed
        if response == 'n': 
            confirm = input("All changes will be lost. Type n again to confirm (or type anything else to go back)\t").lower()
            if confirm == 'n': return annotated
            else: response = ''

    # 5. concat annotated & newly annotated
    if sorted(list(annotated.df.columns)) != sorted(list(unannotated.df.columns)):
        print("WARNING: annotated and newly_annotated tdfs have different columns: "+(
            f"{annotated.df.columns=} vs. {unannotated.df.columns=}"))

    merged = TransactionsDF(data=pd.concat([annotated.df, unannotated.df], ignore_index=True), folder=config.FOLDERS_DICT['merged'])
    merged.set_merged_filename()
    merged.sort_transactions()
    if write: merged.to_csv()
    return merged


def string_to_list_of_strings(s: str) -> List[str]: # TODO: use this for name_test, desctest etc
    """Converts the string "['a', 'b', 'c']" to the list ['a', 'b', 'c'] etc.
    Quotes within the list string can be single or double quotes.
    """
    list_without_brackets = s[1:-1]
    return [s[1:-1] for s in list_without_brackets.split(',')]