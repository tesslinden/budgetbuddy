import os
from datetime import datetime
from typing import Dict, List

import fire
import pandas as pd
import re

from .. import config
from ..utils import importing, misc, plotting
from ..classes.column import Column
from ..classes.folder import Folder
from ..classes.transactions_df import TransactionsDF


def BudgetBuddyCLI(
    merge: bool = None, 
    i: str = None,
    plot: bool = None, 
    date_line: str = None,
    filter: str = None, 
    keyword: str = None,
    sort_by: List[str] = None,
    show_all_columns: bool = False,
    use_date_orig_for_sorting: bool = False,
    show_sum: bool = False,
    export: bool = False,
    replace: Dict[str,str] = None,
):
    """
    BudgetBuddyCLI is a command line interface for the BudgetBuddy package. 

    -m, --merge: bool = None
        Creates a new merged transactions file by importing any new transactions from non-'merged' folders and merging 
        them with the most recent merged transactions file. Saves the new merged transactions file in the 'merged' folder.
        If no new transactions are found, the most recent merged transactions file is imported instead.

    -i, --i: str = None
        When used with --plot or --filter, specifies the filename of the merged transactions file to import. No need to 
        specify the folder.

    -p, --plot: bool = None
        Creates a plot made of multiple subplots and saves it in the 'plots' folder. 

    -d, --date_line: str = None
        Sets the vertical "today" line in the spending trajectory lineplots to the specified date (format: YYYY-MM-DD).

    -f, --filter: str = None
        Displays transactions matching the specified query. 

    -k, --keyword: str = None
        Displays transactions with descriptions containing the specified keyword. --keyword 'search string' is equivalent 
        to --filter 'transaction.str.contains("search string",case=False)'.

    --sort_by: List[str] = None
        When used with --filter, sorts the displayed transactions by the specified columns.

    --show_all_columns: bool = False
        When used with --filter, displays all columns for the displayed transactions.

    --show_sum: bool = False
        When used with --filter, displays the sum of the amounts of the displayed transactions.

    -e, --export: bool = False
        When used with --filter, exports the displayed transactions to a csv file in the 'exports' folder.

    -r, --replace: Dict[str,str] = None
        Replaces all instances of the specified strings in all csv or txt files with the expected prefixes in the folders 
        specified in folders.keys().
    """

    pd.set_option('display.max_rows', 500) # not sure if this is the best spot for this?

    if replace:
        arguments = misc.get_arguments(main, locals())
        assert all([value in [False, None] for name, value in arguments.items() if name != 'replace']), (
            "--replace cannot be used with other flags"
        )
        assert isinstance(replace, dict) and all(isinstance(k, str) and isinstance(v, str) for k,v in replace.items()), (
            "--replace must be a dict with the format: {'find': 'replace'}"
        )
        for k in replace.keys():
            response = ''
            while response not in ['y', 'n']:
                response = input(
                    f"\nYou are about to replace all instances of '{k}' with '{replace[k]}' (case-sensitive) " + 
                    f"in all csv or txt files with the expected prefixes in the following folders: " + 
                    f"{list(config.FOLDERS_DICT.keys())}. Proceed? (y/n)\t"
                )
            if response == 'n': continue
            find_replace(find=k, replace=replace[k])
            print("Done.")
            print("WARNING: --replace skips files ending in .xlsx.")
        return

    if keyword is not None:
        if filter is None:
            filter = f"transaction.str.contains('{keyword}',case=False)"
        else:
            filter = f"{filter} and transaction.str.contains('{keyword}',case=False)"
    
    if filter is None:
        if merge is None: 
            if i is None: merge = True
            else: merge = False
        if plot is None: plot = True
    else:
        if merge is None: merge = False
        if plot is None: plot = False

    if merge:
        assert i in [False, None], "--merge cannot be used with -i"
        merged = importing.merge_all(write=True)
    else:
        if i is None:
            print("\nImporting latest merged transactions file...")
            merged = importing.FromCsvTransactionsDF(folder=config.FOLDERS_DICT['merged']) 
            print(f"Imported '{merged.filename}'")
        else:
            print(f"\nImporting '{i}'...")
            merged = importing.FromCsvTransactionsDF(folder=config.FOLDERS_DICT['merged'], filename=i)
            print(f"Imported '{merged.filename}'")
        merged.sort_transactions()

    timestamp = datetime.today()
    if date_line:
        assert re.match(r"^\d{4}-\d{2}-\d{2}$",date_line), "--date_line must be in the format YYYY-MM-DD"
        assert pd.to_datetime(date_line) < datetime.today(), "--date_line must be in the past"
        assert pd.to_datetime(date_line).year >= 2023, "--date_line cannot be earlier than 2023"
    else:
        date_line = datetime.today()
    
    if plot:
        assert merged is not None, "merged is None."
        last_date_shown = timestamp
        number_of_previous_months_shown = 6
        first_date_shown = last_date_shown - pd.DateOffset(months=number_of_previous_months_shown)
        first_date_shown = first_date_shown.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        plot_tdf = merged.split_dates().filter("not exclude").filter(
            f"date >= '{first_date_shown}' and month <= {last_date_shown.month} and year <= {last_date_shown.year}"
        )
        plotting.plot_all(
            plot_tdf,
            budget_df=pd.read_csv(config.PATH_TO_BUDGET), 
            timestamp=timestamp,
            date_line=pd.to_datetime(date_line),
        )
    
    if filter is not None:
        assert merged is not None, "merged is None."
        merged.sort_transactions(use_date_orig_for_sorting=use_date_orig_for_sorting)
        display_filtered_transactions(
            merged,
            query=filter,
            display_columns=merged.df.columns if show_all_columns else None, 
            sort_by=sort_by,
            use_date_orig_for_sorting=use_date_orig_for_sorting,
            show_sum=show_sum,
            export=export,
            timestamp=timestamp,
        )

    print('')
    return


def display_filtered_transactions(
    tdf: TransactionsDF,
    query: str = None,
    display_columns: List[str] = None,
    sort_by: Dict[str,bool] = None,
    use_date_orig_for_sorting: bool = False,
    show_sum: bool = False,
    export: bool = False,
    timestamp: datetime = datetime.today(),
):
    """
    Calls TransactionsDF.filter() using the specified query and displays the results. If sort_by is specified, sorts the
    results by the specified columns. If show_sum is True, displays the sum of the amounts of the results. If export is
    True, exports the results to a csv file in the 'exported' folder.
    """

    if display_columns is None: display_columns = TransactionsDF.DISPLAY_COLUMNS_NAMES.copy()

    filtered = tdf.split_dates()
    filtered.sort_transactions(use_date_orig_for_sorting=use_date_orig_for_sorting)

    abs_amount_warning = False
    not_exclude_warning = False
    if query is not None:
        if type(query) is not bool and query != "": # type=bool indicates no query was passed to main()
            if "amount" in query and "abs(amount)" not in query: abs_amount_warning = True
            if "exclude" not in query and 'exclude' not in display_columns: not_exclude_warning = True
            filtered = filtered.filter(query)

    if len(filtered.df) == 0:
        print(f"\nNo transactions were found for query '{query}'.")
        return

    if sort_by is not None:
        filtered.df = filtered.df.sort_values(by=list(sort_by.keys()),ascending=list(sort_by.values()))    
    
    print('')
    print(f"Results for query '{query}':")
    print('')
    print(filtered.copy(cols=display_columns))
    
    if show_sum:
        print('')
        print(f"Sum: {filtered.copy(cols=['amount']).df.sum()[0]}")

    if export:
        filtered = filtered.unsplit_dates()
        filtered.df.insert(len(filtered.df.columns), 'transactions_filename', tdf.filename)
        filtered.df.insert(len(filtered.df.columns), 'query', query)        
        export_path = f"{config.PATH_TO_EXPORTS}/{timestamp.strftime('%y%m%d_%H%M%S')}.csv"
        print(f"\nWriting filtered transactions to '{export_path}'...")
        filtered.df.to_csv(export_path, index=False)
        print('Done.')

    if abs_amount_warning: print("\n!!!!!!!!!!! WARNING: Are you sure you didn't mean 'abs(amount)'? !!!!!!!!!!!")
    if not_exclude_warning: print("\n!!!!!!!!!!! WARNING: Are you sure you didn't mean 'and not exclude'? !!!!!!!!!!!")

    return


def find_replace(
    find: str = None,
    replace: str = None,
):
    """Finds and replaces all instances of the specified strings in all non-xlsx files with the expected prefixes in
    the folders specified in config.FOLDERS_LIST. 
    """
    for folder in config.FOLDERS_LIST:
        files = [file for file in os.listdir(folder.path) if file.startswith(folder.prefix) and not file.endswith('.xlsx')] 
        files.sort()
        for file in files:
            path_to_file = f"{folder.path}/{file}"
            with open(path_to_file) as f:
                s = f.read()
                s_replaced = s.replace(find, replace)
            if s != s_replaced:
                print(f"Replaced {s.count(find)} instances of '{find}' with '{replace}' in '{path_to_file}'.")
            else:
                print(f"No instances of '{find}' found in '{path_to_file}'.")
                continue
            with open(path_to_file, 'w') as f:
                f.write(s_replaced)    
    return


def main():
    fire.Fire(BudgetBuddyCLI)