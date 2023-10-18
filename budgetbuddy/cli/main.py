import os
from datetime import datetime
from typing import Dict, List

import fire

from .. import config
from .. import utils # to use functions from the utils folder, e.g. my_function() in the importing.py file, call them like so: utils.importing.my_function()
from ..classes.column import Column
from ..classes.folder import Folder
from ..classes.transactions_df import TransactionsDF

class BudgetBuddyCLI:

    def __call__(self):
        """
        Default method to be called when no specific command is provided.
        """
        print("Welcome to BudgetBuddy! Please provide a specific command.")
        print(f"{config.PATH_TO_TRANSACTIONS=}")
        # You can add more default logic here if desired

    def add_transaction(self, amount, category, description=None):
        """
        Adds a transaction to the transactions file.
        
        Args:
        - amount (float): Amount of the transaction.
        - category (str): Category of the transaction (e.g., "food", "bills").
        - description (str, optional): Additional description for the transaction.
        """
        # Logic to add transaction goes here.
        print(f"Added transaction: ${amount} in {category}.")
        if description:
            print(f"Description: {description}")

    def display_category(self, category):
        """
        Displays transactions for a given category.

        Args:
        - category (str): Category to filter transactions.
        """
        # Logic to fetch and display transactions for a category goes here.
        print(f"Displaying transactions for category: {category}.")

    def search(self, keyword):
        """
        Searches transactions for a particular keyword.

        Args:
        - keyword (str): Keyword to search for.
        """
        # Logic to search transactions by keyword goes here.
        print(f"Searching transactions for: {keyword}")


def display_filtered_transactions(
    tdf: TransactionsDF,
    query: str = None,
    display_columns: List[str] = None,
    sort_by: Dict[str,bool] = None,
    show_sum: bool = False,
    export: bool = False,
    timestamp: datetime = datetime.today(),
):
    """TODO: write docstring"""

    if display_columns is None: display_columns = TransactionsDF.DISPLAY_COLUMNS_NAMES.copy()

    filtered = tdf.split_dates()
    filtered.sort_transactions()

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
        export_path = f"{config.PATH_TO_TRANSACTIONS}/exported/{timestamp.strftime('%y%m%d_%H%M%S')}.csv"
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
    """TODO: write docstring"""
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


if __name__ == "__main__":
    main()