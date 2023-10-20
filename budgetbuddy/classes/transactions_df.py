import os
from typing import List, Dict

import numpy as np
import pandas as pd
import re

from .column import Column
from .folder import Folder
from .. import config
from ..utils import misc

class TransactionsDF():
    """A class for transactions data frames.

    Attributes:
    - df: A pandas data frame containing the transactions data.
    - folder: The folder (assumed to be within path_to_transactions) in which the transactions data is stored.
    - filename: The name of the file in which the transactions data is stored.
    """

    COLUMNS_LIST: List[Column] = [ #TODO: maybe add hidden abs_amount column for sorting and querying
        Column(name='exclude', base=False, display=True, dtype=bool),
        Column(name='budgeted', base=False, display=True, dtype=bool),
        Column(name='date', base=True, display=True, dtype=np.dtype('datetime64[ns]')),
        Column(name='date_override', base=False, display=False, dtype=np.dtype('datetime64[ns]')),
        Column(name='date_orig', base=False, display=False, dtype=np.dtype('datetime64[ns]')),
        Column(name='year', base=False, display=False, dtype=int),
        Column(name='month', base=False, display=False, dtype=int),
        Column(name='day', base=False, display=False, dtype=int),
        Column(name='transaction', base=True, display=True, dtype=str),
        Column(name='amount', base=True, display=True, dtype=float),
        Column(name='category', base=False, display=True, dtype=str),
        Column(name='subcategory', base=False, display=True, dtype=str),
        Column(name='account', base=True, display=True, dtype=str),
        Column(name='filename', base=True, display=True, dtype=str),
    ]
    COLUMNS_NAMES: List[str] = [col.name for col in COLUMNS_LIST]
    COLUMNS_DICT: Dict[str,Column] = dict(zip(COLUMNS_NAMES, COLUMNS_LIST))

    BASE_COLUMNS_LIST: List[Column] = [col for col in COLUMNS_LIST if col.base]
    BASE_COLUMNS_NAMES: List[str] = [col.name for col in BASE_COLUMNS_LIST]
    BASE_COLUMNS_DICT: Dict[str,Column] = dict(zip(BASE_COLUMNS_NAMES, BASE_COLUMNS_LIST))

    DISPLAY_COLUMNS_LIST: List[Column] = [col for col in COLUMNS_LIST if col.display]
    DISPLAY_COLUMNS_NAMES: List[str] = [col.name for col in DISPLAY_COLUMNS_LIST]
    DISPLAY_COLUMNS_DICT: Dict[str,Column] = dict(zip(DISPLAY_COLUMNS_NAMES, DISPLAY_COLUMNS_LIST))
    
    SORTCOLUMNS_ASCENDINGBOOLS: Dict[str,bool] = {
        'date': False,
        'category': True,
        'subcategory': True,
        'amount': False,
        'transaction': True,
        'account': True,
        'filename': True,
    }
    
    CATEGORIES_SUBCATEGORIES_SEARCHTERMS: Dict[str,Dict] = {
        'bills & utilities': {
            'cell': ['verizon', 'vzw'],
            'internet': ['comcast', 'xfinity', 'internet'],
            'pge': ['pge','pgande', 'pg&e'],
            'rent': ['rent'],
            'other': [],
        },
        'dining & drinks': {},
        'entertainment': {},
        'gifts & donations': {
            'ea': ['givewell', 'animal charity', 'nfgthe life you can s', 'effective altruism'],
            'patronage': ['actblue', 'maximum fun', 'planned parenthood', 'doleac', 'patreon', 'vox media', 'aclu'],
            'one-off': [],
        },
        'groceries': {},
        'home': {},
        'income': {
            'salary': ['gusto', 'fid', 'rover', 'walk'],
            'rewards': []
        },
        'medical': {},
        'other': {},
        'shopping': {},
        'taxes': {},
        'transfers': {},
        'transportation': {},
        'travel': {},
    }

    def __init__(
        self, 
        folder: Folder = None, 
        filename: str = None, 
        prune_columns: bool = False, 
        fill_false: bool = False,
        *args, 
        **kwargs
    ):
        """Creates a TransactionsDF containing at least the columns specified in 
        TransactionsDF.BASE_COLUMNS_NAMES.
        If prune_columns is True, the data frame is pruned to only include the columns specified in 
        TransactionsDF.BASE_COLUMNS_NAMES; else, the data frame is not pruned, and any columns not in 
        TransactionsDF.BASE_COLUMNS_NAMES are moved to the end of the data frame.
        *args and **kwargs are passed to pd.DataFrame().
        """
        if folder is not None: assert folder.name in config.FOLDERS_DICT.keys(), f"Invalid folder: '{folder.name}'"

        self.folder: Folder = folder
        self.filename: str = filename
        self.df: pd.DataFrame = pd.DataFrame(*args, **kwargs)

        cols = TransactionsDF.BASE_COLUMNS_NAMES.copy()
        if not prune_columns: 
            cols.extend([col for col in self.df.columns if col not in TransactionsDF.BASE_COLUMNS_NAMES])
        self.df = self.df.reindex(columns=cols)

        self.df['amount'] = [
            float(amount.replace(',','')) if type(amount) is str else float(amount) for amount in self.df['amount']
        ]
        self.df['amount'] = [round(amount,2) for amount in self.df['amount']] 
        # ^this is important to compare identity of transactions from different sources, especially manual.xlsx 
        # vs. other sources
        
        bool_mappings = {
            'true': True,
            'false': False,
            '1': True,
            '0': False,
            '1.0': True,
            '0.0': False,
            '': False if fill_false else np.nan,
            'nan': False if fill_false else np.nan,
        }
        
        for col_name, col in TransactionsDF.COLUMNS_DICT.items():
            if col_name not in self.df.columns: continue
            if col.dtype == bool: # TODO: replace this rigamarole with pd dtype boolean to handle nulls correctly
                self.df[col_name] = self.df[col_name].astype(str).str.lower().map(bool_mappings)
                if all(self.df[col_name].isin([True,False])):
                    self.df[col_name] = self.df[col_name].astype(bool) 
            elif col.dtype == np.dtype('datetime64[ns]'):
                if all([re.match(r"^\d{1,2}/\d{1,2}/\d{2}$", date_str) for date_str in self.df[col_name].dropna().astype(str)]):
                    # pandas prints a UserWarning if you try to pass a date with a 2-digit year, so we have to specify the format explicitly
                    # ugh
                    self.df[col_name] = pd.to_datetime(self.df[col_name], format='%m/%d/%y')
                else: 
                    self.df[col_name] = pd.to_datetime(self.df[col_name])
            else:
                continue #TODO ???
        

    def __len__(self):
        return len(self.df)


    def __repr__(self): # i think this is called when a variable is the final line of the code
        return (f"TransactionsDF(folder='{self.folder.name}', filename='{self.filename}', " +
               f"df.shape={self.df.shape}, df.columns={list(self.df.columns)})")
    

    def __str__(self): # this is called by print()
        str_df = self.df.copy(deep=True)
        for col in str_df.columns:
            if str_df[col].dtype == 'datetime64[ns]':
                str_df[col] = str_df[col].dt.strftime('%Y-%m-%d')
            elif col == 'amount':
                str_df[col] = [f"{amount:,.2f}" for amount in str_df[col]]
        str_df = str_df.fillna('')
        return str_df.__str__()
    

    def annotate_all(self):
        """Calls self.annotate_column() to request annotations as needed for the following columns (in order):
        category
        exclude
        date_override
        budgeted
        subcategory
        """

        categories = list(TransactionsDF.CATEGORIES_SUBCATEGORIES_SEARCHTERMS.keys())
        self.annotate_column(
            column = TransactionsDF.COLUMNS_DICT['category'],
            annotate_automatically_with_budget = True,
            annotate_question = "Options: "+', '.join(categories)+'\nEnter category (or r=restart): ',
            response_format = rf"^{'$|^'.join(categories)}$",
            store_yn_as_TF = False,
            store_n_as_nan = False,
            fill_nan_with_false = False,
            annotate_rows_that_are_transfers = False,
            annotate_rows_that_are_excluded = True,
            annotate_rows_that_are_from_manual = False,
            columns_to_display = None,
        )

        self.annotate_column(
            column = TransactionsDF.COLUMNS_DICT['exclude'],
            skip_question = "None of the above transactions will be excluded. OK? y/n: ",
            annotate_question = "Exclude this transaction? y/n (r=restart): ",
            response_format = r"^[yn]$",
            store_yn_as_TF = True,
            store_n_as_nan = False,
            fill_nan_with_false = True,
            annotate_rows_that_are_transfers = False,
            annotate_rows_that_are_excluded = True,
            annotate_rows_that_are_from_manual = False,
            columns_to_display = None,
        )

        self.annotate_column(
            column = TransactionsDF.COLUMNS_DICT['date_override'],
            skip_question = "None of the above dates will be overridden. OK? y/n: ",
            annotate_question = "Override date? YYYY-MM-DD=override / n=no / r=restart: ",
            response_format = r"^n$|^\d{4}-\d{2}-\d{2}$",
            store_yn_as_TF = False,
            store_n_as_nan = True,
            fill_nan_with_false = False,
            annotate_rows_that_are_transfers = False,
            annotate_rows_that_are_excluded = True,
            annotate_rows_that_are_from_manual = False,
            columns_to_display = ['date_override','date'] + [col for col in TransactionsDF.DISPLAY_COLUMNS_NAMES if col != 'date'],
        )
        self.override_dates()

        self.annotate_column(
            column = TransactionsDF.COLUMNS_DICT['budgeted'],
            annotate_automatically_with_budget = True,
            annotate_question = "Budgeted? y/n (r=restart): ",
            response_format = r"^[yn]$",
            store_yn_as_TF = True,
            store_n_as_nan = False,
            fill_nan_with_false = True,
            annotate_rows_that_are_transfers = False,
            annotate_rows_that_are_excluded = True,
            annotate_rows_that_are_from_manual = False,
            columns_to_display = None,
        )

        for category in [k for k,v in TransactionsDF.CATEGORIES_SUBCATEGORIES_SEARCHTERMS.items() if len(v)>0]:
            subcategories_searchterms = TransactionsDF.CATEGORIES_SUBCATEGORIES_SEARCHTERMS[category]
            subcategories = list(subcategories_searchterms.keys())
            self.annotate_column(
                column = TransactionsDF.COLUMNS_DICT['subcategory'],
                starting_statement = f"\nAnnotating subcategories within '{category}' category...",
                annotate_automatically_with_dict = True,
                annotation_dict = subcategories_searchterms,
                annotate_question = f"Enter subcategory (options: {subcategories}) or r=restart: ",
                response_format = rf"^{'$|^'.join(subcategories)}$",
                store_yn_as_TF = False,
                store_n_as_nan = False,
                fill_nan_with_false = False,
                annotate_rows_that_are_transfers = False,
                annotate_rows_that_are_excluded = True,
                annotate_rows_that_are_from_manual = False,
                columns_to_display = None,
                exclude_rows = [row for row in self.df.index if self.df.loc[row, 'category'] != category],
            )


    def annotate_column(
        self,
        column: Column,
        starting_statement: str = None,
        skip_question: str = "None of the above transactions will be excluded. OK? y/n: ",
        annotate_automatically_with_budget: bool = False,
        annotate_automatically_with_dict: bool = False,
        annotation_dict: Dict = None,
        annotate_question: str = "Exclude this transaction? y/n (r=restart): ",
        response_format: str = r"^[yn]$",
        store_yn_as_TF: bool = True,
        store_n_as_nan: bool = False,
        fill_nan_with_false: bool = True,
        annotate_rows_that_are_transfers: bool = False,
        annotate_rows_that_are_excluded: bool = True,
        annotate_rows_that_are_from_manual: bool = False,
        exclude_rows: List = None,
        columns_to_display: bool = None,
    ):
        """TODO: Write docstring"""

        from ..utils.importing import get_matching_row_of_budget

        def restart_annotation(locals):
            print("Restarting...")
            arguments = misc.get_arguments(TransactionsDF.annotate_column, locals)
            self.annotate_column(**{k:v for k,v in arguments.items() if k!='self'})

        def get_rows_needing_annotation(tdf: TransactionsDF):
            rows_needing_annotation = tdf.df.loc[
                (tdf.df[column.name].isna()) & 
                (tdf.df['category']!='transfers' if (
                    'category' in tdf.df.columns and not annotate_rows_that_are_transfers
                    ) else [True]*len(tdf.df)) &
                (tdf.df['exclude']==False if (
                    'exclude' in tdf.df.columns and not annotate_rows_that_are_excluded
                    ) else [True]*len(tdf.df)) &
                (tdf.df['filename']!='manual.xlsx' if (
                    'filename' in tdf.df.columns and not annotate_rows_that_are_from_manual
                    ) else [True]*len(tdf.df)) 
            ].index
            if exclude_rows is not None: rows_needing_annotation = [row for row in rows_needing_annotation if not (row in exclude_rows)]
            return rows_needing_annotation

        columns_to_display = columns_to_display or TransactionsDF.DISPLAY_COLUMNS_NAMES
        if column.name not in columns_to_display: columns_to_display = [column.name] + columns_to_display

        assert not (store_yn_as_TF and store_n_as_nan), "store_yn_as_TF and store_n_as_nan cannot both be True"
        if store_yn_as_TF: assert response_format == r"^[yn]$", "if store_yn_as_TF is True then response_format must be r\"^[yn]$\""
        if fill_nan_with_false: assert column.dtype == bool, "if fill_nan_with_false is True then column.dtype must be bool"
        assert not (annotate_automatically_with_budget and annotate_automatically_with_dict), (
            "annotate_automatically_with_budget and annotate_automatically_with_dict cannot both be True"
        )
        if annotate_automatically_with_dict: assert annotation_dict is not None, (
            "annotate_automatically_with_dict is True but annotation_dict is None"
        )

        print(f"\nAnnotating column '{column.name}'..." if starting_statement is None else starting_statement) 

        rows_needing_annotation = get_rows_needing_annotation(self)
        if len(rows_needing_annotation) == 0:
            print(f"No rows need '{column.name}' annotations.")
            return
        
        copy = self.copy()
        
        if not (annotate_automatically_with_budget or annotate_automatically_with_dict):
            print(self.copy(
                rows=rows_needing_annotation,
                cols=[col for col in columns_to_display if col != column.name]
            ))
            response = ''
            while not re.match(r"^[yn]$", response):
                response = input(skip_question).lower()
                if response == 'y':
                    if fill_nan_with_false: 
                        self.df.loc[rows_needing_annotation, column.name] = False
                        self.df[column.name] = self.df[column.name].fillna(False).astype(bool)
                    return
        else:
            if annotate_automatically_with_budget:
                budget = pd.read_csv(config.PATH_TO_BUDGET)
                for row in rows_needing_annotation:
                    name = copy.df.loc[row, 'transaction']
                    amount = copy.df.loc[row, 'amount']
                    row_of_budget = get_matching_row_of_budget(budget, name, amount)
                    if row_of_budget is not None: 
                        copy.df.loc[row, column.name] = budget.loc[row_of_budget, column.name]
            elif annotate_automatically_with_dict:
                for row in rows_needing_annotation:
                    for key in annotation_dict.keys():
                        searchterms = annotation_dict[key]
                        if any([searchterm in copy.df.loc[row, 'transaction'].lower() for searchterm in searchterms]):
                            copy.df.loc[row, column.name] = key
                            break
                        copy.df.loc[row, column.name] = key
            if fill_nan_with_false: copy.df[column.name] = copy.df[column.name].fillna(False).astype(bool)

            if all(copy.df[column.name].isna()):
                print(f"No values of '{column.name}' were able to be automatically annotated.")
            else:
                print('Automatically annotated rows:')
                print(copy.copy(
                    rows=rows_needing_annotation,
                    cols=columns_to_display
                    ).column_to_first(column.name).column_to_last('filename'))
                response = ''
                while not re.match(r"^[yn]$", response):
                    response = input(f"Accept these values of '{column.name}'? y/n: ").lower()
                    if response == 'n':
                        copy=self.copy()
                    if response == 'y': 
                        rows_needing_further_annotation = get_rows_needing_annotation(copy)
                        if len(rows_needing_further_annotation) == 0: self.df = copy.df; return

        rows_needing_further_annotation = get_rows_needing_annotation(copy)         
        for row in rows_needing_further_annotation:
            print('')
            print(copy.copy(rows=row,cols=[col for col in columns_to_display if col != column.name]))
            response = ''
            while not re.match(response_format, response):
                response = input(annotate_question).lower()
                if response == 'r':
                    restart_annotation(locals())
                    return
                if store_yn_as_TF:
                    copy.df.loc[row, column.name] = False if response == 'n' else True
                elif store_n_as_nan:
                    copy.df.loc[row, column.name] = None if response == 'n' else response
                else:
                    copy.df.loc[row, column.name] = response

        print('\nAnnotated rows:')
        print(copy.copy(
            rows=rows_needing_annotation,
            cols=columns_to_display
        ).column_to_first(column.name).column_to_last('filename'))
        response = ''
        while not re.match(r"^[yn]$", response):
            response = input(f"Save these values of '{column.name}'? y=save/n=restart: ").lower()
            if response == 'n': 
                restart_annotation(locals())
                return
            if response == 'y': 
                self.df = copy.df
                if fill_nan_with_false: 
                    self.df[column.name] = self.df[column.name].fillna(False).astype(bool)
                return
    
   
    def category_to_bottom(self, category: str):
        """Returns a copy of the tdf with all rows with a specified value of 'category' moved to the bottom of the 
        data frame."""

        def df_category_to_bottom(df: pd.DataFrame, category: str):
            """Moves all rows a certain value of 'category' to the bottom of the dataframe."""
            top = df[df['category'] != category]
            bottom = df[df['category'] == category]
            df = pd.concat([top, bottom], ignore_index=True).reset_index(drop=True)
            return df.reset_index(drop=True)
        
        copy = self.copy()
        copy.df = df_category_to_bottom(copy.df, category)
        return copy
    
    
    def column_to_first(self, col: str):
        """Returns a copy of the tdf with a specified column moved to the end of the data frame."""

        def df_column_to_first(df: pd.DataFrame, col: str):
            """Moves a column to the end of the dataframe."""
            cols = df.columns.tolist()
            cols.remove(col)
            cols = [col] + cols
            df = df[cols]
            return df

        copy = self.copy()
        copy.df = df_column_to_first(copy.df, col)
        return copy
    
    def column_to_last(self, col: str):
        """Returns a copy of the tdf with a specified column moved to the end of the data frame."""
        
        def df_column_to_last(df: pd.DataFrame, col: str):
            """Moves a column to the end of the dataframe."""
            cols = df.columns.tolist()
            cols.remove(col)
            cols.append(col)
            df = df[cols]
            return df
        
        copy = self.copy()
        copy.df = df_column_to_last(copy.df, col)
        return copy
   
   
    def copy(self, rows: List[int] = None, cols: List[str] = None):
        """Returns a deep copy of self. 
        If a list is passed to cols, the returned copy contains only the columns listed in cols.
        This can be used to print only selected columns of a TransactionsDF.
        """
        copy = TransactionsDF()
        for attr, value in self.__dict__.items():
            copy.__dict__[attr] = value
        copy.df = copy.df.copy(deep=True)
        if rows is not None:
            if isinstance(rows,int): rows = [rows]
            assert all([row in self.df.index for row in rows]), "One or more row indices was not found."
            copy.df = copy.df.loc[rows].copy(deep=True)
        if cols is not None:
            if isinstance(cols,str): cols = [cols]
            assert all([col in self.df.columns for col in cols]), "One or more column names was not found."
            copy.df = copy.df[cols].copy(deep=True)
        return copy

    
    def filter(self, *queries: str, **kwargs):
        """Returns a new transactions data frame that contains only the rows that match the
        specified filters. Filters can be specified as a string that is a valid pandas query, or 
        as keyword arguments, where the keyword is the column name and the value is the value that 
        that column must be equal to.
        TIP: to query for transactions that contain a given substring (e.g. 'xyz'), use the query
        "transaction.str.contains('xyz', case=False)", where case=False means case-insensitive.
        """
        # NOTE: filter() will fail with an inscrutable error message if the query references a column as if it is a 
        # bool when the column's dtype is coded internally as 'object'. e.g.: the query is "not exclude" but 
        # when df['exclude'].dtype is called, the dtype is 'object'.
        # The error message is along the lines of:
        # "None of [Int64Index([-1, -1, -1, -1, -1, -1, -1, ... are in the [index]"
        # Solution: convert the column to bool before running the query. NOTE THAT THIS WILL CONVERT np.nan to TRUE.
        # df['exclude'] = df['exclude'].astype(bool)

        for query in queries: assert isinstance(query, str), f"Invalid query argument: '{query}'"
        for k in kwargs.keys(): assert k in self.df.columns, f"Invalid column name: '{k}'"

        filtered_self = self.copy()
        if len(queries) != 0: filtered_self.df = filtered_self.df.query(' and '.join(queries)) 
        if len(kwargs) != 0: 
            for k, v in kwargs.items():
                filtered_self.df = filtered_self.df[filtered_self.df[k] == v]
        return filtered_self
    
    
    def override_dates(self):
        """Saves the original dates in a 'date_orig' column, then replaces the dates in the 'date column with the 
        ones in the 'date_override' column. If the 'date_override' column is empty, the original date is kept in
        the 'date' column.
        """
        if 'date_override' not in self.df.columns: 
            self.df['date_override'] = np.nan
            self.df['date_orig'] = np.nan
            return
        if 'date_orig' not in self.df.columns:
            self.df['date_orig'] = np.nan        
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['date_override'] = pd.to_datetime(self.df['date_override'])
        rows_to_update_date_orig = self.df.loc[self.df['date_orig'].isnull()].index
        self.df.loc[rows_to_update_date_orig,'date_orig'] = self.df['date']
        self.df['date'] = [x if pd.isnull(y) else y for x, y in zip(self.df['date'], self.df['date_override'])]
        if 'year' in self.df.columns: self.df['year'] = self.df['date'].dt.year
        if 'month' in self.df.columns: self.df['month'] = self.df['date'].dt.month
        if 'day' in self.df.columns: self.df['day'] = self.df['date'].dt.day
    
    
    def prune_external_duplicates(
        self, 
        other, 
        subset: List[str] = ['date_orig', 'transaction', 'account', 'amount'],
    ):
        """prune_duplicates removes transactions from self that are also in other.
        Transactions are considered duplicates if they have the same date, transaction, account,
        and amount.
        """
        if 'date_orig' not in self.df.columns:
            self.df['date_orig'] = self.df['date']
        if self.df['date_orig'].isna().all():
            self.df['date_orig'] = self.df['date']
        if 'date_orig' not in other.df.columns:
            other.df['date_orig'] = other.df['date']
        if other.df['date_orig'].isna().all():
            other.df['date_orig'] = other.df['date']
        self.df['date_orig'] = pd.to_datetime(self.df['date_orig'])
        other.df['date_orig'] = pd.to_datetime(other.df['date_orig'])
        pruned_df = pd.merge(
            left=self.df,
            right=other.df, 
            on=subset,
            how='left', 
            suffixes=['', '_drop'],
            indicator=True,
        )
        # TODO: double check with user if any transactions in self that have duplicates in other 
        # have different category assignments 
        pruned_df = pruned_df.query('_merge == "left_only"')
        columns_to_drop = [col for col in pruned_df.columns if '_drop' in col or col=='_merge']
        self.df = pruned_df.drop(columns=columns_to_drop)

    
    def prune_internal_duplicates(
        self, 
        subset: List[str] = ['date','transaction','account','amount'],
        get_input: bool = True,
    ):
        """Checks for duplicate transactions in the merged transactions data frame.
        If there are duplicates, and 'get_input' is True, prints a list of the duplicates and asks the user
        whether to drop them.
        """
        #TODO: check for Venmo duplicates on different days, like David's Airbnb payment.
        print("\nChecking for internal duplicates...") 
        view_cols = subset
        if 'category' in self.df.columns: view_cols = ['category'] + view_cols
        if 'filename' in self.df.columns: view_cols = view_cols + ['filename']
        df = self.df.copy(deep=True).fillna('nan').sort_values(subset)
        duplicates = df[df.duplicated(subset=subset, keep='first')] # keep=False means all duplicates are marked True
        if len(duplicates) == 0: 
            print('No internal duplicates were found.')
            return
        sets_of_duplicates = []
        for i in duplicates.index:
            duplicates_of_i = list(df[(df[subset] == df.loc[i, subset]).all(axis=1)].index)
            if duplicates_of_i not in sets_of_duplicates:
                sets_of_duplicates.append(duplicates_of_i)
        rows_to_drop = []
        print(f"\nFound {len(sets_of_duplicates)} set(s) of duplicates.")
        for i, set_of_duplicates in enumerate(sets_of_duplicates):
            print('\n',df.loc[set_of_duplicates, view_cols])
            if get_input:
                response = ''
                question = (
                    "For the rows shown above: Drop all but one or keep all? " + 
                    "d=drop / k=keep / r=restart\t"
                )
                while response not in ['d','k']:
                    response = input(question)
                    if response == 'r': 
                        print("Restarting...")
                        self.prune_internal_duplicates(subset=subset, get_input=get_input)
                        return
                    if response == 'd':
                        rows_to_drop.extend(set_of_duplicates[:-1])
            else:
                rows_to_drop.extend(set_of_duplicates[:-1])
        print('\nThe following shows which transactions were dropped:')
        display_df = df.loc[df[df.duplicated(subset=subset, keep=False)].index]
        display_df['dropped?'] = ['DROPPED' if i in rows_to_drop else '' for i in display_df.index]
        print(display_df[['dropped?'] + view_cols] )
        if len(rows_to_drop) != 0:
            df = df.drop(rows_to_drop).reset_index(drop=True).replace('nan', np.nan)
            self.df = df


    def set_merged_filename(self):
        """TODO: write docstring"""

        from ..utils.importing import get_max_filename

        if self.folder.name != 'merged':
            print("WARNING: set_merged_filename() should only be called on a TransactionsDF in the 'merged' folder.")
            return
        #TODO: improve the system for naming merged files
        max_avis_filename = get_max_filename(config.FOLDERS_DICT['avis']).replace('.csv','')
        max_asav_filename = get_max_filename(config.FOLDERS_DICT['asav']).replace('.csv','')
        max_manual_filename = get_max_filename(config.FOLDERS_DICT['manual']).replace('.xlsx','')
        max_relay_filename =  get_max_filename(config.FOLDERS_DICT['relay']).replace('.txt','')
        merged_filename = 'merged_'+'_'.join([max_relay_filename,max_avis_filename,max_asav_filename,max_manual_filename])+'.csv'
        self.filename = merged_filename


    def sort_transactions(self):
        """Sorts the data frame by the columns specified in the keys of TransactionsDF.SORTCOLUMNS_ASCENDINGBOOLS, 
        with the ascending/descending order specified in the values of TransactionsDF.SORTCOLUMNS_ASCENDINGBOOLS.
        """
        #TODO: add inplace=False option

        def df_sort_transactions(df: pd.DataFrame):
            """Sorts a df by the column names and ascending booleans in TransactionsDF.SORTCOLUMNS_ASCENDINGBOOLS."""

            cols_ascendingbools = TransactionsDF.SORTCOLUMNS_ASCENDINGBOOLS.copy()
            cols_ascendingbools = {
                k: v for k, v in cols_ascendingbools.items() if k in df.columns
            }
            df = df.sort_values(
                by=list(cols_ascendingbools.keys()), 
                ascending=list(cols_ascendingbools.values()),
            )
            return df

        self.df = df_sort_transactions(self.df)


    def split_dates(self, inplace=False):
        """Adds three new columns to the TransactionsDF: 'year', 'month', and 'day', based
        on the 'date' column. If inplace=True, the changes are made to self.df. If inplace=False,
        a copy of self.df is returned with the changes.
        """
        if inplace: 
            self.unsplit_dates(inplace=True)
            self.df.insert(0, 'year', self.df['date'].dt.year)
            self.df.insert(1, 'month', self.df['date'].dt.month)
            self.df.insert(2, 'day', self.df['date'].dt.day)
        else: 
            copy = self.copy()
            copy.split_dates(inplace=True)
            return copy
        

    def to_csv(self, folder: Folder = None, filename: str = None, *args, **kwargs):
        """Writes the transactions data frame to a csv file. The path is path_to_transactions/self.folder/self.filename. 
        If the file already exists, the user is asked whether to overwrite it.
        """
        folder = folder or self.folder
        filename = filename or self.filename
        assert folder is not None, "folder is None."
        assert self.filename is not None, "filename is None."
        if os.path.exists(self.path_to_file):
            response = ''
            while response not in ['y','n']:
                response = input(f"\nFile already exists: '{self.path_to_file}'\nOverwrite? y/n\t")
                if response == 'n': 
                    i = 1
                    while os.path.exists(self.path_to_file): 
                        search_string = r"_(?P<j>\d+).csv$"
                        if re.search(pattern=search_string, string=self.filename): 
                            self.filename = re.sub(pattern=search_string, repl=".csv", string=self.filename)
                        number_string = f"{i}" if i >= 10 else f"0{i}"
                        self.filename = self.filename.replace(".csv",f"_{number_string}.csv")
                        i += 1
                if response == 'y': 
                    response = input("Previous file will be overwritten. Type y again to confirm (or type anything else to go back)\t").lower()
                    if response == 'n': response = ''

        date_formatted_df = self.df.copy()
        date_cols = [col for col in date_formatted_df.columns if (col.startswith('date') or col in ['year','month','day'])]
        for col in date_cols:
            date_formatted_df[col] = [d.strftime('%m/%d/%y') if not pd.isnull(d) else '' for d in date_formatted_df[col]]

        date_formatted_df.to_csv(self.path_to_file, index=False, *args, **kwargs)
        print(f"\nWrote '{self.path_to_file}'")

    
    def unsplit_dates(self, inplace=False):
        """Removes the columns 'year', 'month', and 'day' from the TransactionsDF, if they exist.
        If inplace=True, the changes are made to self.df. 
        If inplace=False, a copy of self.df is returned with the changes.
        """
        if inplace: 
            if 'year' in self.df.columns: self.df = self.df.drop(columns=['year'])
            if 'month' in self.df.columns: self.df = self.df.drop(columns=['month'])
            if 'day' in self.df.columns: self.df = self.df.drop(columns=['day'])
        else: 
            copy = self.copy()
            copy.unsplit_dates(inplace=True)
            return copy
        

    @property
    def path_to_file(self):
        """Returns the full path to the file that contains the transactions data frame."""
        parts_of_path = [self.folder.path, self.filename] 
        if any([part is None for part in parts_of_path]): return None
        return '/'.join(parts_of_path)
    

def assert_compatible_dates(tdf: TransactionsDF):
    """Raises an exception if the 'date' and 'date_override' columns of the TransactionsDF are incompatible 
    (i.e. if there are rows where 'date_override' is not None and 'date_override' != 'date').
    """
    copy = tdf.copy()
    copy.override_dates()
    assert (copy.df['date'] == tdf.df['date']).all(), (
        f"Incompatible values of date and date_override detected in file '{tdf.filename}': \n" +
        f"{tdf.df.loc[copy.df['date'] != tdf.df['date'], ['date_orig','date_override','date','transaction','amount','account']]}"
    )


def assert_no_duplicate_rows(tdf: TransactionsDF, subset: List[str] = None):
    """Raises an exception if there are duplicate rows in the TransactionsDF. 'subset' argument is a list of column names 
    to be passed to the pandas.DataFrame.duplicated method."""
    subset = subset or tdf.df.columns.tolist()
    duplicate_transactions = TransactionsDF(data=tdf.df[tdf.df.duplicated(subset=subset, keep=False)])
    assert len(duplicate_transactions) == 0, (
        f"Duplicate transactions are not supported in file '{tdf.filename}'. The duplicate transactions are:\n" +
        f"{duplicate_transactions.copy(cols=subset)}"
    )