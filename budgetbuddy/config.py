import os
from typing import List, Dict

from .classes.folder import Folder

PATH_TO_BUDGET = '/Users/tesslinden/Dropbox/Misc/! Financial/transactions/budget.csv'
PATH_TO_PLOTS = '/Users/tesslinden/Dropbox/Misc/! Financial/transactions/plots'
PATH_TO_TRANSACTIONS = '/Users/tesslinden/Dropbox/Misc/! Financial/transactions'

for path in [PATH_TO_BUDGET, PATH_TO_PLOTS, PATH_TO_TRANSACTIONS]:
    assert os.path.exists(path), f"Invalid path: '{path}'"

FOLDERS_LIST: List[Folder]= [
    Folder(
        name='asav',
        prefix='History',
        suffix='.csv',
        columns_to_rename={'description': 'transaction'},
        account='Alliant Savings',
        reformat_amounts_alliant=True,
        flip_sign_of_amounts=False,
        prune_columns=True,
        fill_false=False,
        filename_format=(
            r"History-(?P<month>\d{2})(?P<day>\d{2})(?P<year>\d{2})-" + 
            r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<seconds>\d{2})"
        ),
    ),
    Folder(
        name='avis', 
        prefix='History', 
        suffix='.csv', 
        columns_to_rename={'description': 'transaction'}, 
        account='Alliant Visa', 
        reformat_amounts_alliant=True, 
        flip_sign_of_amounts=True, 
        prune_columns=True, 
        fill_false=False,
        filename_format=(
            r"History-(?P<month>\d{2})(?P<day>\d{2})(?P<year>\d{2})-" +
            r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<seconds>\d{2})"
        ),
    ),
    Folder(
        name='manual',
        prefix='manual',
        suffix='.xlsx',
        fill_false=True,
    ),
    Folder(
        name='merged',
        prefix='merged',
        suffix='.csv',
        fill_false=False,
    ),
    Folder(
        name='relay',
        prefix='relay',
        suffix='.txt',
        fill_false=False,
    ),
    Folder(
        name='usbac',
        prefix='USB AC - 5327',
        suffix='.csv',
        columns_to_rename={'transaction': 'transaction_type', 'name': 'transaction',},
        account='USB_AC',
        flip_sign_of_amounts=False,
        prune_columns=True,
        fill_false=False,
        filename_format=(
            r"USB AC - 5327_\d{2}-\d{2}-\d{4}_(?P<month>\d{2})-(?P<day>\d{2})-" + 
            r"(?P<year>\d{4})(?P<hour>)(?P<minute>)(?P<seconds>)"
        ),
    ),
]

FOLDERS_DICT: Dict[str,Folder] = {folder.name: folder for folder in FOLDERS_LIST}

for folder in FOLDERS_LIST:
    assert os.path.exists(folder.path), f"Invalid path: '{folder.path}'"