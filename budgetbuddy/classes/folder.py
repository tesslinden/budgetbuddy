from dataclasses import dataclass
from typing import Dict

from .. import config


@dataclass
class Folder:
    name: str
    prefix: str
    suffix: str
    final_column_index: int = None
    columns_to_rename: Dict[str,str] = None
    account: str = None
    reformat_amounts_alliant: bool = False 
    flip_sign_of_amounts: bool = False
    prune_columns: bool = False
    fill_false: bool = False
    filename_format: str = None

    @property
    def path(self):
        return f"{config.PATH_TO_TRANSACTIONS}/{self.name}"