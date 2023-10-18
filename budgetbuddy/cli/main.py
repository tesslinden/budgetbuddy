import fire

from .. import utils # to use functions from the utils folder, e.g. my_function() in the importing.py file, call them like so: utils.importing.my_function()
from .. import config
from classes.folder import Folder
from classes.column import Column
from classes.transactions_df import TransactionsDF

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

def main():
    fire.Fire(BudgetBuddyCLI)

if __name__ == "__main__":
    main()