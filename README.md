# BudgetBuddy: a command-line budget tracking app


## Table of contents

- Summary
- Requirements
- Installation
- Usage
- Configuration
- To Do


## Summary

BudgetBuddy is a command-line app that I use to keep track of my spending habits. It was born out of a frustration with personal finance apps like Mint or RocketMoney. After a month or two of trying and failing to find an app with all of the features I wanted, I decided to make one myself.

BudgetBuddy is somewhat customized to my specific use cases and was not originally intended to be publicly useful. For that reason, there's not much in the way of documentation, although I'd be happy to answer questions. Some pieces of the package would be unnecessary for most people, but the base code could be useful to someone else.

BudgetBuddy currently has some sections that are messy due to accumulated patches (especially plotting.py and importing.py). I mean to refactor them eventually when I have a bit of time. :)


## Requirements

- Python 3.11 or higher


## Installation

To install this package, run the following command:

```bash
pip install git+https://github.com/tesslinden/budgetbuddy.git
```

For next steps, see "Configuration."


## Configuration

Before you can use `budgetbuddy`, you need to create a `config.py` file with your own settings. You can do this by copying the `config.py.example` file and renaming it to `config.py`.

The `config.py.example` file is located in the `budgetbuddy` package directory. To find your python package directories, you can open an interactive python shell and run:

```
import site
print(site.getsitepackages())
```

This should print a path to a directory called `site-packages`. Navigate to the `site-packages` directory, then create `config.py` by copying `config.py.example`: 

```
cp budgetbuddy/config.py.example budgetbuddy/config.py
```

Edit `config.py` to specify your salary and savings targets, as well as the paths to your transactions folders.

TODO: explain the required folders and files


## Usage

By default (if you run `budgetbuddy` with no arguments), BudgetBuddy will do the following:
1. Import the most recent merged transactions file
2. Import the manual transactions file and check if any new transactions are present or if any transactions have been deleted
3. Import the raw transactions files and look for any new transactions not found in the merged file; if there are new transactions, ask the user to annotate them
4. Make a new merged file with the deletions and additions identified in steps 2 & 3; save it in the merged transactions folder
5. Import the budget file; use the budget file and the new merged file from step 4 to plot spending and income patterns for the past 6 months; save the plot in the plots folder 

Here is an example output made with mock data: 
![Example output made with mock data](example_output_using_mock_data.png)

The default merge-and-plot mode accepts some arguments to modify its behavior: for example, `--plot False` skips step 5; `--merge False` skips steps 2-4; `-i filename.csv` specifies a file to import in step 1 and skips steps 2-4.

BudgetBuddy can also be run in two other modes that are mutually exclusive with the default merge-and-plot mode: 
* Search mode: if `--keyword` and/or `--filter` arguments are specified, BudgetBuddy will import the most recent merged transactions file and print any rows matching the queries specified by the arguments.
* Find & replace mode: if `--replace` argument is specified, BudgetBuddy will find & replace the specified phrase(s) in all previous transactions files.

For more details, run `budgetbuddy --help`.


## To Do

* Add --config argument to enter configuration setting mode; encode config as json instead of py
* Write unit tests
* Change the importing process so that old raw transactions files are ignored
* Debug bar plot label positioning (note to self: I heard the adjustText library can do this automatically -- should look into it)
* Expand the config.py file to include many parameters that are currently hard-coded elsewhere (e.g. plotting parameters, categories & subcategories)
* Refactor plotting.py and importing.py which are currently a little bit monstrous
* Add feature: ability to edit merged transaction files via command line
* Add feature: ability to annotate a transaction with a note