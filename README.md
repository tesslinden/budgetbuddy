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

- **Python Version**: 3.11.3
- **Dependencies**:
  - brewer2mpl==1.4.1
  - contourpy==1.1.1
  - cycler==0.12.1
  - et-xmlfile==1.1.0
  - fire==0.5.0
  - fonttools==4.43.1
  - kiwisolver==1.4.5
  - matplotlib==3.8.0
  - numpy==1.26.0
  - openpyxl==3.1.2
  - packaging==23.2
  - pandas==2.1.1
  - Pillow==10.0.1
  - pyparsing==3.1.1
  - python-dateutil==2.8.2
  - pytz==2023.3.post1
  - seaborn==0.13.0
  - six==1.16.0
  - termcolor==2.3.0
  - tzdata==2023.3


## Installation

TODO


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

## Configuration

Before you can use `budgetbuddy`, you need to create a `config.py` file with your own settings. You can do this by copying the `config.py.example` file and renaming it to `config.py`.

In "config.py", specify the paths to the relevant folders using the PATH_TO_[folder] variables.

TODO: more on configuration

## To Do

* Write unit tests
* Change the importing process so that old raw transactions files are ignored
* Debug bar plot label positioning (note to self: I heard the adjustText library can do this automatically -- should look into it)
* Expand the config.py file to include many parameters that are currently hard-coded elsewhere (e.g. plotting parameters, categories & subcategories)
* Refactor plotting.py and importing.py which are currently a little bit monstrous
* Add feature: ability to edit merged transaction files via command line
* Add feature: ability to annotate a transaction with a note
* Double check which dependencies are required; automatically install dependencies 
* Add installation option