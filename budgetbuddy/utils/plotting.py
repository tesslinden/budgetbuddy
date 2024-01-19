import calendar
from datetime import datetime
from typing import Dict, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import misc
from .. import config
from ..classes.transactions_df import TransactionsDF

def as_money_string(
    x: float, 
    include_cents: bool = False, 
    thousands_as_k: bool = False, 
    k_with_decimal: bool = False
) -> str: 
    """Returns a string representation of an int or float with a dollar sign at the 
    beginning and commas separating thousands. 
    If include_cents is True, includes two decimal places; else, rounds to the nearest 
    dollar.
    """
    s = "${:,.2f}".format(x) if include_cents else "${:,.0f}".format(x)
    if thousands_as_k:
        if k_with_decimal: s = s.replace(',000', '.0k')
        else: s = s.replace(',000', 'k')
        for i in range(1, 10):
            s = s.replace(','+str(i)+'00', '.'+str(i)+'k')
    return s.replace('$-', '-$')


def make_rgb_darker(rgb: Tuple[float,float,float], d: float) -> Tuple[float]:
    """Returns a darker version of the given rgb color."""
    return tuple([max(0, min(1, x-d)) for x in rgb])


def make_rgb_lighter(rgb: Tuple[float,float,float], d: float) -> Tuple[float]:
    """Returns a lighter version of the given rgb color."""
    return tuple([max(0, min(1, x+d)) for x in rgb])


def plot_all(
    tdf: TransactionsDF,
    budget_df: pd.DataFrame,
    salary_target: float = config.SALARY_TARGET,
    savings_target: float = config.SAVINGS_TARGET,
    show_negative_savings: bool = True,
    figsize: Tuple[float,float] = (18, 10), # width, height
    subplot_title_fontsize: float = 15,
    axis_label_fontsize: float = 15,
    categories_bar_label_fontsize: float = 7,
    totals_bar_label_fontsize: float = 10,
    xtick_label_fontsize: float = 11,
    ytick_label_fontsize: float = 12,
    line_label_fontsize: float = 12.5,
    corner_text_fontsize: float = 9,
    show_source: bool = True,
    show_date: bool = True,
    write: str = True,
    show: str = False,
    timestamp: datetime = datetime.today(),
    date_line: datetime = datetime.today(),
    palette: Union[str,Dict[int,Tuple[float,float,float]]] = None,
) -> None:
    """Creates a plot made of multiple subplots (listed below). If write=True, the plot is saved to a file. 
    If show=True, the plot is displayed. 
    If show_source=True, the filename of the merged transactions file (data source) is displayed in the bottom left
    corner of the plot. If show_date=True, today's date and time are displayed in the bottom left corner of the plot.
    Subplots:
    Top left: spending trajectory lineplot (discretionary only)
    Top right: spending trajectory lineplot
    Middle left: spending by category barplot (discretionary only)
    Middle right: spending by category barplot
    Bottom left: total income barplot
    Bottom middle: total savings barplot
    Bottom right: total spending barplot
    """

    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"

    print("\nMaking plots...")

    tdf = tdf.filter("category != 'transfers' and not exclude")
    tdf.split_dates()

    spending_target = (1 - savings_target)*salary_target
    budgeted_spending_target = -budget_df.loc[budget_df['budgeted']==True,'actual_cost'].sum()
    discretionary_spending_target = spending_target - budgeted_spending_target

    if palette is None:
        colors = sns.color_palette('tab10',10)+[sns.color_palette('Set2',10)[5]]+[sns.color_palette('tab20',20)[9]]
        palette = dict(zip(np.arange(1,13), colors))
        yrmo_list = list(tdf.df['yrmo'].sort_values().unique())
        month_to_yrmo = dict(zip([int(yrmo[-2:]) for yrmo in yrmo_list], yrmo_list))
        assert len(month_to_yrmo.keys()) == len(set(month_to_yrmo.keys())), "month_to_yrmo dict has duplicate keys"
        palette = {month_to_yrmo[mo]: palette[mo] for mo in palette.keys() if mo in month_to_yrmo.keys()}
        

    plt.rcParams['figure.constrained_layout.use'] = True
    fig = plt.figure(figsize=figsize, layout='constrained')
    fontsizes = {k: v for k,v in misc.get_arguments(plot_all, locals().copy()).items() if 'fontsize' in k}
    shape = (3,12) # shape (nrows, ncols) is 1-indexed but loc (row #, col #) is 0-indexed
    plot_spending_lines( 
        tdf.filter("not budgeted").df,
        spending_target=discretionary_spending_target,
        shape=shape,
        loc=(0,0), # shape (nrows, ncols) is 1-indexed but loc (row #, col #) is 0-indexed
        rowspan=1,
        colspan=int(shape[1]/2-1),
        title='Spending trajectory (discretionary only)',
        timestamp=date_line,
        palette=palette,
        **fontsizes,
    )
    plot_spending_lines(
        tdf.df,
        spending_target=spending_target,
        yint=-budget_df.loc[budget_df['beginning_of_month'],'actual_cost'].sum(),
        shape=shape,
        loc=(0,int(shape[1]/2)), 
        rowspan=1,
        colspan=int(shape[1]/2-1),
        title='Spending trajectory',
        timestamp=date_line,
        palette=palette,
        **fontsizes,
    )
    plot_categories_bars( 
        tdf.filter("not budgeted").df,  
        transaction_type='spending',
        shape=shape,
        loc=(1,0), 
        rowspan=1,
        colspan=int(shape[1]/2),
        title='Spending by category (discretionary only)',
        palette=palette,
        **fontsizes,
    )
    plot_categories_bars( 
        tdf.df, 
        transaction_type='spending',
        shape=shape,
        loc=(1,int(shape[1]/2)), 
        rowspan=1,
        colspan=int(shape[1]/2),
        title='Spending by category',
        palette=palette,
        **fontsizes,
    )
    plot_totals_bars(
        tdf.df, 
        transaction_type='income',
        salary_target=salary_target,
        shape=shape,
        loc=(2,0), 
        rowspan=1,
        colspan=int(shape[1]/3-1),
        color='seagreen',
        title='Total income',
        stacked=True,
        timestamp=timestamp,
        **fontsizes,
    )
    plot_totals_bars(
        tdf.df, 
        transaction_type='savings',
        salary_target=salary_target,
        savings_target=savings_target,
        shape=shape,
        loc=(2,int(shape[1]/3)), 
        rowspan=1,
        colspan=int(shape[1]/3-1),
        color='royalblue',
        show_negative_savings=show_negative_savings,
        title='\nTotal savings',
        timestamp=timestamp,
        **fontsizes,
    )
    plot_totals_bars(
        tdf.df, 
        transaction_type='spending',
        salary_target=salary_target,
        savings_target=savings_target,
        shape=shape,
        loc=(2,int(2*shape[1]/3)), 
        rowspan=1,
        colspan=int(shape[1]/3-1),
        color='firebrick',
        title='Total spending',
        stacked=True,
        timestamp=timestamp,
        **fontsizes,
    )
    if show_source or show_date:
        corner_text = ''#for some reason adding '\n' here doesn't do anything
        if show_date: corner_text = corner_text + 'Created: ' + timestamp.strftime('%Y-%m-%d at %H:%M:%S')
        if show_date and show_source: corner_text = corner_text + "\n"
        if show_source: corner_text = corner_text + 'Source: ' + tdf.filename
        fig.text(
            0.001,
            0.001,
            corner_text,
            fontsize=corner_text_fontsize,
            va='bottom',
            ha='left',
        )
    if write: 
        path = f"{config.PATH_TO_PLOTS}/{timestamp.strftime('%y%m%d-%H%M%S')}_budget-tracker.png"
        plt.savefig(path, bbox_inches='tight', dpi=300)
        print(f"\nSaved plot to '{path}'")
    if show: plt.show()
    return


def plot_categories_bars(
    transactions_df: pd.DataFrame,
    transaction_type: str,
    shape: Tuple[int,int],
    loc: Tuple[int,int],
    rowspan: int,
    colspan: int,
    title: str = None,
    palette: Union[str,List[Tuple[float,float,float]]] = None,
    **kwargs
) -> None:
    """Plots a barplot of monthly spending by category. Bars are colored by month.
    """
    
    assert transaction_type in ['income','spending'], "transaction_type must be 'income' or 'spending'"

    if transaction_type == 'spending': 
        transactions_df = transactions_df.loc[transactions_df['category']!='income'].copy()
        transactions_df['amount'] = -1*transactions_df['amount']
    elif transaction_type == 'income':
        transactions_df = transactions_df.loc[transactions_df['category']=='income'].copy()
        transactions_df = transactions_df.drop(columns=['category']).rename(columns={'subcategory':'category'})

    monthly_sums = transactions_df[['yrmo','category','amount']
        ].groupby(['yrmo', 'category']).sum().reset_index()
    rows_to_add = []
    for category in monthly_sums['category'].unique():
        for yrmo in monthly_sums['yrmo'].unique():
            if category not in monthly_sums.loc[monthly_sums['yrmo']==yrmo, 'category'].values:
                rows_to_add.append({
                    'yrmo': yrmo,
                    'category': category,
                    'amount': 0,
                })
    monthly_sums = pd.concat([monthly_sums, pd.DataFrame(rows_to_add)], ignore_index=True)
    monthly_sums['abs_amount'] = [abs(x) for x in monthly_sums['amount']]
    monthly_sums = monthly_sums.sort_values(
        by=['yrmo','abs_amount'],
        ascending=[False,False]
    ).reset_index(drop=True)
    monthly_sums = monthly_sums.drop(columns=['abs_amount'])

    plot_df = monthly_sums.copy(deep=True)
    if transaction_type == 'spending':
        category_label_changes = {
            '& ': '&\n',
            'transportation': 'transpor-\ntation',
            'entertainment': 'entertain-\nment',
        }
        for k, v in category_label_changes.items():
            plot_df['category'] = plot_df['category'].str.replace(k, v)

    plot_df['category'] = [category.capitalize() for category in plot_df['category']]
    category_order = plot_df.loc[
        plot_df['yrmo'] < misc.get_yrmo(datetime.today()),
        ['category','amount']
        ].groupby(['category']).sum().sort_values('amount', ascending=False).index
    plot_df['category'] = pd.Categorical(
        plot_df['category'], 
        categories=category_order, 
        ordered=True
    )
    plot_df.sort_values(by=['category','yrmo'], inplace=True)

    ax = plt.subplot2grid(shape=shape, loc=loc, colspan=colspan, rowspan=rowspan)
    sns.barplot(
        data=plot_df,
        ax=ax,
        hue='yrmo',
        y='amount',
        x='category',
        edgecolor='black',
        palette=palette,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, 
        [misc.as_concise_yrmo(label) for label in labels], 
        loc='upper right', 
        edgecolor='black',
        title='Month',
    )
            
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, p: as_money_string(y, thousands_as_k=False))
    )
    for label in ax.get_xticklabels(): label.set_fontsize(kwargs['xtick_label_fontsize']-1.5)
    for label in ax.get_yticklabels(): label.set_fontsize(kwargs['ytick_label_fontsize'])
    ax.set_ylim(0, ax.get_ylim()[1]*1.1)

    for container in ax.containers:
        ax.bar_label(
            container=container, 
            labels=[format(int(v), ',d') if int(v)!=0 else '' for v in container.datavalues],
            fontsize=kwargs['categories_bar_label_fontsize'],
            rotation=45,
            padding=1,
        )
        
    #TODO: when labels are close together in y, bump up every other label to avoid overlap. 
    # plot_df = plot_df.reset_index(drop=True)
    # num_months = plot_df['month'].nunique()
    # for row in plot_df.index:
    #     amount = plot_df.loc[row, 'amount']
    #     if amount == 0: continue
    #     month = plot_df.loc[row, 'month']
    #     label_x = np.floor(row/num_months) + (month-1)/(num_months) - 1/2
    #     ax.annotate(
    #         text=f"{amount:,.0f}",
    #         xy=(label_x, amount),
    #         verticalalignment='bottom',
    #         horizontalalignment='center',
    #         fontsize=kwargs['bar_label_fontsize'],
    #         color='black',
    #     )   

    ax.set_xlabel('\n')
    ax.set_ylabel(None)#transaction_type.capitalize(), fontsize=kwargs['axis_label_fontsize'])
    if title: ax.set_title(title, fontsize=kwargs['subplot_title_fontsize'])
    return


def plot_spending_lines(
    transactions_df: pd.DataFrame,
    spending_target: float,
    shape: Tuple[int,int],
    loc: Tuple[int,int],
    rowspan: int,
    colspan: int,
    show_current: bool = True,
    show_today: bool = True,
    yint: float = None, # TODO: use this to set y intercept
    title: str = None,
    timestamp: datetime = datetime.today(),
    palette: Union[str,List[Tuple[float,float,float]]] = None,
    **kwargs
) -> None:
    """Plots a lineplot of the cumulative spending trajectory over time each month. The plot is annotated with a 
    line indicating the daily spending target and a horizontal line indicating the cumulative amount spent up to today, 
    and a text box saying how much remains to be spent per day in order to stay under the spending target.
    """

    if yint: assert yint >= 0, "yint must be >= 0"

    transactions_df = transactions_df.loc[transactions_df['category']!='income'].copy()
    days_in_current_month = calendar.monthrange(datetime.now().year, datetime.now().month)[1]
    daily_target = (spending_target-(yint or 0))/(days_in_current_month)

    daily_spending = transactions_df[['yrmo','day','amount']].groupby(['yrmo', 'day']).sum().reset_index()
    daily_spending['amount'] = -1*daily_spending['amount']
    date_range = pd.date_range(
        start=transactions_df['date'].min(), 
        end=max(transactions_df['date'].max(),timestamp), 
        freq='D'
    )
    all_days = pd.DataFrame({
        'yrmo': [misc.get_yrmo(dt) for dt in date_range],
        'day':date_range.day,
    })
    daily_spending = pd.merge(left=all_days, right=daily_spending, how='left', on=['yrmo','day'])
    daily_spending['amount'] = daily_spending['amount'].fillna(0)

    yrmo_dfs = [daily_spending[daily_spending['yrmo']==yrmo] for yrmo in daily_spending['yrmo'].unique()]
    for yrmo_df in yrmo_dfs:
        yrmo_df.insert(len(yrmo_df.columns)-1, 'cumulative', yrmo_df['amount'].cumsum())
    daily_spending['cumulative'] = pd.concat(yrmo_dfs)['cumulative'].values    
    
    ax = plt.subplot2grid(shape=shape, loc=loc, rowspan=rowspan, colspan=colspan)
    sns.lineplot(
        data=daily_spending,
        ax=ax,
        hue='yrmo',
        y='cumulative',
        x='day',
        palette=palette,
        linewidth=2,
    )
    ax.set_ylim(yint*0.9 if yint else 0, ax.get_ylim()[1]*1.05)
    ax.set_xlim(1, 31)

    num_months_shown = int(len(ax.lines)/2)
    current_month_index = num_months_shown-1
    for line in ax.lines[:current_month_index]: line.set_alpha(0.3)   

    target_x = np.linspace(1,days_in_current_month,100)
    target_y = daily_target*target_x+(yint or 0)
    ax.plot(target_x, target_y, color='gray', linestyle='--', linewidth=1)

    if show_current:
        daily_spending_current_month = daily_spending[daily_spending['yrmo']==misc.get_yrmo(timestamp)]
        cumulative_spending_today = daily_spending_current_month['cumulative'].iloc[-1]
        today = timestamp.day
        amount_remaining = spending_target - cumulative_spending_today
        num_days_remaining = pd.Timestamp(datetime.today()).daysinmonth-today
        amount_per_day_remaining = amount_remaining/num_days_remaining if num_days_remaining != 0 else amount_remaining
        ax.plot(
            np.linspace(daily_spending_current_month['day'].iloc[-1],31,100), 
            [cumulative_spending_today]*100, 
            color='gray', 
            linestyle='--', 
            linewidth=1,
        )
        if cumulative_spending_today > max(target_y): current_va, target_va = 'bottom', 'top'
        else: current_va, target_va = 'top', 'bottom'
        current_label = (
            f"Current: {as_money_string(cumulative_spending_today)}\n" + 
            f"(Remaining:\n" + 
            f"{as_money_string(amount_per_day_remaining)}/d for {num_days_remaining}\n" + 
            f"days)"
        )
        current_text = ax.text(
            x=ax.get_xlim()[1]+0.2, 
            y=cumulative_spending_today if cumulative_spending_today >= 0 else 0, 
            s=current_label,
            fontsize=kwargs['line_label_fontsize'], 
            va=current_va,
            ha='left',
        )
        current_text.set_in_layout(False)
    if yint:
        target_label = (
            f"Target: {as_money_string(spending_target)}\n" + 
            f"({as_money_string(spending_target-(yint or 0))} after\n" + 
            f"day 1, {as_money_string(daily_target)}/d)"
        )
    else:
        target_label = (
            f"Target: {as_money_string(spending_target)}\n" + 
            f"({as_money_string(daily_target)}/d)"
        )
    target_text = ax.text(
        x=ax.get_xlim()[1]+0.2, 
        y=max(target_y), 
        s=target_label,
        fontsize=kwargs['line_label_fontsize'], 
        va='center' if not show_current else target_va,
        ha='left',
    )
    target_text.set_in_layout(False) # means this object will be ignored by constrained_layout

    if show_today:
        today = timestamp.day
        ax.axvline(x=today, color='black', linestyle='--', linewidth=1, zorder=0)

    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, p: as_money_string(y, thousands_as_k=False))
    )
    for label in ax.get_xticklabels(): label.set_fontsize(kwargs['xtick_label_fontsize'])
    for label in ax.get_yticklabels(): label.set_fontsize(kwargs['ytick_label_fontsize'])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, 
        [misc.as_concise_yrmo(label) for label in labels], 
        loc='upper left', 
        edgecolor='black',
        title='Month',
    )
    ax.set_ylabel(None)
    ax.set_xlabel('Day of month\n', fontsize=kwargs['axis_label_fontsize'])  
    if title: ax.set_title(title, fontsize=kwargs['subplot_title_fontsize'])
    return 


def plot_totals_bars(
    transactions_df: pd.DataFrame,
    transaction_type: str,
    shape: Tuple[int,int],
    loc: Tuple[int,int],
    rowspan: int,
    colspan: int,
    color: str,
    show_negative_savings: bool = True,
    show_target: bool = True,
    savings_target: float = None,
    salary_target: float = None,
    show_average: bool = True,
    stacked: bool = False,
    title: str = None,
    timestamp: datetime = datetime.today(),
    **kwargs
) -> None:
    """Plots a barplot of monthly totals for income, spending, or savings (= income - spending). 
    If show_target==True, the plot is annotated with a line and textbox indicating the target value.
    If show_average==True, the plot is annotated with a line and textbox indicating the average value for
    past months, not including the current month.
    """

    assert transaction_type in ['income','savings','spending'], f"transaction_type must be 'income', 'savings', or 'spending'"
    assert not (transaction_type == 'savings' and stacked == True), f"stacked cannot be True if transaction_type is 'savings'"
    if show_target:
        assert salary_target is not None, "salary_target must be specified if show_target is True"
        assert not (transaction_type in ['savings','spending'] and savings_target is None), (
            f"savings_target must be specified if show_target is True and transaction_type is {transaction_type}"
        )
    
    if show_target:
        if transaction_type == 'income': target = salary_target
        elif transaction_type == 'spending': target = -salary_target*(1-savings_target)
        elif transaction_type == 'savings': target = salary_target*savings_target

    if transaction_type == 'income':
        transactions_df = transactions_df.loc[transactions_df['category'] == 'income']
    elif transaction_type == 'spending':
        transactions_df = transactions_df.loc[transactions_df['category'] != 'income']

    if stacked:
        if transaction_type == 'income':
            transactions_df = transactions_df.drop(columns=['category'])
            transactions_df['category'] = [subcategory.capitalize() for subcategory in transactions_df['subcategory']]
        elif transaction_type == 'spending':
            transactions_df = transactions_df.drop(columns=['category'])
            transactions_df['category'] = ['Budgeted' if budgeted else 'Discretionary' for budgeted in transactions_df['budgeted']]
        plot_df = transactions_df[['yrmo','category','amount']].groupby(['yrmo', 'category']).sum().reset_index()
        plot_df_summed = plot_df[['yrmo','amount']].groupby(['yrmo']).sum().reset_index()
    else:
        plot_df = transactions_df[['yrmo','amount']].groupby(['yrmo']).sum().reset_index()

    #TODO: figure out why some march merged files fail around here. maybe it is when salary < 0 ?
    # if any(plot_df['amount'] < 0):
    #     raise ValueError("plot_totals_bars() is not yet supported when negative values are present in plot_df")

    ax = plt.subplot2grid(shape=shape, loc=loc, colspan=colspan, rowspan=rowspan)
    xs = np.arange(1,len(plot_df['yrmo'].unique())+1)

    if stacked:
        date_range = pd.date_range(start=transactions_df['date'].min(), end=transactions_df['date'].max(), freq='D')
        all_months = pd.concat([pd.DataFrame({
            'yrmo': [misc.get_yrmo(dt) for dt in date_range],
            'category': category
        }) for category in plot_df['category'].unique()])
        all_months = all_months.drop_duplicates().reset_index(drop=True)
        plot_df = pd.merge(left=all_months, right=plot_df, how='left', on=['yrmo','category']).fillna(0)
        categories_amounts = {
            category: plot_df.loc[plot_df['category']==category, 'amount'] for category in plot_df['category'].unique()
        }
        categories_amounts = {k: categories_amounts[k] for k in list(categories_amounts.keys())[::-1]}
        bottom = np.zeros(len(plot_df.loc[plot_df['category']==plot_df.loc[0,'category']]))
        color = make_rgb_darker(matplotlib.colors.to_rgb(color), 0.15)
        for category, amounts in categories_amounts.items():
            ax.bar(
                xs,
                amounts, 
                color=color,
                edgecolor='black',
                label=category, 
                bottom=bottom,
            )
            bottom += amounts
            color = make_rgb_lighter(matplotlib.colors.to_rgb(color), 0.45)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[::-1] if transaction_type == 'income' else handles, 
            labels[::-1] if transaction_type == 'income' else labels, 
            loc='upper left' if transaction_type == 'income' else 'lower left', 
            edgecolor='black'
        )
    else:
        ax.bar(
            xs,
            plot_df['amount'], 
            edgecolor='black',
            color=color,
        )

    if stacked: previous_yrmos_df = plot_df_summed.loc[plot_df_summed['yrmo'] < misc.get_yrmo(timestamp)]
    else: previous_yrmos_df = plot_df.loc[plot_df['yrmo'] < misc.get_yrmo(timestamp)]
    if show_average: previous_months_average = previous_yrmos_df['amount'].mean()

    #TODO: for spending & income, the references to plot_df['amount'].min() or .max() should be changed 
    # from the min/max of any subcategory to the min/max of the two subcategories together 
    if transaction_type=='savings':
        ymin = min([-6500, plot_df['amount'].min()*(1.05 if plot_df['amount'].min() < 0 else 0.95)]) if show_negative_savings else 0
        ymax = max([4000, plot_df['amount'].max()*(1.05 if plot_df['amount'].max() > 0 else 0.95)]) if show_negative_savings else 2500
    elif transaction_type=='spending': 
        ymax = 0 
        ymin = min([-12000, plot_df['amount'].min()*(1.05 if plot_df['amount'].min() < 0 else 0.95)])
    elif transaction_type=='income':
        ymin = 0
        ymax = max([11200, plot_df['amount'].max()*(1.05 if plot_df['amount'].max() > 0 else 0.95)])
        
    ax.set_ylim(ymin,ymax)
    if ax.get_ylim()[0] < 0 and ax.get_ylim()[1] > 0:
        ax.axhline(y=0, color='black', linewidth=1)
    plot_range = ymax - ymin

    bar_label_bottom_padding = plot_range/1000
    bar_label_top_padding = -5*bar_label_bottom_padding
    boost_label_threshold = plot_range/18
    if stacked:
        for i, (category, amounts) in enumerate(categories_amounts.items()):
            amounts = np.array(amounts)
            if i == 0:
                label_ys = amounts
            elif transaction_type=='income':
                label_ys = amounts + [max(label_y,boost_label_threshold) for label_y in label_ys]
            else:
                label_ys = amounts + [min(label_y,boost_label_threshold) for label_y in label_ys]
            for j, amount in enumerate(amounts):
                label_color = 'white' if i==0 else 'black'
                boost_label = True if transaction_type == 'income' and i == 1 and amount <= boost_label_threshold else False
                label_y = label_ys[j]
                if transaction_type == 'income' and label_y < boost_label_threshold:
                    label_y = 0 if i == 0 else boost_label_threshold
                    label_color = 'black'
                verticalalignment = 'bottom' if label_y <= 0 or boost_label else 'top'
                label_y = label_y + (bar_label_bottom_padding if verticalalignment == 'bottom' else bar_label_top_padding)
                ax.annotate(
                    text=f"{amount:,.0f}",
                    xy=(j+1,label_y),
                    verticalalignment=verticalalignment,
                    horizontalalignment='center',
                    fontsize=kwargs['totals_bar_label_fontsize'],
                    color=label_color,
                )
    else:
        amounts = plot_df['amount']
        for i, amount in enumerate(amounts):
            if show_negative_savings:
                if amount >= 0:
                    if amount > boost_label_threshold: 
                        bar_label_color = 'white'
                        verticalalignment = 'top'
                        label_y = amount+bar_label_top_padding
                    else:
                        bar_label_color = 'black'
                        verticalalignment = 'bottom'
                        label_y = amount+bar_label_bottom_padding
                else:
                    if -1*amount > boost_label_threshold:
                        bar_label_color = 'white'
                        verticalalignment = 'bottom'
                        label_y = amount+bar_label_bottom_padding
                    else:
                        bar_label_color = 'black'
                        verticalalignment = 'top'
                        label_y = amount+bar_label_top_padding
            else:
                if amount > ax.get_ylim()[0]:
                    bar_label_color = 'white'
                    if amount < ax.get_ylim()[1]:
                        verticalalignment = 'top'       
                    else: 
                        verticalalignment = 'bottom'       
                else:
                    bar_label_color = 'black'
                    verticalalignment = 'bottom' 
                if verticalalignment == 'top':
                    label_y = amount+bar_label_top_padding
                else:
                    label_y = ax.get_ylim()[0]+bar_label_bottom_padding
            ax.annotate(
                text=f"{amount:,.0f}",
                xy=(i+1,label_y),
                verticalalignment=verticalalignment,
                horizontalalignment='center',
                fontsize=kwargs['totals_bar_label_fontsize'],
                color=bar_label_color,
            )   

    target_va, average_va = ('center', 'center')
    if show_target and show_average:
        label_closeness_limit = plot_range/5
        if abs(abs(target) - abs(previous_months_average)) < label_closeness_limit: 
            target_va, average_va = ('top', 'bottom') if target < previous_months_average else ('bottom', 'top')
    if show_target:
        ax.axhline(y=target, color='black', linestyle='--', linewidth=1, zorder=0)
        target_text = ax.text(
            x=ax.get_xlim()[1]+0.05, 
            y=target if target > ax.get_ylim()[0] else ax.get_ylim()[0], 
            s='Target:\n'+as_money_string(target), 
            fontsize=kwargs['line_label_fontsize'], 
            va=target_va,
            ha='left',
        )
        target_text.set_in_layout(False)
    if show_average:
        average_color = 'green' if previous_months_average >= target else 'crimson'
        ax.axhline(
            y=previous_months_average, 
            color=average_color, 
            linestyle='--', 
            linewidth=1,
            zorder=0,
        )
        average_below_ymin = previous_months_average < ax.get_ylim()[0]
        if average_below_ymin: average_va = 'bottom'
        avg_text = ax.text(
            x=ax.get_xlim()[1]+0.05, 
            y=ax.get_ylim()[0] if average_below_ymin else previous_months_average,
            s=f"Avg ({misc.as_concise_yrmo(previous_yrmos_df['yrmo'].min())}-" +
                f"{misc.as_concise_yrmo(previous_yrmos_df['yrmo'].max())}):\n" + 
                f"{as_money_string(previous_months_average)}", 
            fontsize=kwargs['line_label_fontsize'], 
            color=average_color,
            va=average_va,
            ha='left',
        )
        avg_text.set_in_layout(False)

    ax.set_xticks(xs)
    yrmos = plot_df.loc[plot_df['category']==plot_df['category'].unique()[0],'yrmo'] if stacked else plot_df['yrmo']
    ax.set_xticklabels([misc.as_concise_yrmo(yrmo) for yrmo in yrmos]) 
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, p: as_money_string(y, thousands_as_k=False))
    )
    for label in ax.get_xticklabels(): label.set_fontsize(kwargs['xtick_label_fontsize']+2)
    for label in ax.get_yticklabels(): label.set_fontsize(kwargs['ytick_label_fontsize'])
    ax.set_xlabel("Month\n\n", fontsize=kwargs['axis_label_fontsize'])
    ax.set_ylabel(None)
        
    if title: ax.set_title(title, fontsize=kwargs['subplot_title_fontsize'])
    return