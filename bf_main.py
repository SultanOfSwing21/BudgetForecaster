import pandas as pd
import numpy as np


def get_forecast(df_year, xls_path, pivots=['category', 'sub_category'], erase_former_forecast=False):
    success = False
    forecast = df_year.groupby(
        ['category', 'sub_category', 'origin', 'month_values', 'nature']).sum().pivot_table(
        index=['category', 'sub_category', 'origin', 'nature'], columns=['month_values'], values='amount').reset_index()
    forecast.to_csv('forecast.csv', index=False)

    forecast = df_year.groupby(
        ['category', 'sub_category', 'attribution', 'origin', 'month_values', 'nature']).sum().pivot_table(
        index=['category', 'sub_category', 'attribution', 'origin', 'nature'], columns=['month_values'],
        values='amount').reset_index()

    try:
        former_forecast = pd.read_excel(xls_path, sheet_name='forecast_table - forecast', header=1).set_index(pivots)
        success = True
    except Exception as e:
        print(e)
        pass
    if erase_former_forecast:
        success = True

    if success:
        new_index = pd.DataFrame(forecast.set_index(pivots).index.to_list() + former_forecast.index.to_list(),
                                 columns=pivots).drop_duplicates().set_index(pivots)

        former_for = new_index.merge(former_forecast, left_index=True, right_index=True, how='inner')

        new_index = new_index.loc[[n for n in new_index.index if n not in former_for.index]]
        new_for = new_index.merge(forecast.set_index(pivots), left_index=True, right_index=True, how='inner')
        df_forecast = pd.concat([former_for, new_for], axis=0)
        df_forecast = df_forecast.reset_index().sort_values(pivots)
    else:
        df_forecast = forecast

    return df_forecast


def get_actual_vs_forecast(df):
    df['nature'] = 'actual'
    df_forecast = get_forecast(df, dataset_path, pivots=['category', 'sub_category', 'origin'],
                               erase_former_forecast=False)

    df_forecast_conc = pd.DataFrame()
    temp = pd.DataFrame()
    for c in df_forecast.columns:
        if c not in range(1, 13):
            temp[c] = df_forecast[c].copy()
    for c in df_forecast.columns:
        if c in range(1, 13):
            temp['months'] = months[c - 1]
            temp['amount'] = df_forecast[c].copy()
            temp['month_values'] = c
            temp['nature'] = 'forecast'
            df_forecast_conc = pd.concat([df_forecast_conc, temp.dropna(how='all', axis=1)], axis=0, ignore_index=True)

    #df_forecast_conc = df_forecast_conc[df.drop('attribution', axis=1).columns]

    df_budget_vs_forecast = pd.concat([df_year_details[df_forecast_conc.columns], df_forecast_conc], axis=0).dropna().groupby(
        ['category', 'sub_category', 'origin', 'month_values', 'nature']).sum().pivot_table(
        index=['category', 'sub_category', 'origin'], columns=['month_values', 'nature'],
        values='amount').reset_index()
    df_budget_vs_forecast.to_csv('actual_vs_forecast.csv')

    return df_budget_vs_forecast, df_forecast


def get_expenses(xls_path):
    variable_expenses = pd.read_excel(xls_path, sheet_name='Achats 2022', header=[0, 1])
    fixed_expenses = pd.read_excel(xls_path, sheet_name='recurrent_expenses - 2022')
    incomes = pd.read_excel(xls_path, sheet_name='Incomes', header=1)
    return variable_expenses.reset_index(drop=True), fixed_expenses.reset_index(drop=True), incomes


def extract_data_4_month_analysis(variable_expenses, fixed_expenses, incomes, selected_month):
    ''' Get variable expenses '''
    var_exp = variable_expenses[variable_expenses['mois '] == selected_month]
    var_exp.rename(columns={'montant': 'amount', 'categorie': 'category',
                            'sous-categorie': 'sub_category'}, inplace=True)

    ''' Get fixed recurrent expenses '''
    month_col_name = fixed_expenses.columns[selected_month]
    col_names = list([month_col_name]) + list(['Categorie', 'sub-categories', 'Attribution'])
    fix_exp = fixed_expenses[col_names].rename(columns={month_col_name: 'amount', 'Categorie': 'category',
                                                        'sub-categories': 'sub_category', 'Attribution': 'attribution'})

    ''' Get incomes '''
    col_names = list([month_col_name]) + list(['Categorie', 'sub-categories', 'Attribution'])
    inc = incomes[col_names].rename(columns={month_col_name: 'amount', 'Categorie': 'category',
                                             'sub-categories': 'sub_category', 'Attribution': 'attribution'})

    col_names = ['amount', 'category', 'sub_category', 'attribution']
    df = pd.concat([var_exp[col_names], fix_exp[col_names], inc[col_names]], axis=0).reset_index(drop=True)

    df['amount'] = np.where(df['category'] == 'Income', df['amount'], -df['amount'])
    return df


def extract_data_4_year_analysis(variable_expenses, fixed_expenses, incomes):
    months = list()  # ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    pivots = list(['category', 'sub_category', 'attribution'])

    '''Get variable expenses'''
    var_exp = variable_expenses.copy()
    var_exp.rename(columns={'montant': 'amount', 'categorie': 'category',
                            'sous-categorie': 'sub_category'}, inplace=True)

    '''var_col = [f"{c[0]}_{c[1]}" for c in var_exp.columns]
    var_exp.columns = var_col
    amount_col = var_exp.columns[var_exp.columns.str.contains("amount")]'''

    var_conc = pd.DataFrame()
    col_inds = list()
    for a, b in var_exp.columns:
        if b == 'amount': \
                col_inds.append((a, b))
    for c in col_inds:
        months.append(c[0])
    for m in months:
        temp = var_exp[m].copy()
        temp['months'] = m
        var_conc = pd.concat([var_conc, temp.dropna(how='all', axis=1)], axis=0, ignore_index=True)

    var = var_conc.groupby(pivots + ['months']).sum().reset_index('months').pivot_table(index=pivots, columns='months',
                                                                                        values='amount').reset_index()

    aug = var['Jul']
    var.drop('Jul', axis=1, inplace=True)
    for m in months:
        if m not in var.columns:
            var[m] = aug
        elif m == 'Jul':
            var[m] = aug
    var['origin'] = 'var_exp'

    '''Get fixed recurrent expenses'''
    fix_exp = fixed_expenses.rename(columns={'depenses': 'cat', 'Categorie': 'category',
                                             'sub-categories': 'sub_category', 'Attribution': 'attribution'})
    fix_exp.drop(['cat', 'Comments'], axis=1, inplace=True)

    for i in range(12):
        fix_exp.rename(columns={fix_exp.columns[i]: months[i]}, inplace=True)
    fix_exp[months] = fix_exp[months].fillna(0)
    fix_exp[pivots] = fix_exp[pivots].fillna('None')
    fix = fix_exp[months + pivots].groupby(pivots).sum().reset_index()
    fix['origin'] = 'fix_exp'

    '''Get incomes'''
    inc = incomes.rename(
        columns={'Categorie': 'category', 'sub-categories': 'sub_category', 'Attribution': 'attribution'})
    for i in range(1, 1 + 12):
        inc.rename(columns={inc.columns[i]: months[i - 1]}, inplace=True)
    inc[months] = inc[months].fillna(0)
    inc[pivots] = inc[pivots].fillna('None')
    incs = inc[months + pivots].groupby(pivots).sum().reset_index()
    incs['origin'] = 'Income'

    df = pd.concat([var, fix, incs], axis=0).reset_index(drop=True)

    for m in months:
        df[m] = np.where(df['category'] == 'Income', df[m], -df[m])
    return df, var.set_index(pivots), fix.set_index(pivots), incs.set_index(pivots)


def extract_data_4_detailed_year_analysis(variable_expenses, fixed_expenses, incomes):
    pivots = list(['category', 'sub_category', 'attribution'])

    '''Get variable expenses'''
    var_exp = variable_expenses.copy()
    var_exp.rename(columns={'montant': 'amount', 'categorie': 'category',
                            'sous-categorie': 'sub_category'}, inplace=True)

    '''var_col = [f"{c[0]}_{c[1]}" for c in var_exp.columns]
    var_exp.columns = var_col
    amount_col = var_exp.columns[var_exp.columns.str.contains("amount")]'''

    def concat_months(df, pivots, multi_index=False):

        months = list()  # ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        df_conc = pd.DataFrame()
        col_inds = list()
        if multi_index:
            for a, b in df.columns:
                if b == 'amount': \
                        col_inds.append((a, b))
            for c in col_inds:
                months.append(c[0])
            for i in range(1, 13):
                temp = df[months[i-1]].copy()
                temp['months'] = months[i-1]
                temp['month_values'] = i
                df_conc = pd.concat([df_conc, temp.dropna(how='all', axis=1)], axis=0, ignore_index=True)
        else:
            months = df.set_index(pivots).columns
            for i in range(1, 13):
                temp = df.drop(months, axis=1)
                temp['amount'] = df[months[i-1]].copy()
                temp['months'] = months[i-1]
                temp['month_values'] = i
                df_conc = pd.concat([df_conc, temp.dropna(how='all', axis=1)], axis=0, ignore_index=True)

        return df_conc.sort_values('month_values'), months

    var, months = concat_months(var_exp, pivots, multi_index=True)
    var['origin'] = 'var_exp'

    '''aug = var['Aug']
    var.drop('Aug', axis=1, inplace=True)
    for m in months:
        if m not in var.columns:
            var[m] = aug
        elif m == 'Aug':
            var[m] = aug'''

    '''Get fixed recurrent expenses'''
    fix_exp = fixed_expenses.rename(columns={'depenses': 'cat', 'Categorie': 'category',
                                             'sub-categories': 'sub_category', 'Attribution': 'attribution'})
    fix_exp.drop(['cat', 'Comments'], axis=1, inplace=True)
    for i in range(12):
        fix_exp.rename(columns={fix_exp.columns[i]: months[i]}, inplace=True)

    fix, months = concat_months(fix_exp, pivots)
    fix['origin'] = 'fix_exp'

    ''' fix_exp[months] = fix_exp[months].fillna(0)
    fix_exp[pivots] = fix_exp[pivots].fillna('None')
    fix = fix_exp[months + pivots].groupby(pivots).sum().reset_index()'''

    '''Get incomes'''
    inc = incomes.rename(
        columns={'Categorie': 'category', 'sub-categories': 'sub_category', 'Attribution': 'attribution'})
    for i in range(1, 1 + 12):
        inc.rename(columns={inc.columns[i]: months[i - 1]}, inplace=True)
    inc.drop(['depenses'], axis=1, inplace=True)

    incs, months = concat_months(inc, pivots)
    incs['origin'] = 'inc_exp'

    '''inc[months] = inc[months].fillna(0)
    inc[pivots] = inc[pivots].fillna('None')
    incs = inc[months + pivots].groupby(pivots).sum().reset_index()
    '''
    df = pd.concat([var, fix, incs], axis=0).reset_index(drop=True)
    df['amount'] = np.where(df['category'] == 'Income', df['amount'], -df['amount'])

    '''    df_months = pd.DataFrame(months, columns=['months'])
    df_months['month_values'] = range(1, 13)

    df = df.merge(df_months, on='months', how='left')'''
    df.to_csv('df_year_details.csv', index=False)
    return df, var, fix, incs


def get_year_balance(var_expenses, fix_expenses, incomes, aggregate=True):
    '''    var = pd.DataFrame(var_expenses.iloc[:, 1:13].sum(), columns='amount')
    fix = pd.DataFrame(fix_expenses.iloc[:, 1:13].sum(), columns='amount')
    inc = pd.DataFrame(incomes.iloc[:, 1:13].sum(), columns='amount')'''

    if aggregate:
        var = pd.DataFrame(var_expenses.sum() * -1, columns=["var_exp"])
        fix = pd.DataFrame(fix_expenses.sum() * -1, columns=["fix_exp"])
        inc = pd.DataFrame(incomes.sum(), columns=['incomes'])
        names = ['months']
        summary = pd.DataFrame(fix_expenses.columns, columns=names)
        summary.set_index(names, inplace=True)

    else:
        var = pd.DataFrame(var_expenses.sum(axis=1) * -1, columns=["var_exp"])
        fix = pd.DataFrame(fix_expenses.sum(axis=1) * -1, columns=["fix_exp"])
        inc = pd.DataFrame(incomes.sum(axis=1), columns=['incomes'])

        names = var.reset_index().columns[:-1].to_list()
        summary = pd.DataFrame(var.index.to_list() + fix.index.to_list() + inc.index.to_list()).drop_duplicates()
        summary.columns = names
        summary.set_index(names, inplace=True)

    summary = summary.merge(var, how='left', left_index=True, right_index=True)
    summary = summary.merge(fix, how='left', left_index=True, right_index=True)
    summary = summary.merge(inc, how='left', left_index=True, right_index=True)
    summary['Balance'] = summary[["var_exp", 'fix_exp', 'incomes']].sum(axis=1)

    return summary


if __name__ == '__main__':
    dataset_path = 'budget_forecaster.xlsx'
    month_of_analysis = 7

    ve, fe, incomes = get_expenses(dataset_path)

    '''fe = fe.set_index('depenses')
    fe.loc['pret voiture', :12] = 0
    fe.reset_index(inplace=True)'''

    # df_month = extract_data_4_month_analysis(ve, fe, incomes, month_of_analysis)
    df_year, var, fix, inc = extract_data_4_year_analysis(ve, fe, incomes)
    df_year_details, var_details, fix_details, inc_details = extract_data_4_detailed_year_analysis(ve, fe, incomes)

    pivots = ['category', 'sub_category', 'attribution']
    # df_year_details.pivot_table(index=pivots, columns=['months', 'origin'], values='amount')
    # df_year_details.groupby(pivots[:-1] + ['months', 'origin']).sum().pivot_table(index=pivots[:-1],
    #                                                                              columns=['months', 'origin'],
    #                                                                              values='amount').sum()

    months = ['Jan', 'Feb', 'mar', 'apr', 'May', 'jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    '''df_year_det_piv = df_year_details.pivot_table(index=pivots, columns=['months', 'origin'], values='amount')
    # df_year_det_piv.reset_index().to_csv("df_year_det_piv_wo_attr.csv")
    df_year_det_piv = df_year_details.groupby(
        ['category', 'sub_category', 'origin', 'months', 'month_values']).sum().pivot_table(
        index=['category', 'sub_category', 'origin'], columns=['month_values', 'months'], values='amount').reset_index()
    '''

    df_actual_vs_forecast, df_forecast = get_actual_vs_forecast(df_year_details)
    month_summary = df_year[df_year['category'] != 'Income'].iloc[:, :14].groupby(['category', 'sub_category']).sum()

    print(f"Expenses per category in August: {month_summary.iloc[:, month_of_analysis].sort_values()} \n")

    summary = get_year_balance(var, fix, inc, aggregate=False)
    summary = df_forecast.drop('attribution', axis=1).fillna(0).groupby(['origin', 'category', 'sub_category']).sum()

    print("Summary of the year balance:")
    print(summary, '\n')
    #    print(f"Total variable expenses per month: {ve['amount'].sum()} \n")
    # print(f"Total fixed expenses per month: {fe.iloc[:, 1:13].sum()} \n")
    # print(f"Total income per month: {incomes.iloc[:, :13].sum()} \n")

    for o in ['inc_exp', 'fix_exp', 'var_exp']:
        print(summary.sum(axis=1).loc[o].sort_values(), '\n')

    print('Tale of yearly expenses: ')
    print(summary.sum(axis=1).groupby(['origin', 'category']).sum().sort_values(), '\n')

    print('Tale of yearly expenses divided by 12: ')
    print((summary.sum(axis=1).groupby(['category']).sum()/12).sort_values(), '\n')


    print(f"Total balance per month:\n {df_year[months].sum()} \n")
    print(f"Total projected balance per month:\n {summary[range(1,13)].sum()} \n")

    total_year_balance = df_year[months].sum().sum()
    print(f"Total balance of the year: {total_year_balance}")
    total_projected_year_balance = summary.sum().sum()
    print(f"projection with extra loans: {total_projected_year_balance}")
    missing_monthly_budget = min(total_projected_year_balance, 0) / 12
    print(f"Need for extra monthly budget of {missing_monthly_budget} Eur")

    1 + 2
