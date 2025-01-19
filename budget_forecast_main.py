from dash import Dash, html, dcc, Input, Output, State, dash_table

import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go

import plotly.express as px
import numpy as np
import pandas as pd

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
act_vs_dd_options = [{'label': m, 'value': m} for m in month_names]
sel_months = [7, 8, 9]
target = 'Indrani'
table_col = ['category', 'sub_category', 'origin', 'forecast', 'expenses', 'act_vs_for']

class data_variable:
    def __init__(self):
        self.df = None
        self.df_full = None
        self.act_vs_for = None
        self.monthly_purchases = None
        self.monthly_forecast = None
        self.incomes = None

dv = data_variable()

def import_data():
    real_exp = pd.read_excel('Indrani_forecast.xlsx', sheet_name='Achats 2022', header=[0, 1])
    db = pd.read_excel('Indrani_forecast.xlsx', sheet_name='database', header=[2])
    forecast_glabais = pd.read_excel('Indrani_forecast.xlsx', sheet_name='Expense forecast glabais - 2022', header=[0])
    forecast_indrani = pd.read_excel('Indrani_forecast.xlsx', sheet_name='Expense forecast Indrani', header=[0])
    incomes = pd.read_excel('Indrani_forecast.xlsx', sheet_name='Incomes', header=[1])
    return real_exp, db, forecast_glabais, forecast_indrani, incomes

def get_mean_per_cat_and_subcat(real_exp):
    act_vs_forect = run(['Jul'], target)
    var = {m: real_exp[m].groupby(['category', 'sub_category']).sum() for m in month_names}
    ind = pd.DataFrame({k: act_vs_forect['Jul'][k] for k in list(act_vs_forect['Jul'].keys())[:-1]}).groupby(
        ['category', 'sub_category']).sum().index
    mean_var_exp = pd.DataFrame([], index=ind)
    for m in month_names:
        mean_var_exp[m] = mean_var_exp.merge(var[m], right_index=True, left_index=True, how='left')['amount']

    mean_var_exp.dropna(axis=1, how='all').fillna(0).mean(axis=1)

def import_bank_extracts():
    be = pd.read_csv('bank_extract.csv', sep=';', encoding='latin-1')
    be['month'] = pd.DatetimeIndex(be["date de l'opération"]).month
    be['year'] = pd.DatetimeIndex(be["date de l'opération"]).month


def get_actual_vs_forecast(monthly_purchases, monthly_forecast, incomes):
    act_vs_for = {k: {"category": monthly_forecast[k][k].reset_index()['category'].tolist(),
                      "sub_category": monthly_forecast[k][k].reset_index()['sub_category'].tolist(),
                      "origin": monthly_forecast[k][k].reset_index()['origin'].tolist(),
                      "forecast": monthly_forecast[k][k].fillna(0).to_numpy(),
                      "expenses": np.where(monthly_forecast[k][k].reset_index()['origin'] == 'fix_exp',
                                           monthly_forecast[k][k].fillna(0).to_numpy(),
                                           -monthly_purchases[k]['amount'].fillna(0).to_numpy()),
                      "act_vs_for": np.round(
                          -(monthly_forecast[k][k].fillna(0).to_numpy() -
                            np.where(monthly_forecast[k][k].reset_index()['origin'] == 'fix_exp',
                                     monthly_forecast[k][k].fillna(0).to_numpy(),
                                     -monthly_purchases[k]['amount'].fillna(0).to_numpy())), 1),
                      "incomes": incomes[k].fillna(0).to_numpy()} for k in monthly_purchases.keys()}

    if len(act_vs_for.keys()) > 1:
        act_vs_for["All"] = {
            "category": monthly_forecast[list(act_vs_for.keys())[0]][list(act_vs_for.keys())[0]].reset_index()[
                'category'].tolist(),
            "sub_category": monthly_forecast[list(act_vs_for.keys())[0]][list(act_vs_for.keys())[0]].reset_index()[
                'sub_category'].tolist(),
            "origin": monthly_forecast[list(act_vs_for.keys())[0]][list(act_vs_for.keys())[0]].reset_index()[
                'origin'].tolist(),
            "forecast": np.nansum(np.array([monthly_forecast[k][k].fillna(0).to_numpy() for k in act_vs_for.keys()]),
                                  axis=0),
            "expenses": np.nansum(np.array([act_vs_for[k]["expenses"] for k in act_vs_for.keys()]), axis=0)}
        act_vs_for["All"]["act_vs_for"] = np.round(-(act_vs_for["All"]["forecast"] - act_vs_for["All"]["expenses"]), 1)
        act_vs_for["All"]["incomes"] = np.nansum(
            np.array([incomes[k].fillna(0).to_numpy() for k in list(act_vs_for.keys())[:-1]]), axis=0)

    return act_vs_for


def run(months, target='Indrani'):
    real_exp, db, forecast_glabais, forecast_indrani, incomes = import_data()
    dv.incomes = incomes
    # list_of_index = list()
    # for m in month_names:
    #    list_of_index += list(real_exp[m][aggreg].dropna().set_index(aggreg).index.unique())
    # list_of_index += list(forecast_glabais[aggreg].dropna().set_index(aggreg).index.unique())
    # list_of_index += list(forecast_indrani[aggreg].dropna().set_index(aggreg).index.unique())
    # list_of_index = [l for i, l in enumerate(list_of_index) if l not in list_of_index[:i]]

    list_of_index = list()
    for k in db.columns[:-1]:
        for e in db[k].dropna():
            list_of_index.append((k, e))

    aggreg = ['category', 'sub_category']
    base = pd.DataFrame(list_of_index, columns=aggreg).set_index(aggreg)
    base = base.merge(forecast_indrani[['origin'] + aggreg].groupby(aggreg).first(),
                      right_index=True, left_index=True, how='left')
    base['origin'].fillna('var_exp', inplace=True)
    base = base.reset_index().set_index(aggreg + ['origin'])

    dv.monthly_purchases = {m: base.merge(real_exp[m].set_index(aggreg),
                                          right_index=True, left_index=True, how='left').reset_index() for m in months}
    monthly_purchases = {m: base.merge(real_exp[m].groupby(aggreg).sum(),
                                       right_index=True, left_index=True, how='left') for m in months}

    forecast = forecast_indrani if target == 'Indrani' else forecast_glabais
    monthly_forecast = {m: base.merge(forecast[[m] + aggreg].groupby(aggreg).sum(),
                                      right_index=True, left_index=True, how='left') for m in months}
    dv.monthly_forecast = monthly_forecast

    return get_actual_vs_forecast(monthly_purchases, monthly_forecast, incomes)
    # for k in act_vs_for.keys():
    #    print(np.round(pd.DataFrame(act_vs_for[k][c][act_vs_for[k]['act_vs_for'] != 0] for c in [‹ë]), 1), '\n')
    #    print(f"actual vs forecast {k}: \n{np.round(act_vs_for[k]['act_vs_for'].sum(), 1)} Euros\n")
    #    print(f"Expenses per type for {k}: \n{np.round(act_vs_for[k].groupby(['origin']).sum(), 1)}")
    #    #print(f"Expenses per type for {k}: \n{np.round((act_vs_for[k].groupby(['origin']).sum().sum() + incomes[k].sum()).T, 1)}")


app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           )
app.title = "Family Abbate - Expense Analysis"

def get_balance_frame():
    return html.Div(dash_table.DataTable(id='global_balance_table',
                         style_cell={
                             'minWidth': '125px', 'width': '125px',
                             'maxWidth': '125px'}))

def get_category_frame():
    return dbc.Row(
        children=[
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader('By catergory'),
                        dbc.CardBody(
                            children=[
                                dcc.Graph(id='act_vs_for_fig', style={'height': '400px'}),
                            ],
                        ),
                    ],
                    style={"width": "100%", 'height': "500px"},
                ),
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Value table by category"),
                        dbc.CardBody(
                            children=[
                                dash_table.DataTable(id='total_act_vs_for',
                                                     style_cell={
                                                         'minWidth': '125px', 'width': '125px',
                                                         'maxWidth': '125px'}),
                                html.Br(),

                                dash_table.DataTable(id='act_vs_for_table',
                                                     style_table={'height': '310px',
                                                                  'overflowY': 'auto',
                                                                  'overflowX': 'auto'},
                                                     style_cell={
                                                         'minWidth': '125px', 'width': '125px',
                                                         'maxWidth': '125px',
                                                         'textOverflow': 'ellipsis',
                                                         'overflow': 'hidden'}
                                                     )],
                        )
                    ],
                    style={"width": "100%", 'height': "500px"},
                ),
            ),
        ]
    )


def get_sub_category_frame():
    return dbc.Row(children=[
        dbc.Card(
            [
                dbc.CardHeader("Sub-catergory focus"),
                dbc.CardBody(children=[
                    dcc.Dropdown(
                        placeholder="Select category to focus on",
                        id='cat_sel_dd',
                        searchable=True,
                        clearable=True,
                        style={'width': '400px'}
                    ),
                    html.Br(),
                    dbc.Row(children=[
                        dbc.Card(children=[
                            dbc.CardBody(
                                dcc.Graph(id='sub_cat_focus_fig', style={'height': '400px'}),
                            )], style={"width": "50%", 'height': "500px"},),
                        dbc.Card(children=[
                            dbc.CardBody(children=[
                                dash_table.DataTable(id='total_sub_cat_focus',
                                                     style_cell={
                                                         'minWidth': '125px', 'width': '125px',
                                                         'maxWidth': '125px'}),
                                html.Br(),
                                dash_table.DataTable(id='cat_focus_table',
                                                     style_table={'overflowY': 'auto',
                                                                  'overflowX': 'auto'},
                                                     style_cell={
                                                         'minWidth': '125px', 'width': '125px',
                                                         'maxWidth': '125px',
                                                         'textOverflow': 'ellipsis',
                                                         'overflow': 'hidden'}
                                                     ),
                                html.Br(),
                                dcc.Dropdown(
                                    placeholder="Select sub-category to focus on",
                                    id='sub_cat_focus_dd',
                                    searchable=True,
                                    clearable=True,
                                    style={'width': '400px'}),
                                html.Br(),
                                dash_table.DataTable(id='sub_cat_focus_table',
                                                     style_table={'height': '300px',
                                                                  'overflowY': 'auto',
                                                                  'overflowX': 'auto'},
                                                     style_cell={
                                                         'minWidth': '125px', 'width': '125px',
                                                         'maxWidth': '125px',
                                                         'textOverflow': 'ellipsis',
                                                         'overflow': 'hidden'}
                                                     ),
                                ]),
                            ], style={"width": "50%", 'height': "500px"}),
                        ])
                    ])
                ])
            ])

app.layout = html.Div(children=[
    html.H1(children='Actual vs forecast'),
    html.Br(),
    dcc.Checklist(id='hidden_check', options=['ready'], style={'display':'none'}),
    html.Div([
        dcc.Dropdown(
            placeholder="Select month(s) to analyze",
            id='act_vs_for_dd',
            multi=True,
            searchable=True,
            clearable=True,
            style={'width': '400px', 'display': 'inline-block'}
        ),
        dcc.Checklist(
            id='fix_exp_inc',
            options=['Include fixed expenses'],
            style={'margin-left': '30px', 'display': 'inline-block'},
        ),
    ], style={'width': '50%'}),
    html.Br(),
    get_balance_frame(),
    html.Br(),
    get_category_frame(),
    html.Br(),
    get_sub_category_frame(),
])




@ app.callback(
    Output('hidden_check', 'value'),
    Input('act_vs_for_dd', 'value'),
    Input('fix_exp_inc', 'value')
)
def update_actual_vs_foreast_df(value, fix_exp_inc):
    keys = ['category', 'sub_category', 'origin', 'forecast', 'expenses', 'act_vs_for']
    if (value is not None) & (value != []):
        act_vs_for = run(value, target)
        k = 'All' if len(list(act_vs_for.keys())) > 1 else value[0]
        dv.df_full = pd.DataFrame({e: act_vs_for[k][e] for e in keys})
        if fix_exp_inc != ['Include fixed expenses']:
            dv.df = dv.df_full[dv.df_full['origin'] == 'var_exp'].reset_index(drop=True).copy()
        else:
            dv.df = dv.df_full.copy()
        dv.act_vs_for = act_vs_for.copy()
        return ['ready']
    return []



@app.callback(
    Output('act_vs_for_fig', 'figure'),
    Input('hidden_check', 'value')
)
def update_actual_vs_foreast_fig(check):
    if 'ready' in check:
        df = dv.df.copy()
        df['act_vs_for'] = df['act_vs_for'] * -1
        fig = px.bar((pd.melt(df, id_vars=['category'], value_vars=[
            "forecast", 'act_vs_for', "expenses"]).groupby(['category', 'variable']).sum() * -1).reset_index(),
                     x='category', color='variable', y='value', barmode="group")
        return fig
    return go.Figure()


@ app.callback(
    Output('act_vs_for_table', 'columns'),
    Output('act_vs_for_table', 'data'),
    Output('total_act_vs_for', 'columns'),
    Output('total_act_vs_for', 'data'),
    Input('hidden_check', 'value')
)

def update_actual_vs_foreast_table(check):
    if 'ready' in check:
        df = dv.df.copy()
        data_col = ([{'id': k, 'name': k} for k in [t for t in table_col if t != 'sub_category']])
        data = df.groupby(['category', 'origin']).sum().round(2).reset_index().sort_values('forecast')
        data_out = [dict(**{c: data[c].iloc[i] for c in data.columns}) for i in range(data.shape[0])]

        total_col = ([{'id': k, 'name': k} for k in ['forecast', 'expenses', 'act_vs_for']])
        total = [dict(**{c: np.round(data.sum()[c], 1) for c in ['forecast', 'expenses', 'act_vs_for']})]
        return data_col, data_out, total_col, total
    return None, [], None, []

@app.callback(
    Output('sub_cat_focus_fig', 'figure'),
    Input('hidden_check', 'value'),
    Input("cat_sel_dd", "value")
)
def update_actual_vs_foreast_sub_cat_fig(check,sub_cat):
    if ('ready' in check) & (sub_cat is not None):
        df = dv.df[dv.df['category'] == sub_cat].copy()

        fig = px.bar((pd.melt(df, id_vars=['sub_category'], value_vars=[
            "forecast", 'act_vs_for', "expenses"]).groupby(['sub_category', 'variable']).sum() * -1).reset_index(),
                     x='sub_category', color='variable', y='value', barmode="group")
        return fig
    return go.Figure()

@app.callback(
    Output('cat_focus_table', 'columns'),
    Output('cat_focus_table', 'data'),
    Output('total_sub_cat_focus', 'columns'),
    Output('total_sub_cat_focus', 'data'),
    Input('hidden_check', 'value'),
    Input("cat_sel_dd", "value")
)
def update_actual_vs_foreast_sub_cat_table(check, sub_cat):
    if ('ready' in check) & (sub_cat is not None):
        df = dv.df[dv.df['category'] == sub_cat].copy()
        data_col = ([{'id': k, 'name': k} for k in [t for t in table_col if t != 'category']])
        data = df.groupby(['sub_category', 'origin']).sum().round(2).reset_index().sort_values("forecast")
        data_out = [dict(**{c: data[c].iloc[i] for c in data.columns}) for i in range(data.shape[0])]

        total_col = ([{'id': k, 'name': k} for k in ['forecast', 'expenses', 'act_vs_for']])
        total = [dict(**{c: np.round(data.sum()[c], 1) for c in ['forecast', 'expenses', 'act_vs_for']})]
        return data_col, data_out, total_col, total
    return None, [], None, []

@app.callback(
    Output('sub_cat_focus_table', 'columns'),
    Output('sub_cat_focus_table', 'data'),
    Input('hidden_check', 'value'),
    Input("sub_cat_focus_dd", "value"),
    State("cat_sel_dd", "value")
)
def update_sub_cat_zoom_table(check, sub_cat, cat):
    if ('ready' in check) & (sub_cat is not None):
        monthly_purchases = dv.monthly_purchases.copy()
        conc = pd.DataFrame()
        for k in monthly_purchases.keys():
            conc = pd.concat([conc, monthly_purchases[k]], axis=0)

        df = conc[(conc['category'] == cat) & (conc['sub_category'] == sub_cat)].copy()
        df['comment'] = df['comment'].fillna("")
        data = (df.groupby(['sub_category', 'origin', 'comment']).sum()*-1).round(2).reset_index()
        data_col = ([{'id': k, 'name': k} for k in [t for t in data.columns if t != 'category']])
        data_out = [dict(**{c: data[c].iloc[i] for c in data.columns}) for i in range(data.shape[0])]
        return data_col, data_out
    return None, []

@app.callback(
    Output('global_balance_table', 'columns'),
    Output('global_balance_table', 'data'),
    Input('hidden_check', 'value'),
    State('act_vs_for_dd', 'value'),
)
def update_global_balance_table(check, months):
    if ('ready' in check):
        df = dv.df.copy()
        data = df.round(1).sum()
        inc = np.sum(np.array([dv.incomes[m].sum() for m in months]))
        forecast, expenses, act_vs_for = dv.df_full.sum()[['forecast', 'expenses', 'act_vs_for']]
        expenses = dv.df_full.sum()['expenses']
        balance = [forecast + inc, expenses + inc, act_vs_for]
        data_col = ([{'id': k, 'name': k} for k in ['forecast balance','expenses balance', 'act_vs_for']])
        data_out = [dict(**{c: np.round(balance[i], 1) for i, c in enumerate(['forecast balance','expenses balance',
                                                                              'act_vs_for'])})]
        return data_col, data_out
    return None, []


@app.callback(
    Output("act_vs_for_dd", "options"),
    Input("act_vs_for_dd", "search_value"),
    State("act_vs_for_dd", "value")
)
def update_act_vs_for_dd_options(search_value, value):
    if not search_value:
        return [o for o in act_vs_dd_options]
    return [o for o in act_vs_dd_options if search_value in o["label"] or o["value"] in (value or [])]

@app.callback(
    Output("cat_sel_dd", "options"),
    Output("cat_sel_dd", "value"),
    Input('hidden_check', 'value'),
    Input("cat_sel_dd", "search_value"),
    State("cat_sel_dd", "value")
)
def update_sub_cat_dd_options(check, search_value, value):
    value_out = value if value is not None else 'alimentation'
    if 'ready' not in check:
        return [], None
    elif not search_value:
        return [o for o in dv.df['category'].unique()], value_out
    options_out = [o for o in dv.df['category'].unique() if search_value in o["label"] or o["value"] in (value or [])]
    return options_out, value_out

@app.callback(
    Output("sub_cat_focus_dd", "options"),
    Input('hidden_check', 'value'),
    Input("sub_cat_focus_dd", "search_value"),
    Input("cat_sel_dd", "value"),
    State("sub_cat_focus_dd", "value"),
)
def update_sub_cat_zoom_dd_options(check, search_value, cat_sel, value):
    if 'ready' not in check:
        return []
    else:
        list_of_options = dv.df[dv.df['category'] == cat_sel]['sub_category'].unique()
        if not search_value:
            return [o for o in list_of_options]
        options_out = [o for o in list_of_options if search_value in o["label"] or o["value"] in (value or [])]
        return options_out

if __name__ == '__main__':
    app.run_server(debug=True)
