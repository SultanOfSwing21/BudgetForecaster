import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import os
from axa_bank_report_manager import import_bank_reports, get_base_of_information, receiver, account
import uuid

import app_functions as app_fun

# Fonction pour gérer la sauvegarde du DataFrame
def save_dataframe(df, filename='saved_dataframe.csv'):
    df.to_csv(filename, index=False)

# Fonction pour charger le DataFrame depuis un fichier
def load_dataframe(filename='saved_dataframe.csv'):
    return pd.read_csv(filename) if os.path.exists(filename) else None

if "df" not in locals():
    df = import_bank_reports()

if "all_manual_df" not in globals():
    real_exp = pd.read_excel('budget_forecaster.xlsx', sheet_name='Achats 2022', header=[0, 1])
    all_manual_df = pd.DataFrame()
    for month in np.unique([month for month, ind in real_exp.columns]):
        all_manual_df = pd.concat([all_manual_df, real_exp[month]], axis=0)


if "all_information" not in globals():
    account_inf = pd.read_excel('bank_reports/axa/axa_map.xlsx', sheet_name=None, header=[1])
    account_inf.pop("database")
    all_information = pd.DataFrame()
    for sh_name, sh_df in account_inf.items():
        all_information = pd.concat([all_information, sh_df], axis=0)


#test = get_base_of_information(df)
#test.merge(all_information, on=[receiver, account], how='left')

print("done")


# Menu latéral pour naviguer entre les pages (onglets)
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Page 1", "Page 2", "Page 3"],
        icons=["house", "table", "gear"],
        menu_icon="cast",
        default_index=0,
    )

# Page 1
if selected == "Page 1":
    st.title("Page 1")
    st.write("Contenu de la Page 1")
    app_fun._editable_dataframe(df=all_information.copy(), df_key="page1")


# Page 2 (contient le DataFrame éditable)
elif selected == "Page 2":
    st.title("Page 2 : DataFrame éditable")
    app_fun._editable_dataframe(df=df.copy(), df_key="page2")

#    st.write(st.session_state.editable_df)

# Page 3
elif selected == "Page 3":
    st.title("Page 3")
    st.write("Contenu de la Page 3")

