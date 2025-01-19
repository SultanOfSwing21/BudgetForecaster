import numpy as np
import pandas as pd
import os
from dataclasses import dataclass


@dataclass
class GlobalVariables:
    visa_conf_path = './conf/visa_conf'
    bank_rep = "./bank_reports/"
    visa_rep_name = "visa"

    rep_format = '.csv'
    rep_encoding = 'windows-1252'
    rep_sep = ';'
    rep_skiprows = 8


global_var = GlobalVariables
account = 'to_account'
subcat_name = "subcategory"
cat_name = "category"
receiver = "receiver"
amount = "amount"
acr = 'acr'
occ = 'occ'
occ_rec = 'occ_rec'

col_map_dict = {
    "axa":
        {
            "date de l'opération": "date",
            "montant": "amount",
            "description du type d'opération": "type",
            "contrepartie": "receiver",
            "compte bénéficiaire": "to_account",
            "nom du terminal": "terminal",
            "communication": "communication"
        },
    "crelan":
        {
            'Date': "date",
            'Montant': "amount",
            'Contrepartie': "receiver",
            'Compte contrepartie': "to_account",
            "Type d'opération": "type",
            'Communication': "communication"
        }

}

def import_bank_reports():

    df = pd.DataFrame()
    for bank in [k for k in os.listdir(global_var.bank_rep) if os.path.isdir(os.path.join(global_var.bank_rep, k))]:
        bank_path = os.path.join(global_var.bank_rep, bank, "ca")
        match bank:
            # case "axa":
            #     files = [k for k in os.listdir(bank_path) if global_var.rep_format in k]
            #     for file in files:
            #         temp = pd.read_csv(os.path.join(bank_path, file),
            #                            encoding=global_var.rep_encoding,
            #                            sep=global_var.rep_sep,
            #                            skiprows=range(global_var.rep_skiprows),
            #                            usecols=list(col_map_dict[bank].keys()))
            #         df.rename(columns=col_map_dict[bank], inplace=True)
            #
            #         df = pd.concat([df, temp], axis=0)

            case "crelan":
                files = [k for k in os.listdir(bank_path) if "xls" in k]
                for file in files:
                    temp = pd.read_excel(os.path.join(bank_path, file))
                    temp.rename(columns=col_map_dict[bank], inplace=True)
                    df = pd.concat([df, temp], axis=0)

    def parse_date(date_str):
        try:
            return pd.to_datetime(date_str, format="%d/%m/%Y")
        except ValueError:
            try:
                return pd.to_datetime(date_str, format="%Y-%m-%d")
            except ValueError:
                return pd.NaT

    df.date = df.date.apply(parse_date)
    df.year = df.date.apply(lambda x: x.year)
    df.month = df.date.apply(lambda x: x.month)
    df.year_month = df.year.astype(str) + '_' + df.month.astype(str)

    df.amount = df.amount.apply(lambda x: x.replace(',', '.') if isinstance(x, str) else x).astype(float)
    return df


# map = pd.read_excel(f'./{global_var.bank_rep}/axa_map.xlsx', skiprows=[0], sheet_name=None)
# map.pop('database')
#
# rows = df.account.where(df.account == -1, True)
# for sh, sheet in map.items():
#     print(sh)
#     cols = list(sheet.columns[:-3])
#     if account in cols:
#         rows = ((df.account.isin(sheet.account.to_list()) == False) & (pd.isna(df.account) == False)) & (rows == True)
#     missing = df[rows].account.drop_duplicates()
#     print(missing)
#
# rows = df.account.where(df.account == -1, True)
# for sh, sheet in map.items():
#     cols = list(sheet.columns[:-3])
#     if account not in cols:
#         print(sh)
#         rows = ((df.terminal.isin(sheet.terminal.to_list()) == False) & (pd.isna(df.terminal) == False)) & (
#                 rows == True)
#         missing_terminal = df[rows].terminal.drop_duplicates()
#         print(missing_terminal)
# df[df.account.isin(missing)][[receiver, account, amount]].groupby([receiver, account]).value_counts().sort_values(
#     ascending=False)


def extract_visa_reports(bank):
    from dataclasses import dataclass
    # import camelot
    # import yaml

    def generate_conf_class(bank_name, config):
        @dataclass
        class Conf:
            bank: str = bank_name
            first_row: int = None
            extract_method: str = None
            repository: str = None

            def __post_init__(self):
                # Générer le nom du répertoire en fonction du nom de la banque
                if self.repository is None:
                    self.repository = f"{global_var.bank_rep}/{self.bank}/{global_var.visa_rep_name}"

        return Conf(bank_name, **config)

    # load yaml conf
    with open(global_var.visa_conf_path, "r") as file:
        visa_conf = yaml.safe_load(file)[bank]

    cf = generate_conf_class(bank, visa_conf)

    df = pd.DataFrame()
    for file in os.listdir(cf.repository):
        print(file)
        temp = camelot.read_pdf(os.path.join(cf.repository, file), flavor=cf.extract_method)[0].df.iloc[cf.first_row:,
               :]
        df = pd.concat([df, temp], axis=0)


def build_new_ref(df):
    # Fonction pour extraire le mot le plus long d'une chaîne de caractères
    def longest_word(sentence):
        words = sentence.split()  # Divise la phrase en mots
        max_word = max(words, key=len)  # Trouve le mot le plus long
        return max_word

    df['key_test'] = df[receiver].apply(lambda x: longest_word(str(x)))
    new_list = []
    for k in [k.lower() for k in df.key_test if k != 'nan']:
        generic_list = [e for e in [i for i in df.receiver.to_list() if pd.isna(i) == False] if
                        (str(k) in e.lower()) & (pd.isna(e) == False)]
        new_list.append([k, len(generic_list), generic_list[0]])

    out = pd.DataFrame(new_list, columns=[acr, occ, 'names'])
    out.drop_duplicates().sort_values(by=occ, ascending=False).head(20)
    base = []
    for a in out.drop_duplicates().sort_values(by='occ', ascending=False).acr.dropna():
        print(a)
        base += [[a, e] for e in [i for i in df.receiver.to_list() if pd.isna(i) == False] if a in e]
    base = pd.DataFrame(base, columns=[acr, 'names']).drop_duplicates().reset_index(drop=True)

    occ_per_account = df[[receiver, account, amount]].groupby([receiver, account]).size().reset_index().rename(
        columns={0: 'occ'})
    occ_per_account = occ_per_account.sort_values('occ', ascending=False)
    #occ_per_account['cum_occ'] = occ_per_account['occ'].cumsum()

    occ_per_receiver = df[[receiver, amount]].groupby([receiver]).size().reset_index().rename(columns={0: 'occ_rec'})
    occ_per_receiver = occ_per_receiver.sort_values('occ_rec', ascending=False)
    #occ_per_receiver['cum_occ'] = occ_per_receiver['occ_rec'].cumsum()

    nb_amount_per_acc = df[[account, amount]].drop_duplicates().dropna().groupby([account, amount]).value_counts().reset_index()
    nb_amount_per_acc = nb_amount_per_acc[account].value_counts().reset_index()
    nb_amount_per_acc.rename(columns={"count": "nb_amount_per_account"}, inplace=True)


    temp = base.merge(occ_per_account, left_on='names', right_on=receiver, how='left')
    temp = temp.merge(occ_per_receiver, left_on=receiver,
                      right_on=receiver, how='left').drop('names', axis=1)
    temp.receiver = temp.receiver.apply(lambda x: str(x).lower())
    temp = temp.groupby([receiver, account]).sum().reset_index()
    temp = temp.sort_values(by=[receiver, occ_rec, occ], ascending=[True, False, False])
    list_of_accounts = temp.drop_duplicates()

    temp = temp.merge(nb_amount_per_acc, on=account, how='left')

    # calculate reccurence of amounts
    rec = base.merge(df[[receiver, account, amount]].groupby([receiver, account]).value_counts().reset_index(),
                     left_on='names', right_on=receiver, how='left').rename(columns={"count": "counts"})
    ind = [receiver, account]
    rec_bis = rec.set_index(ind)
    nb_amount_per_account = rec.groupby(ind)[amount].count()
    filt_accounts = nb_amount_per_account[nb_amount_per_account <= 3]
    recurrent_mov = rec_bis[(rec_bis.counts > 2) & (rec_bis.index.isin(filt_accounts.index))].drop('names', axis=1)

    test = rec.groupby(ind)[amount].count()
    recurrent_mov = pd.concat([recurrent_mov,
                               rec_bis[(rec_bis.counts > 10) & (rec_bis.index.isin(test.index))].drop('names', axis=1)],
                              axis=0)

    # identify counterpart account with only "one shot" payements
    ind = [account]
    rec_bis = rec.set_index(ind)
    nb_amount_per_account = rec.groupby(ind)[amount].count()
    filt_accounts = nb_amount_per_account[nb_amount_per_account == 1]
    one_shoters = rec_bis[(rec_bis.counts > 5) & (rec_bis.index.isin(filt_accounts.index))].drop('names', axis=1)

    filt_accounts = nb_amount_per_account[nb_amount_per_account > 5]
    others = rec_bis[(rec_bis.counts > 5) & (rec_bis.index.isin(filt_accounts.index))].drop('names', axis=1)

    mov_per_account = rec_bis[(rec_bis.counts > 5) & (rec_bis.index.isin(filt_accounts.index))].drop('names', axis=1)

    base.merge(df, left_on='names', right_on=receiver, how='left')[[receiver, account, amount]].drop_duplicates()
    base.merge(df, left_on='names', right_on=receiver, how='left')[[receiver, account, amount]].drop_duplicates()

    return base.merge(df, left_on='names', right_on=receiver, how='left')


def get_base_of_information(df):
    # Fonction pour extraire le mot le plus long d'une chaîne de caractères
    def longest_word(sentence):
        words = sentence.split()  # Divise la phrase en mots
        max_word = max(words, key=len)  # Trouve le mot le plus long
        return max_word

    df['key_test'] = df.receiver.apply(lambda x: longest_word(str(x)))
    new_list = []
    for k in [k for k in df.key_test if k != 'nan']:
        generic_list = [e for e in [i for i in df.receiver.to_list() if pd.isna(i) == False] if
                        (str(k) in e) & (pd.isna(e) == False)]
        new_list.append([k, len(generic_list), generic_list[0]])

    out = pd.DataFrame(new_list, columns=[acr, occ, 'names'])
    out.drop_duplicates().sort_values(by=occ, ascending=False).head(20)
    base = []
    for a in out.drop_duplicates().sort_values(by='occ', ascending=False).acr.dropna():
        print(a)
        base += [[a, e] for e in [i for i in df.receiver.to_list() if pd.isna(i) == False] if a in e]
    base = pd.DataFrame(base, columns=[acr, 'names']).drop_duplicates().reset_index(drop=True)

    occ_per_account = df[[receiver, account, amount]].groupby([receiver, account]).size().reset_index().rename(
        columns={0: 'occ'})
    occ_per_receiver = df[[receiver, amount]].groupby([receiver]).size().reset_index().rename(
        columns={0: 'occ_rec'})
    nb_amount_per_acc = df[[account, amount]].drop_duplicates().dropna().groupby(
        [account, amount]).value_counts().reset_index()
    nb_amount_per_acc = nb_amount_per_acc[account].value_counts().reset_index()
    nb_amount_per_acc.rename(columns={"count": "nb_amount_per_account"}, inplace=True)

    temp = base.merge(occ_per_account, left_on='names', right_on=receiver, how='left')
    temp = temp.merge(occ_per_receiver, left_on=receiver,
                      right_on=receiver, how='left').drop('names', axis=1)
    temp.receiver = temp.receiver.apply(lambda x: str(x).lower())
    temp = temp.groupby([receiver, account]).sum().reset_index()
    temp = temp.sort_values(by=[receiver, occ_rec, occ], ascending=[True, False, False])
    list_of_accounts = temp.drop_duplicates()

    temp = temp.merge(nb_amount_per_acc, on=account, how='left')
    return temp

data = import_bank_reports()
build_new_ref(data)

#
# df[cat_name] = np.nan
# df[subcat_name] = np.nan
# res = {}
# for m in df.year_month.unique():
#     temp = df[df.year_month == m]
#     for sh, sheet in map.items():
#         cols = list(sheet.columns[:-3])
#         temp2 = temp.drop([cat_name, subcat_name], axis=1).merge(sheet, on=cols, how='left')[
#             [cat_name, subcat_name]]
#         temp.category = np.where(pd.isna(temp.category), temp2.category, temp.category)
#         temp.subcategory = np.where(pd.isna(temp.subcategory), temp2.subcategory, temp.subcategory)
#     print(m, temp.shape[0], temp[[cat_name, subcat_name]].dropna().shape[0])
#     res.update({m: temp})
