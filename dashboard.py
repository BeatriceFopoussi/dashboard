# Import des librairies
import streamlit as st
from PIL import Image
import shap
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from shap.plots._waterfall import waterfall_legacy
from pandas import DataFrame


API_URL  = "https://dashboard-2-6eo5.onrender.com/"

# Chargement des dataset
data_train = pd.read_csv('train_df.csv', nrows=100)
data_test = pd.read_csv('test_df.csv', nrows= 100)

def print_json(data, indent=0):
    for key, value in data.items():
        print(" " * indent + f"{key}: {value}")
        if isinstance(value, dict):  # Si la valeur est un dictionnaire, parcourir récursivement
            print_json(value, indent + 4)
        elif isinstance(value, list):  # Si la valeur est une liste, afficher chaque élément
            for item in value:
                if isinstance(item, dict):
                    print_json(item, indent + 4)
                else:
                    print(" " * (indent + 4) + str(item))

# Afficher les clés et valeurs
#

def minmax_scale(df, scaler):
    """Preprocessing du dataframe en paramètre avec le scaler renseigné.
    :param: df, scaler (str).
    :return: df_scaled.
    """
    cols = df.select_dtypes(['float64']).columns
    df_scaled = df.copy()
    if scaler == 'minmax':
        scal = MinMaxScaler()
    else:
        scal = StandardScaler()

    df_scaled[cols] = scal.fit_transform(df[cols])
    return df_scaled


data_train_mm = minmax_scale(data_train, 'minmax')
data_test_mm = minmax_scale(data_test, 'minmax')


def get_prediction(client_id):
    """Récupère la probabilité de défaut du client via l'API.
    :param: client_id (int).
    :return: probabilité de défaut (float) et la décision (str)
    """
    url_get_pred = API_URL + "prediction/" + str(client_id)
    response = requests.get(url_get_pred)
    proba_default = round(float(response.content), 3)
    best_threshold = 0.54
    if proba_default >= best_threshold:
        decision = "Refusé"
    else:
        decision = "Accordé"

    return proba_default, decision


def jauge_score(proba):
    """Construit une jauge indiquant le score du client.
    :param: proba (float).
    """
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=proba * 100,
        mode="gauge+number+delta",
        title={'text': "Jauge de score"},
        delta={'reference': 54},
        gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
               'bar': {'color': "MidnightBlue"},
               'steps': [
                   {'range': [0, 20], 'color': "Green"},
                   {'range': [20, 45], 'color': "LimeGreen"},
                   {'range': [45, 54], 'color': "Orange"},
                   {'range': [54, 100], 'color': "Red"}],
               'threshold': {'line': {'color': "brown", 'width': 4}, 'thickness': 1, 'value': 54}}))

    st.plotly_chart(fig)


def get_shap_val_local(client_id = 0):
    """Récupère les shap value du client via l'API pour une interprétation locale.
    :param: client_id (int).
    :return: shap_local
    """
    url_get_shap_local = API_URL + "shaplocal/" + str(client_id)
    print('valeur url', url_get_shap_local)

    response = requests.get(url_get_shap_local)
    res = json.loads(response.content)
    print_json(res)
    print('valeur de  res',res) 
    data = res['data'][0]
    data = np.array(data, dtype='float')
    feature_names = res['feature_names']

    values = res['shap_values'][0]
    values = np.array(values, dtype='float')
    #values = values[:,1]
    base_values = res['base_value'][0]

    explanation = shap.Explanation(values=values,
                                   base_values=base_values,
                                   data=data,
                                   feature_names=feature_names)

    #return explanation
    
    return {
        'values': values,
        'base_values': base_values,
        'data': data,
        'feature_names': feature_names
    }

def get_shap_val_global():
    """Récupère les shap value du client via l'API pour une interprétation locale.
    :param: client_id (int).
    :return: shap_local
    """
    url_get_shap_global = API_URL + "shap/" 

    response = requests.get(url_get_shap_global)
    data = json.loads(response.content)

    return data
    
def df_voisins(client_id):
    url_client_similaire = API_URL + 'clients_similaires/' + str(client_id)

    response = requests.get(url_client_similaire)
    data = json.loads(response.content)

    return DataFrame(data)


def distribution(feature, id_client, df):
    """Affiche la distribution de la feature indiquée en paramètre et ce pour les 2 target.
    Affiche également la position du client dont l'ID est renseigné en paramètre dans ce graphique.
    :param: feature (str), id_client (int), df.
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.hist(df[df['TARGET'] == 0][feature], bins=30, label='accordé')
    ax.hist(df[df['TARGET'] == 1][feature], bins=30, label='refusé')

    observation_value = data_test.loc[data_test['SK_ID_CURR'] == id_client][feature].values
    ax.axvline(observation_value, color='green', linestyle='dashed', linewidth=2, label='Client')

    ax.set_xlabel('Valeur de la feature', fontsize=20)
    ax.set_ylabel('Nombre d\'occurrences', fontsize=20)
    ax.set_title(f'Histogramme de la feature "{feature}" pour les cibles accordé et refusé', fontsize=22)
    ax.legend(fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    st.pyplot(fig)


def scatter(id_client, feature_x, feature_y, df):
    """Affiche le nuage de points de la feature_y en focntion de la feature_x.
    Affiche également la position du client dont l'ID est renseigné en paramètre dans ce graphique.
    :param: id_client (int), feature_x (str), feature_y (str), df.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    data_accord = df[df['TARGET'] == 0]
    data_refus = df[df['TARGET'] == 1]
    ax.scatter(data_accord[feature_x], data_accord[feature_y], color='blue',
               alpha=0.5, label='accordé')
    ax.scatter(data_refus[feature_x], data_refus[feature_y], color='red',
               alpha=0.5, label='refusé')

    data_client = data_test.loc[data_test['SK_ID_CURR'] == id_client]
    observation_x = data_client[feature_x]
    observation_y = data_client[feature_y]
    ax.scatter(observation_x, observation_y, marker='*', s=200, color='black', label='Client')

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title(f'Analyse bivariée des caractéristiques sélectionnées')
    ax.legend()

    st.pyplot(fig)


def boxplot_graph(id_client, feat, df_vois):
    """Affiche les boxplot des variables renseignéees en paramètre pour chaque target.
    Affiche également la position du client dont l'ID est renseigné en paramètre dans ce graphique.
    Affiche les 10 plus proches voisins du client sur les boxplot.
    :param: id_client (int), feat (str), df_vois.
    """
    df_box = data_train_mm.melt(id_vars=['TARGET'], value_vars=feat,
                                var_name="variables", value_name="values")
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(data=df_box, x='variables', y='values', hue='TARGET', ax=ax)


    df_voisins_scaled = minmax_scale(df_vois, 'minmax')
    df_voisins_box = df_voisins_scaled.melt(id_vars=['TARGET'], value_vars=feat,
                                            var_name="var", value_name="val")
    sns.swarmplot(data=df_voisins_box, x='var', y='val', hue='TARGET', size=8,
                  palette=['green', 'red'], ax=ax)

    data_client = data_test_mm.loc[data_test['SK_ID_CURR'] == id_client][feat]
    categories = ax.get_xticks()
    for cat in categories:
        plt.scatter(cat, data_client.iloc[:, cat], marker='*', s=250, color='blueviolet', label='Client')

    ax.set_title(f'Boxplot des caractéristiques sélectionnées')
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) < 8:
        ax.legend(handles[:4], ['Accordé', 'Refusé', 'Voisins', 'Client'])
    else:
        ax.legend(handles[:5], ['Accordé', 'Refusé', 'Voisins (accordés)', 'Voisins (refusés)', 'Client'])

    st.pyplot(fig)


# Titre de la page
st.set_page_config(page_title="Dashboard Prêt à dépenser", layout="wide")

# Sidebar
with st.sidebar:
    logo = Image.open('image.jpg')
    st.image(logo, width=200)
    # Page selection
    page = st.selectbox('Navigation', ["Home", "Information du client", "Interprétation locale",
                                               "Interprétation globale"])

    # ID Selection
    st.markdown("""---""")

    list_id_client = list(data_test['SK_ID_CURR'])
    list_id_client.insert(0, '<Select>')
    id_client_dash = st.selectbox("ID Client", list_id_client)
    st.write('Vous avez choisi le client ID : '+str(id_client_dash))

    st.markdown("""---""")
    st.write("Created")


if page == "Home":
    st.markdown(
    "<h1 style='color: darkblue;'>Dashboard Prêt à dépenser - Home Page</h1>",
    unsafe_allow_html=True
)
    #st.title("Dashboard Prêt à dépenser - Home Page")
    st.markdown("Ce site contient un dashboard interactif permettant d'expliquer aux clients les raisons\n"
                "d'approbation ou refus de leur demande de crédit.\n"
                
                "\nLes prédictions sont calculées à partir d'un algorithme d'apprentissage automatique, "
                "préalablement entraîné. Il s'agit d'un modèle *Light GBM* (Light Gradient Boosting Machine). "
                "Les données utilisées sont disponibles [ici](https://www.kaggle.com/c/home-credit-default-risk/data). "
                "Lors du déploiement, un échantillon de ces données a été utilisé.\n"
                
                "\nLe dashboard est composé de plusieurs pages :\n"
                "- **Information du client**: Vous pouvez y retrouver toutes les informations relatives au client "
                "selectionné dans la colonne de gauche, ainsi que le résultat de sa demande de crédit. "
                "Je vous invite à accéder à cette page afin de commencer.\n"
                "- **Interprétation locale**: Vous pouvez y retrouver quelles caractéritiques du client ont le plus "
                "influençé le choix d'approbation ou refus de la demande de crédit.\n"
                "- **Intérprétation globale**: Vous pouvez y retrouver notamment des comparaisons du client avec "
                "les autres clients de la base de données ainsi qu'avec des clients similaires.")


if page == "Information du client":
    st.title("Dashboard Prêt à dépenser - Page Information du client")

    st.write("Cliquez sur le bouton ci-dessous pour commencer l'analyse de la demande :")
    button_start = st.button("Statut de la demande")
    if button_start:
        if id_client_dash != '<Select>':
            # Calcul des prédictions et affichage des résultats
            st.markdown("RÉSULTAT DE LA DEMANDE")
            probability, decision = get_prediction(id_client_dash)

            if decision == 'Accordé':
                st.success("Crédit accordé")
            else:
                st.error("Crédit refusé")

            # Affichage de la jauge
            jauge_score(probability)

    # Affichage des informations client
    with st.expander("Afficher les informations du client", expanded=False):
        st.info("Voici les informations du client:")
        st.write(pd.DataFrame(data_test.loc[data_test['SK_ID_CURR'] == id_client_dash]))


if page == "Interprétation locale":
    st.title("Dashboard Prêt à dépenser - Page Interprétation locale")

    locale = st.checkbox("Interprétation locale")
    if locale:
        st.info("Interprétation locale de la prédiction")
        explanation = get_shap_val_local(id_client_dash)
        nb_features = st.slider('Nombre de variables à visualiser', 0, 20, 10)

        # Affichage du waterfall plot : shap local
        fig = waterfall_legacy(
            1.0,
            explanation['values'],
            explanation['data'],
            explanation['feature_names'],
            max_display=nb_features,
            show=False)
        # fig = shap.waterfall_plot(explanation, max_display=nb_features, show=False)
        #fig = plt.figure()
        #fig.add_axes(ax)
        #fig.sca(ax)
        # fig = st_shap(fig)
        st.pyplot(fig)
        # plt.show()


        with st.expander("Explication du graphique", expanded=False):
            st.caption("Ici sont affichées les caractéristiques influençant de manière locale la décision. "
                       "C'est-à-dire que ce sont les caractéristiques qui ont influençé la décision pour ce client "
                       "en particulier.")


if page == "Interprétation globale":
    st.title("Dashboard Prêt à dépenser - Page Interprétation globale")
    # Création du dataframe de voisins similaires
    data_voisins = df_voisins(id_client_dash)
    print(data_voisins)

    globale = st.checkbox("Importance globale")
    if globale:
        st.info("Importance globale")
        shap_values = get_shap_val_global()
        data_test_std = minmax_scale(data_test.drop('SK_ID_CURR', axis=1), 'std')
        nb_features = st.slider('Nombre de variables à visualiser', 0, 20, 10)
        fig, ax = plt.subplots()
       # Vérifiez et corrigez la structure des shap_values
    shap_values_matrix = np.array(shap_values['shap_values_1'], dtype='float')
    if len(shap_values_matrix.shape) == 1:  # Si c'est un vecteur
        shap_values_matrix = shap_values_matrix.reshape(1, -1)  # Convertir en matrice 2D

    # Affichage du summary plot : shap global
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values_matrix, data_test_std, plot_type='bar', max_display=nb_features)
    st.pyplot(fig)

    with st.expander("Explication du graphique", expanded=False):
        st.caption("Ici sont affichées les caractéristiques influençant de manière globale la décision.")
    distrib = st.checkbox("Comparaison des distributions")
    if distrib:
        st.info("Comparaison des distributions de plusieurs variables de l'ensemble de données")
        # Possibilité de choisir de comparer le client sur l'ensemble de données ou sur un groupe de clients similaires
        distrib_compa = st.radio("Choisissez un type de comparaison :", ('Tous', 'Clients similaires'), key='distrib')

        list_features = list(data_train.columns)
        list_features.remove('SK_ID_CURR')
        # Affichage des distributions des variables renseignées
        with st.spinner(text="Chargement des graphiques..."):
            col1, col2 = st.columns(2)
            with col1:
                feature1 = st.selectbox("Choisissez une caractéristique", list_features,
                                        index=list_features.index('AMT_CREDIT'))
                if distrib_compa == 'Tous':
                    distribution(feature1, id_client_dash, data_train)
                else:
                    distribution(feature1, id_client_dash, data_voisins)
            with col2:
                feature2 = st.selectbox("Choisissez une caractéristique", list_features,
                                        index=list_features.index('EXT_SOURCE_2'))
                if distrib_compa == 'Tous':
                    distribution(feature2, id_client_dash, data_train)
                else:
                    distribution(feature2, id_client_dash, data_voisins)

            with st.expander("Explication des distributions", expanded=False):
                st.caption("Vous pouvez sélectionner la caractéristique dont vous souhaitez observer la distribution. "
                           "En bleu est affichée la distribution des clients qui ne sont pas considérés en défaut et "
                           "dont le prêt est donc jugé comme accordé. En orange, à l'inverse, est affichée la "
                           "distribution des clients considérés comme faisant défaut et dont le prêt leur est refusé. "
                           "La ligne pointillée verte indique où se situe le client par rapport aux autres clients.")

    bivar = st.checkbox("Analyse bivariée")
    if bivar:
        st.info("Analyse bivariée")
        # Possibilité de choisir de comparer le client sur l'ensemble de données ou sur un groupe de clients similaires
        bivar_compa = st.radio("Choisissez un type de comparaison :", ('Tous', 'Clients similaires'), key='bivar')

        list_features = list(data_train.columns)
        list_features.remove('SK_ID_CURR')
        list_features.insert(0, '<Select>')

        # Selection des features à afficher
        c1, c2 = st.columns(2)
        with c1:
            feat1 = st.selectbox("Sélectionner une caractéristique X ", list_features)
        with c2:
            feat2 = st.selectbox("Sélectionner une caractéristique Y", list_features)
        # Affichage des nuages de points de la feature 2 en fonction de la feature 1
        if (feat1 != '<Select>') & (feat2 != '<Select>'):
            if bivar_compa == 'Tous':
                scatter(id_client_dash, feat1, feat2, data_train)
            else:
                scatter(id_client_dash, feat1, feat2, data_voisins)
            with st.expander("Explication des scatter plot", expanded=False):
                st.caption("Vous pouvez ici afficher une caractéristique en fonction d'une autre. "
                           "En bleu sont indiqués les clients ne faisant pas défaut et dont le prêt est jugé comme "
                           "accordé. En rouge, sont indiqués les clients faisant défaut et dont le prêt est jugé "
                           "comme refusé. L'étoile noire correspond au client et permet donc de le situer par rapport "
                           "à la base de données clients.")

    boxplot = st.checkbox("Analyse des boxplot")
    if boxplot:
        st.info("Comparaison des distributions de plusieurs variables de l'ensemble de données à l'aide de boxplot.")

        feat_quanti = data_train.select_dtypes(['float64']).columns
        # Selection des features à afficher
        features = st.multiselect("Sélectionnez les caractéristiques à visualiser: ",
                                  sorted(feat_quanti),
                                  default=['AMT_CREDIT', 'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3'])

        # Affichage des boxplot
        boxplot_graph(id_client_dash, features, data_voisins)
        with st.expander("Explication des boxplot", expanded=False):
            st.caption("Les boxplot permettent d'observer les distributions des variables renseignées. "
                       "Une étoile violette représente le client. Ses plus proches voisins sont également "
                       "renseignés sous forme de points de couleurs (rouge pour ceux étant qualifiés comme "
                       "étant en défaut et vert pour les autres).")