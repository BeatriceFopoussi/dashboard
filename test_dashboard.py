url_get_prediction = "https://dashboard-2-6eo5.onrender.com/prediction/192535"
response = requests.get(url_get_prediction)

if response.status_code == 200:
    print("Prediction received:", response.json())
else:
    print(f"Error {response.status_code}: Unable to get prediction")



fig, ax = plt.subplots()
shap.summary_plot(shap_values_matrix, data_test_std, plot_type='bar', max_display=nb_features)
st.pyplot(fig)

# Convertir en matrice si nécessaire
shap_values_matrix = np.array(shap_values['shap_values_1'], dtype='float')
if len(shap_values_matrix.shape) == 1:  # Si c'est un vecteur
    shap_values_matrix = shap_values_matrix.reshape(1, -1)  # Convertir en matrice 2D



    globale = st.checkbox("Importance globale")
if globale:
    st.info("Importance globale")
    shap_values = get_shap_val_global()
    data_test_std = minmax_scale(data_test.drop('SK_ID_CURR', axis=1), 'std')
    nb_features = st.slider('Nombre de variables à visualiser', 0, 20, 10)

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