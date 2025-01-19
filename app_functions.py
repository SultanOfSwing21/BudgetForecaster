import streamlit as st
import uuid

def _editable_dataframe(df, df_key):
    """
    Fonction pour afficher et éditer un DataFrame en fonction de sa clé dans `st.session_state`.
    """
    # Initialiser les données dans st.session_state si elles n'existent pas
    if f"original_df_{df_key}" not in st.session_state:
        st.session_state[f"original_df_{df_key}"] = df.copy()
    if f"editable_df_{df_key}" not in st.session_state:
        st.session_state[f"editable_df_{df_key}"] = st.session_state[f"original_df_{df_key}"].copy()
    if f"keg_{df_key}" not in st.session_state:
        st.session_state[f"keg_{df_key}"] = str(uuid.uuid4())

    # Initialiser des drapeaux pour les confirmations
    if f"show_confirm_button_{df_key}" not in st.session_state:
        st.session_state[f"show_confirm_button_{df_key}"] = False
    if f"confirm_{df_key}" not in st.session_state:
        st.session_state[f"confirm_{df_key}"] = False

    # Afficher et permettre l'édition du DataFrame
    editable_df = st.data_editor(
        st.session_state[f"editable_df_{df_key}"],
        use_container_width=True,
        num_rows="dynamic",
        key=st.session_state[f"keg_{df_key}"]
    )

    # Boutons pour sauvegarder ou réinitialiser le DataFrame
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sauvegarder les modifications", key=f"save_{df_key}"):
            st.session_state[f"editable_df_{df_key}"] = editable_df.copy()
            st.session_state[f"original_df_{df_key}"] = st.session_state[f"editable_df_{df_key}"].copy()
            st.success("Modifications sauvegardées dans les données originales.")

    with col2:
        if st.button("Rafraîchir le DataFrame", key=f"refresh_{df_key}"):
            st.session_state[f"show_confirm_button_{df_key}"] = True

    # Afficher la modale si nécessaire
    if st.session_state[f"show_confirm_button_{df_key}"]:
        confirm_action(
            action_type=f"show_confirm_button_{df_key}",
            modal_title="Confirmer le rafraîchissement",
            action_message="Êtes-vous sûr de vouloir rafraîchir les données ?",
            action_button_text="Oui, rafraîchir",
            action_callback=lambda: refresh_data(df_key),
            df_key=df_key
        )

# Fonction commune pour gérer les confirmations
def confirm_action(action_type, modal_title, action_message, action_button_text, action_callback, df_key):
    """
    Affiche une modale de confirmation pour une action donnée (sauvegarde, rafraîchissement, etc.)
    """
    st.write(f"### {modal_title}")
    st.write(action_message)

    col1, col2 = st.columns(2)

    # Bouton pour confirmer l'action
    with col1:
        if st.button(action_button_text, key=f"{action_type}_confirm"):
            st.session_state[f"confirm_{df_key}"] = True  # Capture du clic

    # Bouton pour annuler l'action
    with col2:
        if st.button("Annuler", key=f"{action_type}_cancel"):
            st.session_state[f"show_confirm_button_{df_key}"] = False  # Fermer la modale

    # Si l'utilisateur a confirmé l'action, appeler le callback
    if st.session_state[f"confirm_{df_key}"]:
        action_callback()  # Appeler la fonction associée
        st.session_state[f"confirm_{df_key}"] = False  # Réinitialiser le clic
        st.session_state[f"show_confirm_button_{df_key}"] = False  # Fermer la modale après exécution
        st.rerun()


# Fonction de sauvegarde des modifications
def save_changes():
    st.session_state.original_df = st.session_state.editable_df.copy()
    st.success("Modifications sauvegardées dans les données originales.")


# Fonction de rafraîchissement des données
def refresh_data(df_key):
    """
    Réinitialise un DataFrame spécifique en fonction de sa clé.
    """
    st.session_state[f"editable_df_{df_key}"] = st.session_state[f"original_df_{df_key}"].copy()
    st.session_state[f"keg_{df_key}"] = str(uuid.uuid4())
    st.success("Les données ont été rafraîchies avec succès !")
