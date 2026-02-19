import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# --- CONFIGURATION DES CHEMINS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
DATASET_PATH = os.path.join(PARENT_DIR, "spotify_churn_dataset.csv")
MODEL_PATH = os.path.join(CURRENT_DIR, "modele_sklearn.pkl")
ENC_SUB_PATH = os.path.join(CURRENT_DIR, "subscription_encoder.pkl")
SCALER_STD_PATH = os.path.join(CURRENT_DIR, "standardscaler.pkl")
SCALER_MM_PATH = os.path.join(CURRENT_DIR, "number_of_playlists_scaler.pkl")

# --- COLONNES D'ENTRAÎNEMENT ---
TRAINING_COLUMNS = [
    'subscription_type', 'avg_daily_minutes', 'number_of_playlists', 'skips_per_day', 
    'support_tickets', 'days_since_last_login', 
    'country_BR', 'country_CA', 'country_DE', 'country_FR', 'country_IN', 
    'country_PK', 'country_RU', 'country_UK', 'country_US', 
    'top_genre_Country', 'top_genre_Electronic', 'top_genre_Hip-Hop', 
    'top_genre_Jazz', 'top_genre_Pop', 'top_genre_Rock'
]

# Config de la page
st.set_page_config(page_title="Spotify Analytics", page_icon="🟢", layout="wide")

# --- GESTION DU CSS DYNAMIQUE (DÉBUT) ---
# On doit définir le CSS après avoir choisi le thème, mais on a besoin du thème pour le définir.
# Astuce : On lit le session state ou on utilise une valeur par défaut, puis on re-render.
# Pour faire simple ici, on injecte le CSS de base, puis on injecte le CSS spécifique après le choix.

st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
""", unsafe_allow_html=True)

# --- BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    # Titre Menu
    st.markdown("### <i class='fa-solid fa-bars' style='color:#1DB954;'></i> Menu Principal", unsafe_allow_html=True)
    st.write("") # Espaceur
    
    # --- SECTION 1 : APPARENCE ---
    # Titre avec Vraie Icône Verte
    st.markdown("#### <i class='fa-solid fa-palette' style='color:#1DB954;'></i> Apparence", unsafe_allow_html=True)
    # Toggle
    with st.expander("Modifier le Thème", expanded=False):
        theme_choice = st.radio("", ["Clair (Standard)", "Sombre (Spotify)"], label_visibility="collapsed")
    
    st.markdown("---")

    # --- SECTION 2 : À PROPOS ---
    # Titre avec Vraie Icône Verte
    st.markdown("#### <i class='fa-solid fa-circle-info' style='color:#1DB954;'></i> À Propos", unsafe_allow_html=True)
    # Toggle
    with st.expander("Voir les détails", expanded=False):
        st.markdown("""
            <div style='text-align: justify; font-size: 14px;'>
                <b><i class='fa-solid fa-bullseye' style='color:#1DB954;'></i> Objectif :</b><br>
                Ce tableau de bord utilise l'IA pour prédire le risque de désabonnement (Churn) des utilisateurs Spotify.
                <br><br>
                <b><i class='fa-solid fa-code' style='color:#1DB954;'></i> Stack :</b><br>
                Python, Streamlit, Scikit-Learn.
                <br><br>
                <i class='fa-solid fa-copyright'></i> Version 2.0 Pro
            </div>
        """, unsafe_allow_html=True)

# --- DÉFINITION DES COULEURS (LOGIQUE DE THÈME) ---
if "Sombre" in theme_choice:
    # MODE SOMBRE
    bg_color = "#121212"
    card_bg = "#181818"
    text_color = "#FFFFFF"
    sub_text_color = "#B3B3B3"
    chart_template = "plotly_dark"
    # Sidebar specific
    sidebar_bg = "#000000"
else:
    # MODE CLAIR
    bg_color = "#FAFAFA"
    card_bg = "#FFFFFF"
    text_color = "#191414"
    sub_text_color = "#535353"
    chart_template = "plotly_white"
    sidebar_bg = "#F0F0F0"

# --- INJECTION CSS FINALE ---
st.markdown(f"""
    <style>
        /* Application Background */
        .stApp {{ background-color: {bg_color}; }}
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg};
            border-right: 1px solid #1DB954;
        }}

        /* Global Text Colors */
        h1, h2, h3, h4, h5, h6, p, li, .stMarkdown, .stRadio label {{ color: {text_color} !important; }}
        
        /* Expander Styling (Pour qu'il s'intègre bien sous le titre) */
        .streamlit-expanderHeader {{
            background-color: {card_bg};
            color: {text_color};
            border-radius: 5px;
        }}
        
        /* Style des Icônes Générales */
        .fa-solid {{ margin-right: 8px; }}

        /* Cartes Statistiques */
        .stat-card {{
            background-color: {card_bg};
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
            border-left: 5px solid #1DB954;
            text-align: center;
            transition: transform 0.2s;
            margin-bottom: 10px;
        }}
        .stat-card:hover {{ transform: scale(1.02); }}
        .stat-value {{ font-size: 28px; font-weight: bold; color: {text_color}; }}
        .stat-label {{ font-size: 14px; color: {sub_text_color}; font-weight: 600; text-transform: uppercase; }}
        .stat-icon {{ font-size: 24px; color: #1DB954; margin-bottom: 10px; }}

        /* Bouton Vert Predict */
        .stButton>button {{
            background-color: #1DB954 !important; 
            color: white !important; 
            border-radius: 30px !important;
            font-size: 18px !important;
            font-weight: bold !important; 
            border: none !important; 
            padding: 15px 30px !important;
            width: 100% !important;
            box-shadow: 0 4px 10px rgba(29, 185, 84, 0.4);
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #1ed760 !important; 
            transform: scale(1.02);
            box-shadow: 0 6px 14px rgba(29, 185, 84, 0.6);
        }}

        /* Boîtes de Résultat */
        .result-box {{
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            margin-bottom: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            background-color: {card_bg}; 
        }}
        .status-risk {{ border: 2px solid #FF5F6D; }}
        .status-safe {{ border: 2px solid #1DB954; }}
        
        .box-title {{ font-weight: bold; font-size: 18px; margin-bottom: 5px; opacity: 0.9; color: {text_color}; }}
        .box-value {{ font-weight: 800; font-size: 32px; color: {text_color}; }}
        .box-text {{ font-size: 16px; margin-top: 5px; color: {sub_text_color}; }}
        
        /* Suggestion Box */
        .suggestion-box {{
            background-color: rgba(29, 185, 84, 0.1); 
            border: 1px solid #1DB954;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .suggestion-text {{ color: {text_color}; }}

    </style>
""", unsafe_allow_html=True)

# --- FONCTIONS UTILES ---
@st.cache_data
def load_data(path):
    if not os.path.exists(path): return None
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    if not os.path.exists(path): return None
    return joblib.load(path)

# --- EN-TÊTE PRINCIPAL (LOGO À DROITE) ---
col_head1, col_head2 = st.columns([4, 1])

with col_head1:
    st.title("Tableau de Bord - Spotify Churn")
    st.markdown("##### <i class='fa-solid fa-chart-line' style='color:#1DB954;'></i> Analyse de Rétention & Prédiction IA", unsafe_allow_html=True)

with col_head2:
    # Logo Spotify en haut à droite (Image URL standard)
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg", width=160)

df = load_data(DATASET_PATH)

if df is not None:
    tab1, tab2, tab3 = st.tabs(["Vue d'ensemble", "Analyses Avancées", "Prédire le Churn"])

    # --- TAB 1: VUE D'ENSEMBLE ---
    with tab1:
        st.markdown("### <i class='fa-solid fa-tachometer-alt' style='color:#1DB954;'></i> Indicateurs Clés (KPI)", unsafe_allow_html=True)
        
        total_users = len(df)
        churn_rate = df['churned'].mean() * 100
        avg_mins = df['avg_daily_minutes'].mean()
        premium_users = len(df[df['subscription_type'] == 'Premium'])

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon"><i class="fa-solid fa-users"></i></div>
                    <div class="stat-value">{total_users:,}</div>
                    <div class="stat-label">Utilisateurs Totaux</div>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon"><i class="fa-solid fa-user-slash"></i></div>
                    <div class="stat-value">{churn_rate:.1f}%</div>
                    <div class="stat-label">Taux de Churn</div>
                </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon"><i class="fa-solid fa-clock"></i></div>
                    <div class="stat-value">{avg_mins:.0f} min</div>
                    <div class="stat-label">Écoute Moyenne/Jour</div>
                </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon"><i class="fa-solid fa-crown"></i></div>
                    <div class="stat-value">{premium_users:,}</div>
                    <div class="stat-label">Abonnés Premium</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### <i class='fa-solid fa-database' style='color:#1DB954;'></i> Jeu de Données Complet", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=400)

    # --- TAB 2: ANALYSES ---
    with tab2:
        st.markdown("### <i class='fa-solid fa-magnifying-glass-chart' style='color:#1DB954;'></i> Analyse Approfondie", unsafe_allow_html=True)
        
        # Ligne 1
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### <i class='fa-solid fa-chart-pie' style='color:#1DB954;'></i> Répartition de Churn", unsafe_allow_html=True)
            st.plotly_chart(px.pie(df, names='churned', title="Proportion Churn vs Fidèles", 
                                 color_discrete_sequence=['#1DB954', '#191414'], hole=0.4, template=chart_template), use_container_width=True)
        
        with c2:
            st.markdown("#### <i class='fa-solid fa-globe' style='color:#1DB954;'></i> Churn par Pays", unsafe_allow_html=True)
            if 'country' in df.columns:
                fig_bar_country = px.histogram(df, x="country", color="churned", barmode="group",
                                             title="Taux de Churn par Pays", color_discrete_sequence=['#1DB954', '#535353'], template=chart_template)
                st.plotly_chart(fig_bar_country, use_container_width=True)
            else:
                st.info("Colonne 'Pays' non disponible.")

        # Ligne 2
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### <i class='fa-solid fa-credit-card' style='color:#1DB954;'></i> Impact de l'Abonnement", unsafe_allow_html=True)
            st.plotly_chart(px.histogram(df, x="subscription_type", color="churned", barmode='group', 
                                         title="Churn par Abonnement", color_discrete_sequence=['#1DB954', '#535353'], template=chart_template), use_container_width=True)
        with c4:
            st.markdown("#### <i class='fa-solid fa-headset' style='color:#1DB954;'></i> Support vs Churn", unsafe_allow_html=True)
            st.plotly_chart(px.box(df, x="churned", y="support_tickets", color="churned", 
                                   title="Impact des Réclamations", color_discrete_sequence=['#1DB954', '#191414'], template=chart_template), use_container_width=True)

        # Ligne 3: Heatmap
        st.markdown("#### <i class='fa-solid fa-fire' style='color:#1DB954;'></i> Corrélation des Facteurs", unsafe_allow_html=True)
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        corr = numeric_df.corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Greens", title="Matrice de Corrélation", template=chart_template)
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- TAB 3: PRÉDICTION ---
    with tab3:
        st.markdown("### <i class='fa-solid fa-robot' style='color:#1DB954;'></i> Prédicteur IA en Temps Réel", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            sub_type = st.selectbox("Type d'Abonnement", ["Free", "Premium"])
            country = st.selectbox("Pays", ['BR', 'CA', 'DE', 'FR', 'IN', 'PK', 'RU', 'UK', 'US'])
            top_genre = st.selectbox("Genre Préféré", ['Country', 'Electronic', 'Hip-Hop', 'Jazz', 'Pop', 'Rock'])
        with c2:
            daily_mins = st.number_input("Minutes Moyennes par Jour", 0.0, 1000.0, 60.0)
            skips = st.number_input("Sauts (Skips) par Jour", 0.0, 100.0, 2.0)
            playlists = st.number_input("Nombre de Playlists", 0, 500, 5)
        with c3:
            tickets = st.number_input("Tickets de Support", 0, 20, 0)
            days_login = st.number_input("Jours Depuis Dernière Connexion", 0, 365, 1)

        st.markdown("<br>", unsafe_allow_html=True)
        
        predict_btn = st.button("PRÉDIRE LE STATUT DE CHURN")

        if predict_btn:
            model = load_model(MODEL_PATH)
            std_scaler = load_model(SCALER_STD_PATH)
            mm_scaler = load_model(SCALER_MM_PATH)
            enc_sub = load_model(ENC_SUB_PATH)

            if all([model, std_scaler, mm_scaler, enc_sub]):
                try:
                    # Préparation des Données
                    input_data = {col: 0 for col in TRAINING_COLUMNS}
                    input_data['avg_daily_minutes'] = daily_mins
                    input_data['skips_per_day'] = skips
                    input_data['days_since_last_login'] = days_login
                    input_data['number_of_playlists'] = playlists
                    input_data['support_tickets'] = tickets

                    # Gestion Abonnement
                    encoded_sub = enc_sub.transform(pd.DataFrame({'subscription_type': [sub_type]}))
                    if hasattr(encoded_sub, "toarray"): encoded_sub = encoded_sub.toarray()
                    input_data['subscription_type'] = encoded_sub.flatten()[0]

                    # Gestion One-Hot
                    if f"country_{country}" in input_data: input_data[f"country_{country}"] = 1
                    if f"top_genre_{top_genre}" in input_data: input_data[f"top_genre_{top_genre}"] = 1

                    # Création DF et Scaling
                    input_df = pd.DataFrame([input_data])
                    input_df[["avg_daily_minutes", "skips_per_day", "days_since_last_login"]] = std_scaler.transform(input_df[["avg_daily_minutes", "skips_per_day", "days_since_last_login"]])
                    input_df[['number_of_playlists']] = mm_scaler.transform(input_df[['number_of_playlists']])

                    # Prédiction
                    pred = model.predict(input_df)[0]
                    prob = model.predict_proba(input_df)[0][1]

                    st.markdown("---")
                    
                    # RÉSULTATS AVEC COULEURS ADAPTÉES AU THÈME
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        if pred == 1:
                            st.markdown(f"""
                                <div class="result-box status-risk">
                                    <div class="box-title" style="color:#FF5F6D;"><i class="fa-solid fa-triangle-exclamation"></i> RÉSULTAT</div>
                                    <div class="box-value">Risque Élevé</div>
                                    <div class="box-text">Probabilité de départ : {prob*100:.1f}%</div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class="result-box status-safe">
                                    <div class="box-title" style="color:#1DB954;"><i class="fa-solid fa-check-circle"></i> RÉSULTAT</div>
                                    <div class="box-value">Utilisateur Fidèle</div>
                                    <div class="box-text">Probabilité de départ : {prob*100:.1f}%</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                    with res_col2:
                         if pred == 1:
                            st.markdown(f"""
                                <div class="suggestion-box">
                                    <div class="box-title" style="color:#1DB954;"><i class="fa-solid fa-lightbulb"></i> ACTION SUGGÉRÉE</div>
                                    <div class="box-text suggestion-text">
                                        <b>Envoyer une offre 'Revenez'</b><br>
                                        Cet utilisateur montre des signes de désengagement. 
                                        Recommandé : 30 jours Premium offerts.
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                         else:
                            st.markdown(f"""
                                <div class="suggestion-box">
                                    <div class="box-title" style="color:#1DB954;"><i class="fa-solid fa-heart"></i> ACTION SUGGÉRÉE</div>
                                    <div class="box-text suggestion-text">
                                        <b>Maintenir l'Engagement</b><br>
                                        L'utilisateur est sain. Recommandez des playlists 
                                        "Daily Mix" pour garder une rétention haute.
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Erreur : {e}")
            else:
                st.warning("Fichiers modèles manquants.")
else:
    st.info("Veuillez vous assurer que le fichier `spotify_churn_dataset.csv` est dans le dossier parent.")