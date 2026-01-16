import streamlit as st

st.set_page_config(page_title="Reasoning Agent", layout="wide")

st.title("Reasoning Agent")
st.caption("Simulation de ma façon de raisonner (Data Analyst / Data Scientist).")

st.markdown("""
Pose une question (technique ou métier).  
L’application affiche ensuite : **réponse**, **raisonnement**, **preuves**, **alternatives**.

⚠️ Ceci est une **simulation de ma façon de penser**, pas une automatisation de réponses à ma place.
""")

st.subheader("Exemples de questions")
examples = [
    "Que fais-tu si les données sont de mauvaise qualité ?",
    "Comment choisis-tu un modèle pour un problème de churn ?",
    "Comment traduis-tu un besoin métier flou en analyse data ?",
    "Que fais-tu quand les résultats ne confirment pas l’hypothèse métier ?"
]

selected = st.radio("Clique sur une question ou écris la tienne :", examples)

question = st.text_area("Ta question", value=selected, height=120)

if st.button("Générer"):
    st.subheader("Réponse")
    st.write("MVP en cours — l’IA sera branchée à l’étape suivante.")

    st.subheader("Raisonnement")
    st.markdown("""
- Clarifier l’objectif métier  
- Identifier les hypothèses  
- Vérifier les données disponibles  
- Choisir une approche simple en premier  
- Identifier les limites et risques
""")

    st.subheader("Preuves")
    st.info("À venir : citations issues de mon CV et de mes projets.")

    st.subheader("Alternatives")
    st.markdown("""
- Approche plus complexe dès le départ → risquée sans compréhension métier  
- Refuser la demande → possible si les données ne permettent pas une décision fiable
""")
