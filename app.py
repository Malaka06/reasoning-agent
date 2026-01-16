import streamlit as st

st.set_page_config(page_title="Reasoning Agent", layout="wide")

st.title("Reasoning Agent")
st.caption("Simulation de ma façon de raisonner (Data Analyst / Data Scientist).")

st.markdown("""
Pose une question (technique ou métier).  
L’application affichera ensuite : **réponse**, **raisonnement**, **preuves**, **alternatives**.
""")

question = st.text_area("Ta question", height=120)

if st.button("Générer"):
    if not question.strip():
        st.warning("Écris une question.")
    else:
        st.subheader("Réponse")
        st.write("MVP en cours : on branche l’IA et les preuves (RAG) juste après.")

        st.subheader("Raisonnement")
        st.write("- Étape 1 : clarifier l’objectif\n- Étape 2 : vérifier les données\n- Étape 3 : proposer une baseline\n- Étape 4 : itérer")

        st.subheader("Preuves")
        st.info("Bientôt : citations depuis mon CV et mes projets.")

        st.subheader("Alternatives")
        st.write("- Option A : ...\n- Option B : ...")
