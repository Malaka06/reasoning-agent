import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Reasoning Agent", layout="wide")

st.title("Reasoning Agent")
st.caption("Simulation de ma façon de raisonner (Data Analyst / Data Scientist).")

st.markdown("""
Pose une question (technique ou métier).  
L’agent génère une **réponse**, un **raisonnement** et des **alternatives**.

⚠️ Ceci est une simulation de ma façon de penser, pas une automatisation de réponses à ma place.
""")

examples = [
    "Que fais-tu si les données sont de mauvaise qualité ?",
    "Comment choisis-tu un modèle pour un problème de churn ?",
    "Comment traduis-tu un besoin métier flou en analyse data ?",
    "Que fais-tu quand les résultats ne confirment pas l’hypothèse métier ?"
]

selected = st.radio("Exemples de questions :", examples)
question = st.text_area("Ta question", value=selected, height=120)

def call_llm(prompt: str) -> str:
    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        token=st.secrets["HF_API_TOKEN"]
    )

    return client.text_generation(
        prompt,
        max_new_tokens=300,
        temperature=0.4
    )

if st.button("Générer"):
    if not question.strip():
        st.warning("Écris une question.")
        st.stop()

    with st.spinner("Génération en cours..."):
        prompt = f"""
Tu es un agent qui simule ma façon de raisonner comme Data Analyst / Data Scientist.

Réponds en français avec EXACTEMENT cette structure :

Réponse :
(texte court)

Raisonnement :
- étape 1
- étape 2
- étape 3

Alternatives :
- option 1 + pourquoi non
- option 2 + pourquoi non

Question :
{question}
""".strip()

        try:
            result = call_llm(prompt)
        except Exception as e:
            st.error(f"Erreur IA : {e}")
            st.stop()

    st.markdown(result)
