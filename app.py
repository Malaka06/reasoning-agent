import streamlit as st
import requests
import json

st.set_page_config(page_title="Reasoning Agent", layout="wide")

st.title("Reasoning Agent")
st.caption("Simulation de ma façon de raisonner (Data Analyst / Data Scientist).")

st.markdown("""
Pose une question (technique ou métier).  
L’agent génère une **réponse**, un **raisonnement** et des **alternatives**.

⚠️ Ceci est une simulation de ma façon de penser, pas une automatisation de réponses à ma place.
""")

# -----------------------
# Questions d'exemple
# -----------------------
examples = [
    "Que fais-tu si les données sont de mauvaise qualité ?",
    "Comment choisis-tu un modèle pour un problème de churn ?",
    "Comment traduis-tu un besoin métier flou en analyse data ?",
    "Que fais-tu quand les résultats ne confirment pas l’hypothèse métier ?"
]

selected = st.radio("Exemples de questions :", examples)
question = st.text_area("Ta question", value=selected, height=120)

# -----------------------
# Appel Hugging Face (HTTP)
# -----------------------
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
HF_HEADERS = {
    "Authorization": f"Bearer {st.secrets['HF_API_TOKEN']}"
}

def call_llm(user_question: str) -> str:
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
- option 1 + pourquoi je ne la choisis pas
- option 2 + pourquoi je ne la choisis pas

Question :
{user_question}
""".strip()

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.3
        }
    }

    response = requests.post(
        HF_API_URL,
        headers=HF_HEADERS,
        data=json.dumps(payload),
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(response.text)

    result = response.json()

    # flan-t5 renvoie une liste de dicts
    return result[0]["generated_text"]

# -----------------------
# Bouton Générer
# -----------------------
if st.button("Générer"):
    if not question.strip():
        st.warning("Écris une question.")
        st.stop()

    with st.spinner("Génération en cours..."):
        try:
            answer = call_llm(question)
        except Exception as e:
            st.error(f"Erreur IA : {repr(e)}")
            st.stop()

    st.markdown(answer)
