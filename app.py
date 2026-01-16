import streamlit as st
import requests

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

MODEL_ID = "google/flan-t5-large"

# ✅ Nouveau endpoint Hugging Face (router)
HF_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
HF_HEADERS = {
    "Authorization": f"Bearer {st.secrets['HF_API_TOKEN']}",
    "Accept": "application/json",
    "Content-Type": "application/json",
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
            "temperature": 0.3,
        }
    }

    r = requests.post(HF_URL, headers=HF_HEADERS, json=payload, timeout=90)

    if r.status_code != 200:
        # affiche une erreur lisible
        raise RuntimeError(f"{r.status_code} - {r.text[:500]}")

    data = r.json()

    # La sortie peut être une liste de dicts selon le modèle/provider
    if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
        return data[0]["generated_text"]

    # Parfois, c'est un dict avec des champs différents
    if isinstance(data, dict):
        if "generated_text" in data:
            return data["generated_text"]
        if "error" in data:
            raise RuntimeError(data["error"])

    # fallback
    return str(data)

if st.button("Générer"):
    if not question.strip():
        st.warning("Écris une question.")
        st.stop()

    with st.spinner("Génération en cours..."):
        try:
            answer = call_llm(question)
        except Exception as e:
            st.error(f"Erreur IA : {e}")
            st.stop()

    st.markdown(answer)
