import streamlit as st
from openai import OpenAI

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

# ✅ Hugging Face Router (OpenAI-compatible)
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=st.secrets["HF_API_TOKEN"],  # on réutilise ton secret actuel
)

# Modèle par défaut (exemple officiel HF)
model_id = st.text_input(
    "Model (tu peux changer plus tard)",
    value="moonshotai/Kimi-K2-Instruct-0905"
)

def call_llm(q: str) -> str:
    system = (
        "Tu es un agent qui simule ma façon de raisonner comme Data Analyst / Data Scientist.\n\n"
        "Réponds en français avec EXACTEMENT cette structure :\n\n"
        "Réponse :\n(texte court)\n\n"
        "Raisonnement :\n- étape 1\n- étape 2\n- étape 3\n\n"
        "Alternatives :\n- option 1 + pourquoi je ne la choisis pas\n- option 2 + pourquoi je ne la choisis pas\n"
    )

    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": q},
        ],
        temperature=0.4,
    )
    return completion.choices[0].message.content

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
