import streamlit as st
from huggingface_hub import InferenceClient

# ---------------------------
# Configuration de la page
# ---------------------------
st.set_page_config(
    page_title="Reasoning Agent",
    layout="wide"
)

st.title("Reasoning Agent")
st.caption("Simulation de ma façon de raisonner (Data Analyst / Data Scientist).")

st.markdown("""
Pose une question (technique ou métier).  
L’agent génère une **réponse**, un **raisonnement** et des **alternatives**.

⚠️ Ceci est une **simulation de ma façon de penser**, pas une automatisation de réponses à ma place.
""")

# ---------------------------
# Questions d'exemple
# ---------------------------
examples = [
    "Que fais-tu si les données sont de mauvaise qualité ?",
    "Comment choisis-tu un modèle pour un problème de churn ?",
    "Comment traduis-tu un besoin métier flou en analyse data ?",
    "Que fais-tu quand les résultats ne confirment pas l’hypothèse métier ?"
]

selected = st.radio("Exemples de questions :", examples)
question = st.text_area("Ta question", value=selected, height=120)

# ---------------------------
# Appel Hugging Face (CHAT)
# ---------------------------
def call_llm(user_question: str) -> str:
    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        token=st.secrets["HF_API_TOKEN"]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un agent qui simule ma façon de raisonner comme Data Analyst / Data Scientist.\n\n"
                "Tu dois répondre en français avec EXACTEMENT cette structure :\n\n"
                "Réponse :\n"
                "(texte court, clair, orienté décision)\n\n"
                "Raisonnement :\n"
                "- étape 1\n"
                "- étape 2\n"
                "- étape 3\n\n"
                "Alternatives :\n"
                "- option 1 + pourquoi je ne la choisis pas\n"
                "- option 2 + pourquoi je ne la choisis pas\n"
            )
        },
        {
            "role": "user",
            "content": user_question
        }
    ]

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=400,
        temperature=0.4,
    )

    return response.choices[0].message["content"]

# ---------------------------
# Bouton Générer
# ---------------------------
if st.button("Générer"):
    if not question.strip():
        st.warning("Écris une question.")
        st.stop()

    with st.spinner("Génération en cours..."):
        try:
            result = call_llm(question)
        except Exception as e:
            st.error(f"Erreur IA : {e}")
            st.stop()

    st.markdown(result)
