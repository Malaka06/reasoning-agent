import streamlit as st
import requests

# =========================
# Configuration de la page
# =========================
st.set_page_config(
    page_title="Reasoning Agent – Data Analyst / Data Scientist",
    layout="wide"
)

st.title("🧠 Reasoning Agent")
st.caption("Simulation de ma façon de raisonner (Data Analyst / Data Scientist)")

st.info(
    "👋 **Recruteurs** : cette application ne vise pas à donner des réponses parfaites, "
    "mais à **montrer comment je raisonne**, structure mes décisions et arbitre entre plusieurs options "
    "face à des problématiques data réelles."
)

st.markdown("""
Pose une question **technique ou métier**.  
L’agent génère :
- une **réponse synthétique**
- le **raisonnement étape par étape**
- des **alternatives volontairement écartées**
""")

# =========================
# Questions d'exemple
# =========================
examples = [
    "Que fais-tu si les données sont de mauvaise qualité ?",
    "Comment choisis-tu un modèle pour un problème de churn ?",
    "Comment traduis-tu un besoin métier flou en analyse data ?",
    "Que fais-tu quand les résultats ne confirment pas l’hypothèse métier ?"
]

selected = st.radio(
    "💡 Exemples de questions (ou écris la tienne) :",
    examples
)

question = st.text_area(
    "Ta question",
    value=selected,
    height=120
)

# =========================
# Hugging Face Router
# =========================
MODEL_ID = "moonshotai/Kimi-K2-Instruct-0905"
HF_URL = "https://router.huggingface.co/v1/chat/completions"

HF_HEADERS = {
    "Authorization": f"Bearer {st.secrets['HF_API_TOKEN']}",
    "Content-Type": "application/json",
}

def call_llm(user_question: str) -> str:
    system_prompt = (
        "Tu es un agent qui simule ma façon de raisonner comme Data Analyst / Data Scientist.\n\n"
        "Tu dois répondre en français avec EXACTEMENT la structure suivante :\n\n"
        "Réponse :\n"
        "(1–2 phrases, orientées décision)\n\n"
        "Raisonnement :\n"
        "- étape 1 : clarification / diagnostic\n"
        "- étape 2 : analyse et arbitrages\n"
        "- étape 3 : décision finale\n\n"
        "Alternatives :\n"
        "- option 1 + pourquoi je ne la choisis pas\n"
        "- option 2 + pourquoi je ne la choisis pas\n\n"
        "Sois clair, pragmatique et orienté impact métier."
    )

    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        "temperature": 0.4,
    }

    r = requests.post(
        HF_URL,
        headers=HF_HEADERS,
        json=payload,
        timeout=90
    )

    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} - {r.text}")

    data = r.json()
    return data["choices"][0]["message"]["content"]

# =========================
# Génération
# =========================
if st.button("🚀 Générer"):
    if not question.strip():
        st.warning("Merci d’écrire une question.")
        st.stop()

    with st.spinner("Analyse et raisonnement en cours..."):
        try:
            answer = call_llm(question)
        except Exception as e:
            st.error(f"Erreur IA : {e}")
            st.stop()

    st.markdown(answer)

# =========================
# Explication du projet
# =========================
with st.expander("🔍 Comment cet agent a été conçu"):
    st.markdown("""
    **Objectif**
    - Montrer mon *processus de réflexion*, pas seulement des résultats.
    - Rendre visible la prise de décision data en contexte métier.

    **Stack**
    - Python
    - Streamlit
    - Hugging Face Router (LLM open-source)

    **Approche**
    - Prompt structuré pour forcer la clarté du raisonnement
    - Mise en avant des alternatives non retenues
    - Réponses volontairement concises et orientées impact

    **Limite assumée**
    - Ce n’est pas un chatbot générique.
    - C’est une **simulation de ma manière de raisonner**.
    """)
