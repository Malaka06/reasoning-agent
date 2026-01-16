import re
import streamlit as st
import requests

# =========================
# Configuration de la page
# =========================
st.set_page_config(
    page_title="Reasoning Agent – Data Analyst / Data Scientist",
    layout="wide",
    page_icon="🧠",
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
# Inputs
# =========================
examples = [
    "Que fais-tu si les données sont de mauvaise qualité ?",
    "Comment choisis-tu un modèle pour un problème de churn ?",
    "Comment traduis-tu un besoin métier flou en analyse data ?",
    "Que fais-tu quand les résultats ne confirment pas l’hypothèse métier ?"
]

selected = st.radio("💡 Exemples de questions (ou écris la tienne) :", examples)

question = st.text_area("Ta question", value=selected, height=120)

default_model = "moonshotai/Kimi-K2-Instruct-0905"
model_id = st.text_input("Modèle (tu peux changer plus tard)", value=default_model)

with st.expander("⚙️ Paramètres (optionnel)"):
    temperature = st.slider("temperature", 0.0, 1.0, 0.4, 0.05)
    max_tokens = st.slider("max_tokens", 128, 1200, 600, 32)

# =========================
# Hugging Face Router
# =========================
HF_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TOKEN = st.secrets.get("HF_API_TOKEN")

if not HF_TOKEN:
    st.error("❌ Secret manquant : ajoute `HF_API_TOKEN` dans Streamlit → Settings → Secrets.")
    st.stop()

HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

SYSTEM_PROMPT = (
    "Tu es un agent qui simule ma façon de raisonner comme Data Analyst / Data Scientist.\n\n"
    "Tu dois répondre en français avec EXACTEMENT la structure suivante :\n\n"
    "Réponse :\n"
    "(1–2 phrases, orientées décision)\n\n"
    "Raisonnement :\n"
    "- étape 1 : clarification / diagnostic\n"
    "- étape 2 : analyse et arbitrages\n"
    "- étape 3 : décision finale\n\n"
    "Preuves :\n"
    "- 2–3 critères / signaux concrets que tu utiliserais (même sans données sous les yeux)\n\n"
    "Alternatives :\n"
    "- option 1 + pourquoi je ne la choisis pas\n"
    "- option 2 + pourquoi je ne la choisis pas\n\n"
    "Sois clair, pragmatique et orienté impact métier."
)

@st.cache_data(show_spinner=False, ttl=3600)
def call_llm_cached(model: str, q: str, temp: float, max_toks: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q}
        ],
        "temperature": temp,
        "max_tokens": max_toks,
    }

    r = requests.post(HF_URL, headers=HF_HEADERS, json=payload, timeout=90)

    if r.status_code != 200:
        # On affiche un extrait pour éviter un pavé illisible
        raise RuntimeError(f"{r.status_code} - {r.text[:800]}")

    data = r.json()
    return data["choices"][0]["message"]["content"]

def split_sections(text: str) -> dict:
    """
    Essaie de découper Réponse / Raisonnement / Preuves / Alternatives.
    Si le modèle ne respecte pas parfaitement, on fait au mieux.
    """
    sections = {"Réponse": "", "Raisonnement": "", "Preuves": "", "Alternatives": ""}

    # Normalise
    t = text.strip()

    # Regex basique sur titres
    pattern = r"(Réponse\s*:|Raisonnement\s*:|Preuves\s*:|Alternatives\s*:)"
    parts = re.split(pattern, t)

    if len(parts) <= 1:
        sections["Réponse"] = t
        return sections

    current = None
    for chunk in parts:
        c = chunk.strip()
        if not c:
            continue
        if c.startswith("Réponse"):
            current = "Réponse"
            continue
        if c.startswith("Raisonnement"):
            current = "Raisonnement"
            continue
        if c.startswith("Preuves"):
            current = "Preuves"
            continue
        if c.startswith("Alternatives"):
            current = "Alternatives"
            continue
        if current:
            sections[current] += (c + "\n")

    # Clean
    for k in sections:
        sections[k] = sections[k].strip()

    return sections

# =========================
# Génération
# =========================
if st.button("🚀 Générer"):
    if not question.strip():
        st.warning("Merci d’écrire une question.")
        st.stop()

    with st.spinner("Analyse et raisonnement en cours..."):
        try:
            raw = call_llm_cached(model_id, question, temperature, max_tokens)
        except Exception as e:
            st.error(f"Erreur IA : {e}")
            st.stop()

    sec = split_sections(raw)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Réponse")
        st.write(sec["Réponse"] or "—")

        st.subheader("🧾 Preuves (critères)")
        st.write(sec["Preuves"] or "—")

    with col2:
        st.subheader("🧠 Raisonnement")
        st.write(sec["Raisonnement"] or "—")

        st.subheader("🔁 Alternatives")
        st.write(sec["Alternatives"] or "—")

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
- Hugging Face Router (LLM)

**Approche**
- Prompt structuré pour rendre explicite le raisonnement
- Mise en avant des alternatives non retenues
- Réponses concises et orientées impact

**Limite assumée**
- Ce n’est pas un chatbot générique.
- C’est une **simulation de ma manière de raisonner**.
""")
