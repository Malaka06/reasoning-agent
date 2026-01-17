import os
import re
import base64
from pathlib import Path
from typing import Dict, Optional

import streamlit as st
import requests


# =========================
# Page config (premium minimal)
# =========================
st.set_page_config(
    page_title="Reasoning Agent — Aimée",
    page_icon="🧠",
    layout="wide",
)

# =========================
# Hugging Face Router config
# =========================
MODEL_ID_DEFAULT = "moonshotai/Kimi-K2-Instruct-0905"
HF_URL = "https://router.huggingface.co/v1/chat/completions"


def get_hf_token() -> str:
    """Get HF token from Streamlit secrets or env."""
    if "HF_API_TOKEN" in st.secrets:
        return st.secrets["HF_API_TOKEN"]
    token = os.getenv("HF_API_TOKEN", "")
    return token


def make_hf_headers() -> Dict[str, str]:
    token = get_hf_token()
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


# =========================
# Identity + Facts (strict)
# =========================
IDENTITY_FR = (
    "Je suis l’assistant IA d’Aimée. Je simule sa manière de raisonner et de prendre des décisions "
    "en analyse data, marketing et métier. Je ne la remplace pas et je ne parle pas à sa place."
)
IDENTITY_EN = (
    "I’m Aimée’s AI assistant. I simulate how she reasons and makes decisions across data, marketing, "
    "and business. I do not replace her and I do not speak on her behalf."
)

# ⚠️ Mets ici des faits vrais. N'ajoute rien que tu ne veux pas afficher publiquement.
AIMEE_FACTS = """
Nom : Aimée

Positionnement :
- Business Analyst / Data Analyst junior
- Interface entre data, marketing et métier
- Objectif : aider à la prise de décision (KPIs, analyses, dashboards)

Compétences / outils (niveau honnête) :
- Power BI
- Looker Studio
- Google Analytics (GA4)
- SQL (notions)
- Python (notions)

Points forts :
- Traduire un besoin métier en indicateurs et analyses
- Construire des dashboards utiles (usage + décisions)
- Travailler avec des données imparfaites sans bloquer le projet
- Communiquer clairement avec des profils non techniques

Recherche :
- Postes : Data Analyst / Business Analyst (junior)
"""

REFUSAL_FR = (
    "Cette information relève de la sphère personnelle et n’est pas détaillée ici. "
    "Je peux en revanche parler du parcours professionnel et de la façon de travailler d’Aimée."
)
REFUSAL_EN = (
    "That information belongs to Aimée’s private life and isn’t shared here. "
    "I can however talk about her professional background and how she works."
)

AIMEE_STYLE = """
Style de raisonnement (à simuler) :
- Clarifier l’objectif métier et la décision attendue
- Reformuler un besoin flou en question mesurable
- Choisir des KPIs utiles (pas juste disponibles)
- Qualifier la qualité des données et les limites
- Proposer une approche pragmatique orientée impact
- Expliquer sans jargon inutile
"""

SYSTEM_PROMPT = f"""
You are Aimée’s professional AI assistant.

ABSOLUTE RULES:
- You are NOT Aimée.
- You speak about Aimée in third person ONLY.
- Never say "I am Aimée" or write as if you are Aimée.
- If asked "Who are you?" or "Qui es-tu?" answer ONLY with the identity sentence in the user's language.

ABOUT AIMÉE (single source of truth):
{AIMEE_FACTS}

AIMÉE'S REASONING STYLE:
{AIMEE_STYLE}

TRUTHFULNESS (STRICT):
- Use ONLY the info above for questions ABOUT Aimée.
- Never invent employers, dates, projects, countries, hobbies, private life details.
- If the info is not in the facts: say it’s not specified.
- For private/sensitive topics: use the refusal sentence in the user language.

MANDATORY RESPONSE FORMAT:
Answer:
(1–2 sentences, decision-oriented)

Reasoning:
- step 1: clarify / diagnose
- step 2: analysis / trade-offs
- step 3: conclusion

Evidence:
- 2–3 concrete criteria / signals used

Alternatives:
- option 1 + why rejected
- option 2 + why rejected

Business conclusion:
- next concrete action / recommendation
""".strip()


# =========================
# Helpers
# =========================
def is_english(text: str) -> bool:
    t = (text or "").lower()
    return any(x in t for x in ["why", "what", "how", "who", "resume", "cv", "your", "you "])


def is_identity_question(text: str) -> bool:
    t = (text or "").lower()
    triggers = [
        "qui es-tu", "tu es qui", "c’est qui", "c'est qui",
        "who are you", "about you", "are you aimee", "are you aimée",
        "es-tu aimée", "êtes-vous aimée", "agent", "assistant",
    ]
    return any(k in t for k in triggers)


def identity_answer(lang_en: bool) -> str:
    return IDENTITY_EN if lang_en else IDENTITY_FR


def refusal_answer(lang_en: bool) -> str:
    return REFUSAL_EN if lang_en else REFUSAL_FR


def call_llm(user_question: str, model_id: str) -> str:
    token = get_hf_token()
    if not token:
        raise RuntimeError("HF_API_TOKEN manquant. Ajoute-le dans Settings > Secrets (Streamlit) ou en variable d’environnement.")

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question},
        ],
        "temperature": 0.4,
    }

    r = requests.post(HF_URL, headers=make_hf_headers(), json=payload, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} - {r.text}")

    data = r.json()
    return data["choices"][0]["message"]["content"]


def split_sections(text: str) -> Dict[str, str]:
    """Split output into sections for clean UI. FR/EN supported."""
    if not text:
        return {"raw": ""}

    t = text.replace("\r\n", "\n")

    patterns = {
        "answer": r"(?im)^(réponse|answer)\s*:\s*$",
        "reasoning": r"(?im)^(raisonnement|reasoning)\s*:\s*$",
        "evidence": r"(?im)^(preuves|evidence)\s*:\s*$",
        "alternatives": r"(?im)^(alternatives)\s*:\s*$",
        "conclusion": r"(?im)^(conclusion métier|business conclusion)\s*:\s*$",
    }

    matches = []
    for key, pat in patterns.items():
        for m in re.finditer(pat, t):
            matches.append((m.start(), m.end(), key))
    matches.sort(key=lambda x: x[0])

    if not matches:
        return {"raw": text}

    out: Dict[str, str] = {}
    for i, (start, end, key) in enumerate(matches):
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(t)
        out[key] = t[end:next_start].strip()

    out["raw"] = text
    return out


def read_file_bytes(path: str) -> Optional[bytes]:
    p = Path(path)
    if p.exists() and p.is_file():
        return p.read_bytes()
    return None


def pdf_iframe(pdf_bytes: bytes, height: int = 820) -> None:
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    st.components.v1.html(
        f"""
        <iframe
          src="data:application/pdf;base64,{b64}"
          width="100%"
          height="{height}px"
          style="border:0; border-radius:12px;"
        ></iframe>
        """,
        height=height + 20,
    )


# =========================
# UI — Header
# =========================
left, right = st.columns([2, 1], vertical_alignment="top")

with left:
    st.title("🧠 Reasoning Agent — Aimée")
    st.caption("Business Analyst / Data Analyst (junior) — Marketing × Data × Métier")

with right:
    st.markdown("#### 🎯 Objectif")
    st.write("Montrer **comment Aimée raisonne** et arbitre entre plusieurs options.")
    st.write("➡️ C’est un assistant, pas une automatisation à sa place.")

st.divider()

# =========================
# 3 pages
# =========================
tab_agent, tab_projects, tab_cv = st.tabs(["🧠 Agent", "📁 Projets (mémoire)", "📄 CV"])


# =========================
# Page 1 — Agent
# =========================
with tab_agent:
    st.info("👋 Recruteurs : posez une question. L’agent répond de façon structurée (réponse, raisonnement, preuves, alternatives).")

    examples = [
        "Qui es-tu ?",
        "Qui est Aimée ?",
        "Pourquoi recruter Aimée ?",
        "Comment traduirais-tu un besoin métier flou en analyse data ?",
        "Que fais-tu si les données sont de mauvaise qualité ?",
        "Comment choisir des KPIs utiles pour piloter une campagne marketing ?",
        "How would you handle imperfect data under a tight deadline?",
    ]

    c1, c2 = st.columns([1, 2], vertical_alignment="top")
    with c1:
        selected = st.radio("💡 Exemples", examples, index=2)
        st.caption("Ou écris ta propre question.")
    with c2:
        question = st.text_area("Ta question", value=selected, height=140)

    with st.expander("⚙️ Paramètres (optionnel)"):
        model_id = st.text_input("Model", value=MODEL_ID_DEFAULT)
        st.caption("Tu peux laisser par défaut.")

    generate = st.button("🚀 Générer", use_container_width=True)

    if generate:
        if not question.strip():
            st.warning("Merci d’écrire une question.")
            st.stop()

        lang_en = is_english(question)

        # 1) Identity guard (no API call)
        if is_identity_question(question):
            st.markdown("### ✅ Réponse")
            st.write(identity_answer(lang_en))
            st.stop()

        # 2) LLM call
        with st.spinner("Analyse et raisonnement en cours…"):
            try:
                answer = call_llm(question, model_id=model_id)
            except Exception as e:
                st.error(f"Erreur IA : {e}")
                st.stop()

        sections = split_sections(answer)

        st.markdown("### 📌 Résultat")
        st.markdown("#### ✅ Réponse")
        st.write(sections.get("answer", "—"))

        with st.expander("🧠 Raisonnement", expanded=True):
            st.write(sections.get("reasoning", "—"))

        with st.expander("📌 Preuves / critères", expanded=True):
            st.write(sections.get("evidence", "—"))

        with st.expander("🔁 Alternatives", expanded=False):
            st.write(sections.get("alternatives", "—"))

        with st.expander("🎯 Conclusion métier (prochaine action)", expanded=True):
            st.write(sections.get("conclusion", "—"))

        # fallback if model ignored formatting
        if set(sections.keys()) == {"raw"}:
            st.divider()
            st.caption("Format brut :")
            st.markdown(sections["raw"])


# =========================
# Page 2 — Projets (incl. mémoire)
# =========================
with tab_projects:
    st.subheader("📁 Projets & Mémoire")
    st.caption("L’objectif : montrer la valeur (objectif → méthode → résultat), pas empiler des lignes.")

    st.markdown("### 🎓 Travail de fin d’étude (Mémoire / TFE)")
    st.info("Conseil : une présentation courte et claire. Le recruteur doit comprendre en 30 secondes.")

    # ✅ Remplace le contenu entre [...] par le tien
    st.markdown("""
**Problématique**  
[1–2 phrases : problème métier + décision attendue]

**Contexte**  
- Secteur : [...]
- Enjeu business : [...]
- Contraintes : [...]

**Données**  
- Sources : [...]
- Périmètre : [...]
- Qualité : [...]

**Approche**  
- Cadrage + KPIs  
- Analyse exploratoire  
- Modélisation / scoring (si applicable)  
- Interprétation / restitution

**Résultats clés**  
- [...]
- [...]
- [...]

**Limites**  
- [...]
- [...]

**Livrables**  
- Dashboard / rapport / application : [...]
""")

    st.divider()

    st.markdown("### 🧩 Projets (sélection)")
    st.caption("2–3 projets forts max.")

    for title in ["Projet 1 — Dashboard Power BI", "Projet 2 — Analyse marketing (GA4/Looker)", "Projet 3 — (optionnel)"]:
        with st.expander(f"📌 {title}", expanded=(title == "Projet 1 — Dashboard Power BI")):
            st.markdown("""
**Objectif** : [...]  
**Données** : [...]  
**Ce que j’ai fait** : [...]  
**Résultat / impact** : [...]  
**Livrables** : dashboard + synthèse décisionnelle  
**Lien (optionnel)** : [...]
""")

    st.divider()

    st.markdown("### 📸 Dashboards (captures)")
    st.caption("Ajoute 2–4 captures dans `assets/` et affiche-les ici.")

    # Exemple: décommente et renomme tes fichiers
    # st.image("assets/dashboard_1.png", caption="Dashboard — KPIs marketing", use_container_width=True)
    # st.image("assets/dashboard_2.png", caption="Dashboard — Suivi business", use_container_width=True)


# =========================
# Page 3 — CV
# =========================
with tab_cv:
    st.subheader("📄 CV")
    st.caption("Le CV est disponible au téléchargement et en aperçu.")

    st.markdown("#### ⬇️ Télécharger le CV")
    st.write("Place ton CV dans le repo : `assets/cv.pdf` (recommandé).")

    cv_bytes = read_file_bytes("assets/cv.pdf")

    if cv_bytes:
        st.download_button(
            label="Télécharger le CV (PDF)",
            data=cv_bytes,
            file_name="CV_Aimee.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        st.markdown("#### 👀 Aperçu")
        pdf_iframe(cv_bytes, height=820)
    else:
        st.warning("Je ne trouve pas `assets/cv.pdf`. Ajoute-le puis redeploie.")

st.divider()
st.caption("🧠 Reasoning Agent — Aimée | Candidature : Business Analyst / Data Analyst (junior)")
