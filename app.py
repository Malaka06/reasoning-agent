import streamlit as st
import requests
import base64
from pathlib import Path

# =========================
# CONFIGURATION PAGE
# =========================
st.set_page_config(
    page_title="Reasoning Agent – Aimée | Data & Business Analyst",
    layout="wide",
)

# =========================
# HEADER
# =========================
st.title("🧠 Reasoning Agent")
st.caption("Simulation de ma façon de raisonner — Data Analyst / Business Analyst")

st.info(
    "👋 **Recruteurs** : cette application ne vise pas à donner des réponses parfaites, "
    "mais à **montrer comment je raisonne**, structure mes décisions et arbitre entre plusieurs options "
    "face à des problématiques data et métier réelles."
)

# =========================
# TABS
# =========================
tab_agent, tab_projects, tab_cv = st.tabs(
    ["🤖 Agent de raisonnement", "📊 Projets & Mémoire", "📄 CV"]
)

# ======================================================
# TAB 1 — AGENT
# ======================================================
with tab_agent:
    st.subheader("🤖 Posez une question à mon agent")

    st.markdown("""
    L’agent simule **ma manière de penser** en tant que Data / Business Analyst :
    - réponse **orientée décision**
    - raisonnement **structuré**
    - alternatives **volontairement écartées**
    """)

    examples = [
        "Qui es-tu ?",
        "Pourquoi devrions-nous recruter Aimée ?",
        "Que fais-tu si les données sont de mauvaise qualité ?",
        "Comment traduis-tu un besoin métier flou en analyse data ?",
        "Comment choisis-tu les KPIs pour un dashboard marketing ?",
    ]

    selected = st.radio("💡 Exemples de questions :", examples)

    question = st.text_area(
        "Votre question",
        value=selected,
        height=120,
    )

    # =========================
    # LLM CONFIG
    # =========================
    MODEL_ID = "moonshotai/Kimi-K2-Instruct-0905"
    HF_URL = "https://router.huggingface.co/v1/chat/completions"

    HF_HEADERS = {
        "Authorization": f"Bearer {st.secrets['HF_API_TOKEN']}",
        "Content-Type": "application/json",
    }

    def call_llm(user_question: str) -> str:
        system_prompt = (
            "Tu es l'assistant officiel d'Aimée.\n\n"
            "Règles strictes :\n"
            "- Tu parles TOUJOURS à la première personne comme si tu étais Aimée\n"
            "- Si on te demande 'qui es-tu', tu réponds : "
            "'Je suis l’agent de raisonnement d’Aimée, conçu pour expliquer sa manière de penser.'\n"
            "- Tu peux répondre à des questions personnelles professionnelles "
            "(parcours, compétences, centres d’intérêt : data, marketing, jeux, voyages, cuisine)\n\n"
            "Structure OBLIGATOIRE de la réponse :\n\n"
            "Réponse :\n"
            "(1–2 phrases claires, orientées décision)\n\n"
            "Raisonnement :\n"
            "- étape 1 : clarification / diagnostic\n"
            "- étape 2 : analyse et arbitrages\n"
            "- étape 3 : décision finale\n\n"
            "Preuves / critères :\n"
            "- faits concrets, compétences, outils\n\n"
            "Alternatives :\n"
            "- option 1 + pourquoi je ne la choisis pas\n"
            "- option 2 + pourquoi je ne la choisis pas\n\n"
            "Conclusion métier (prochaine action) :\n"
            "- ce que le recruteur peut tester ou attendre concrètement\n\n"
            "Style : professionnel, clair, orienté business, sans jargon inutile."
        )

        payload = {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question},
            ],
            "temperature": 0.4,
        }

        r = requests.post(HF_URL, headers=HF_HEADERS, json=payload, timeout=90)

        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} - {r.text}")

        return r.json()["choices"][0]["message"]["content"]

    if st.button("🚀 Générer", use_container_width=True):
        if not question.strip():
            st.warning("Merci d’écrire une question.")
            st.stop()

        with st.spinner("Analyse et raisonnement en cours..."):
            try:
                answer = call_llm(question)
            except Exception as e:
                st.error(f"Erreur IA : {e}")
                st.stop()

        st.markdown("---")
        st.markdown(answer)

# ======================================================
# TAB 2 — PROJETS / MÉMOIRE
# ======================================================
with tab_projects:
    st.subheader("📊 Projets, mémoire & dashboards")

    st.markdown("""
    ### 🎓 Mémoire / Projet de fin d’études
    - Problématique métier claire
    - Données imparfaites mais exploitables
    - Choix méthodologiques justifiés
    - Résultats actionnables
    """)

    st.markdown("""
    ### 📈 Dashboards
    - Power BI / Looker Studio
    - KPIs compréhensibles par des non-tech
    - Orientation décision (marketing, performance, produit)
    """)

    st.info(
        "👉 Cette section peut être enrichie avec des captures d’écran, "
        "liens vers dashboards ou notebooks."
    )

# ======================================================
# TAB 3 — CV
# ======================================================
with tab_cv:
    st.subheader("📄 CV — Aimée")
    st.caption("Business Analyst / Data Analyst junior")

    st.markdown("""
    Vous pouvez :
    - télécharger mon CV
    - le consulter directement ci-dessous
    """)

    cv_path = Path("assets/CV_Aimee.pdf")

    if not cv_path.exists():
        st.error("❌ CV introuvable. Vérifiez : assets/CV_Aimee.pdf")
    else:
        with open(cv_path, "rb") as f:
            st.download_button(
                label="⬇️ Télécharger le CV (PDF)",
                data=f,
                file_name="CV_Aimee.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with st.expander("👀 Aperçu du CV"):
            with open(cv_path, "rb") as f:
                pdf_bytes = f.read()

            b64 = base64.b64encode(pdf_bytes).decode("utf-8")

            st.markdown(
                f"""
                <iframe
                    src="data:application/pdf;base64,{b64}"
                    width="100%"
                    height="800px"
                    style="border-radius:12px; border:1px solid #eaeaea;"
                ></iframe>
                """,
                unsafe_allow_html=True,
            )

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "🧠 Reasoning Agent — conçu pour montrer **comment je pense**, pas seulement ce que je sais."
)
