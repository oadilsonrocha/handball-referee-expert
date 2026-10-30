"""Handball Referee Expert – responde com base nos PDFs oficiais da IHF."""
import datetime as dt
import threading

import streamlit as st
from google import genai
from google.genai import errors, types

import rag

# ---------------- Configurações (podem ser alteradas nos Secrets) ----------------
CHAT_MODEL = st.secrets.get("GEMINI_MODEL", "gemini-flash-latest")
MAX_PER_SESSION = int(st.secrets.get("MAX_PERGUNTAS_POR_USUARIO", 20))
MAX_PER_DAY = int(st.secrets.get("MAX_PERGUNTAS_POR_DIA", 400))
TOP_K = 6

SYSTEM_PROMPT = """Você é o Handball Referee Expert, um assistente especialista nas regras oficiais de handebol da IHF.
Responda SOMENTE com base nos trechos de documentos fornecidos em cada pergunta.

Regras de resposta:
- Responda no idioma da pergunta (normalmente português do Brasil), de forma clara e direta.
- Os documentos estão em inglês: traduza com fidelidade e mantenha termos técnicos consagrados
  (ex.: tiro de 7 metros, tempo técnico, exclusão de 2 minutos).
- Cite o número da regra quando ele aparecer no trecho (ex.: Regra 8:5) e indique as fontes
  no formato [n], usando a numeração dos trechos.
- Se as regras forem diferentes entre modalidades (quadra, praia, cadeira de rodas), deixe isso claro.
  Se a pergunta não disser a modalidade, responda pelo handebol de quadra e avise.
- Se a resposta não estiver nos trechos, diga honestamente que não encontrou essa informação
  nos documentos. Nunca invente regras.
- Recuse educadamente perguntas sem relação com handebol."""

st.set_page_config(page_title="Handball Referee Expert", page_icon="🤾", layout="centered")


# ---------------- Recursos compartilhados ----------------
@st.cache_resource
def get_client():
    key = st.secrets.get("GEMINI_API_KEY")
    if not key:
        st.error("Configure a chave GEMINI_API_KEY nos Secrets do app.")
        st.stop()
    return genai.Client(api_key=key)


@st.cache_resource(show_spinner=False)
def get_index():
    chunks = rag.load_chunks()
    emb = rag.load_embeddings(len(chunks))
    if emb is None:
        bar = st.progress(0.0, text="Preparando o índice dos documentos (só na primeira vez)...")
        emb = rag.build_embeddings(get_client(), chunks,
                                   progress=lambda p: bar.progress(p, text="Indexando documentos..."))
        bar.empty()
    return chunks, emb


@st.cache_resource
def daily_counter():
    """Contador global (todos os usuários) para proteger a cota gratuita."""
    return {"day": dt.date.today(), "count": 0, "lock": threading.Lock()}


def take_daily_slot() -> bool:
    c = daily_counter()
    with c["lock"]:
        today = dt.date.today()
        if c["day"] != today:
            c["day"], c["count"] = today, 0
        if c["count"] >= MAX_PER_DAY:
            return False
        c["count"] += 1
        return True


def build_prompt(question, hits, history):
    ctx = "\n\n".join(
        f"[{i}] {h['title']} — página {h['page']}\n{h['text']}" for i, h in enumerate(hits, 1)
    )
    hist = ""
    if history:
        hist = "Conversa anterior (para contexto):\n" + "\n".join(
            f"{'Usuário' if m['role'] == 'user' else 'Assistente'}: {m['content'][:600]}"
            for m in history
        ) + "\n\n"
    return f"{hist}Trechos dos documentos:\n\n{ctx}\n\nPergunta: {question}"


def sources_md(hits):
    lines = [f"**[{i}]** {h['title']} — pág. {h['page']}" for i, h in enumerate(hits, 1)]
    return "\n\n".join(lines)


# ---------------- Interface ----------------
st.title("🤾 Handball Referee Expert")
st.caption("Tire dúvidas sobre as regras oficiais da IHF. As respostas vêm dos documentos oficiais, "
           "com indicação da fonte.")

with st.sidebar:
    st.header("Filtro")
    mod = st.radio("Modalidade", ["Todas", "Quadra", "Praia", "Cadeira de rodas"])
    st.divider()
    st.markdown(
        "**Documentos consultados**\n"
        "- Regras: quadra, praia, cadeira de rodas (6x6 e 4x4)\n"
        "- Classificação – cadeira de rodas\n"
        "- Tempo técnico eletrônico\n"
        "- Replay de vídeo\n"
        "- Equipamentos de proteção"
    )
    st.divider()
    st.caption("Ferramenta de consulta. Em caso de dúvida oficial, prevalece o texto da IHF.")
    if st.button("Nova conversa"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "asked" not in st.session_state:
    st.session_state.asked = 0

if not st.session_state.messages:
    st.info("Exemplos: *Quanto tempo dura uma exclusão?* · *Quando é marcado um tiro de 7 metros?* · "
            "*Como funciona o gol de giro no beach handball?*")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Fontes"):
                st.markdown(m["sources"])

question = st.chat_input("Digite sua pergunta sobre as regras...")

if question:
    question = question.strip()[:800]
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if st.session_state.asked >= MAX_PER_SESSION:
            msg = (f"Você atingiu o limite de {MAX_PER_SESSION} perguntas nesta sessão. "
                   "Volte mais tarde. 🙂")
            st.warning(msg)
            st.stop()
        if not take_daily_slot():
            st.warning("O chatbot atingiu o limite de perguntas de hoje. Tente novamente amanhã.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.asked += 1
        history = st.session_state.messages[-5:-1]

        try:
            client = get_client()
            chunks, emb = get_index()
            prev_q = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
            search_q = f"{prev_q}\n{question}" if prev_q and len(question) < 80 else question
            modalities = None if mod == "Todas" else {mod, "Geral"}
            with st.spinner("Consultando as regras..."):
                hits = rag.search(client, search_q, chunks, emb, k=TOP_K, modalities=modalities)

            stream = client.models.generate_content_stream(
                model=CHAT_MODEL,
                contents=build_prompt(question, hits, history),
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT,
                                                   temperature=0.2),
            )
            answer = st.write_stream(ch.text for ch in stream if ch.text)
            src = sources_md(hits)
            with st.expander("Fontes"):
                st.markdown(src)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": src})
        except errors.APIError as e:
            st.session_state.messages.pop()
            st.session_state.asked -= 1
            if e.code == 429:
                st.warning("Muitas perguntas ao mesmo tempo. Aguarde um minuto e tente de novo.")
            else:
                st.error("Não consegui responder agora. Tente novamente em instantes.")
                print("Erro na API Gemini:", e)
