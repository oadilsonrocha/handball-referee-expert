"""Handball Referee Expert – responde com base nos PDFs oficiais da IHF (versão Groq, gratuita)."""
import datetime as dt
import threading

import groq
import streamlit as st

import rag

# ---------------- Configurações (podem ser alteradas nos Secrets) ----------------
# Modelos tentados em ordem: se um atingir o limite gratuito, o app passa para o próximo.
MODELOS = [m.strip() for m in st.secrets.get(
    "MODELOS", "openai/gpt-oss-120b,llama-3.3-70b-versatile,llama-3.1-8b-instant").split(",")]
MODELO_BUSCA = st.secrets.get("MODELO_BUSCA", "llama-3.1-8b-instant")
MAX_PER_SESSION = int(st.secrets.get("MAX_PERGUNTAS_POR_USUARIO", 20))
MAX_PER_DAY = int(st.secrets.get("MAX_PERGUNTAS_POR_DIA", 200))
AUTOR_NOME = st.secrets.get("AUTOR_NOME", "Dr. Adilson Rocha Ferreira (Maceió, Alagoas, Brasil)")
AUTOR_LINK = st.secrets.get("AUTOR_LINK", "https://oadilsonrocha.netlify.app/")
TOP_K = 6

SYSTEM_PROMPT = """Você é o Handball Referee Expert, um assistente especialista nas regras oficiais de handebol da IHF.
Responda SOMENTE com base nos trechos de documentos fornecidos em cada pergunta.

Regras de resposta:
- Responda em português do Brasil (ou no idioma da pergunta), de forma clara e direta.
- Os documentos estão em inglês: traduza com fidelidade e use os termos técnicos consagrados
  (tiro de 7 metros, tiro livre, tempo técnico, exclusão de 2 minutos, desqualificação, área de gol).
- Cite o número da regra quando ele aparecer no trecho (ex.: Regra 8:5) e indique as fontes
  no formato [n], usando a numeração dos trechos.
- Se as regras forem diferentes entre modalidades (quadra, praia, cadeira de rodas), deixe isso claro.
  Se a pergunta não disser a modalidade, responda pelo handebol de quadra e avise.
- Se a resposta não estiver nos trechos, diga honestamente que não encontrou essa informação
  nos documentos. Nunca invente regras.
- Recuse educadamente perguntas sem relação com handebol."""

REWRITE_PROMPT = """You convert handball rules questions (usually in Portuguese) into English search keywords
for the official IHF rulebooks. Use official IHF terminology, for example:
exclusão/2 minutos -> 2-minute suspension; desqualificação -> disqualification; cartão vermelho -> red card disqualification;
tiro de 7 metros -> 7-metre throw; tiro livre -> free throw; tiro de saída -> throw-off; tiro de meta -> goalkeeper throw;
tiro lateral -> throw-in; tempo técnico/pedido de tempo -> team time-out; passos/andar -> steps taking more than three steps;
dois dribles -> double dribble bouncing; jogo passivo -> passive play forewarning signal; área de gol -> goal area;
goleiro -> goalkeeper; substituição -> substitution zone; bola -> ball size circumference weight; árbitro -> referees;
gol de giro -> spin shot; gol de dois pontos -> two points; especialista -> specialist; cadeira de rodas -> wheelchair.
Answer ONLY with one line of 6 to 20 English keywords, no explanations."""

st.set_page_config(page_title="Handball Referee Expert", page_icon="🤾", layout="centered")


# ---------------- Recursos compartilhados ----------------
@st.cache_resource
def get_client():
    key = st.secrets.get("GROQ_API_KEY")
    if not key:
        st.error("Configure a chave GROQ_API_KEY nos Secrets do app.")
        st.stop()
    return groq.Groq(api_key=key, max_retries=1)


@st.cache_resource
def get_index():
    return rag.load_index()


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


def english_query(client, question, prev_q):
    """Traduz a pergunta para palavras-chave em inglês (os documentos estão em inglês)."""
    text = f"Previous question: {prev_q}\nQuestion: {question}" if prev_q else question
    try:
        r = client.chat.completions.create(
            model=MODELO_BUSCA, temperature=0, max_completion_tokens=80,
            messages=[{"role": "system", "content": REWRITE_PROMPT},
                      {"role": "user", "content": text}])
        return (r.choices[0].message.content or "").strip()
    except groq.APIError as e:
        print("Falha ao traduzir a busca:", e)
        return ""


def build_messages(question, hits, history):
    ctx = "\n\n".join(
        f"[{i}] {h['title']} — página {h['page']}\n{h['text']}" for i, h in enumerate(hits, 1)
    ) or "(nenhum trecho encontrado)"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        msgs.append({"role": m["role"], "content": m["content"][:600]})
    msgs.append({"role": "user",
                 "content": f"Trechos dos documentos:\n\n{ctx}\n\nPergunta: {question}"})
    return msgs


def open_stream(client, messages):
    """Tenta cada modelo da lista até um responder."""
    last = None
    for model in MODELOS:
        extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
        try:
            return client.chat.completions.create(
                model=model, messages=messages, temperature=0.2,
                max_completion_tokens=1200, stream=True, **extra)
        except (groq.RateLimitError, groq.NotFoundError, groq.InternalServerError,
                groq.BadRequestError) as e:
            print(f"Modelo {model} indisponível: {e}")
            last = e
    raise last


def stream_text(stream):
    for ch in stream:
        if ch.choices and ch.choices[0].delta.content:
            yield ch.choices[0].delta.content


def sources_md(hits):
    return "\n\n".join(f"**[{i}]** {h['title']} — pág. {h['page']}" for i, h in enumerate(hits, 1))


# ---------------- Interface ----------------
st.title("🤾 Handball Referee Expert")
if AUTOR_NOME:
    if AUTOR_LINK:
        link = AUTOR_LINK if AUTOR_LINK.startswith("http") else "https://" + AUTOR_LINK
        st.markdown(f"Desenvolvido por [{AUTOR_NOME}]({link})")
    else:
        st.markdown(f"Desenvolvido por {AUTOR_NOME}")
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
            st.warning(f"Você atingiu o limite de {MAX_PER_SESSION} perguntas nesta sessão. "
                       "Volte mais tarde. 🙂")
            st.stop()
        if not take_daily_slot():
            st.warning("O chatbot atingiu o limite de perguntas de hoje. Tente novamente amanhã.")
            st.stop()

        history = st.session_state.messages[-4:]
        try:
            client = get_client()
            index = get_index()
            prev_q = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
            with st.spinner("Consultando as regras..."):
                eng = english_query(client, question, prev_q)
                modalities = None if mod == "Todas" else {mod, "Geral"}
                hits = index.search(f"{eng} {eng} {question}", k=TOP_K, modalities=modalities)
                stream = open_stream(client, build_messages(question, hits, history))
            answer = st.write_stream(stream_text(stream))
            src = sources_md(hits)
            if src:
                with st.expander("Fontes"):
                    st.markdown(src)
            st.session_state.messages += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer, "sources": src}]
            st.session_state.asked += 1
        except groq.RateLimitError:
            st.warning("Muitas perguntas ao mesmo tempo. Aguarde um minuto e tente de novo.")
        except groq.APIError as e:
            st.error("Não consegui responder agora. Tente novamente em instantes.")
            with st.expander("Detalhes técnicos"):
                st.code(f"{type(e).__name__}: {getattr(e, 'message', e)}"[:800])
            print("Erro na API Groq:", e)
