import os
import json
import re
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# the two models to compare
MODELS = {"Low cost (gpt-4o-mini)": "gpt-4o-mini", "Higher cost (gpt-4o)": "gpt-4o"}

# connect to openai
client = OpenAI(api_key=st.secrets["OPEN_AI_KEY"])

# load the saved stories and embeddings
FOLDER = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load():
    news = pd.read_pickle(os.path.join(FOLDER, "news_db.pkl"))
    emb = np.load(os.path.join(FOLDER, "news_emb.npy"))
    news["companies"] = news["companies"].apply(lambda cs: [c.strip() for c in cs])
    return news, emb


news, emb = load()
company_names = sorted({c for cs in news["companies"] for c in cs})
ALIASES = {"meta": "Facebook", "google": "Alphabet"}


def embed(text):
    out = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(out.data[0].embedding, dtype=np.float32)


def find_company(text):
# look for a company name in the text
    low = text.lower()
    for a, name in ALIASES.items():
        if re.search(r"\b" + a + r"\b", low):
            return name
    for name in company_names:
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", low):
            return name
    return None


def remove_repeats(ids):
# drop stories too close to one already kept
    kept = []
    for i in ids:
        if all(float(emb[i] @ emb[j]) < 0.85 for j in kept):
            kept.append(i)
    return kept


def show(i, **extra):
    row = news.iloc[i]
    d = {"title": row["title"], "date": str(row["Date"])[:10], "companies": row["companies"],
         "url": row["urls"][0], "summary": (row["desc"] + " " + row["body"])[:300]}
    d.update(extra)
    return d


def search_news(query):
# find stories about a company or topic
    company = find_company(query)
    scores = emb @ embed(query)
    if company:
        scores = np.where(news["companies"].apply(lambda cs: company in cs), scores, -1)
    best = [int(i) for i in np.argsort(-scores)[:20] if scores[i] > -1]
# no company and a low score means no coverage
    if not best or (company is None and scores[best[0]] < 0.40):
        return {"results": [], "note": "No coverage of that in the news."}
    return {"results": [show(i) for i in remove_repeats(best)[:6]]}


@st.cache_data
def rank_interesting(model, company=""):
# first pass: 20 stories closest to legal news
    legal = "lawsuit, investigation, antitrust, fine, settlement, merger, acquisition, data breach, court ruling"
    scores = emb @ embed(legal)
    if company:
        scores = np.where(news["companies"].apply(lambda cs: company in cs), scores, -1)
    best = [int(i) for i in np.argsort(-scores)[:20] if scores[i] > -1]
    best = remove_repeats(best)
# second pass: the model scores each story from 1 to 10
    items = [{"id": k, "title": news.iloc[i]["title"], "text": news.iloc[i]["desc"][:300]}
             for k, i in enumerate(best)]
    rules = ("You score news for a global law firm. Give each story a score from 1 to 10 for how "
             "interesting it is to a law firm. High: lawsuits, court rulings, regulators, antitrust, fines, "
             "data breaches. Medium: mergers, acquisitions, financing. Low: reviews, coupons, stock tips, "
             "roundups. If a story is about the same event as another story in the list, "
             "put that other story's id in same_as, otherwise use null. "
             'Reply in JSON: {"scores": [{"id": 0, "score": 7, "why": "one short sentence", "same_as": null}]}')
    resp = client.chat.completions.create(
        model=model, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": rules}, {"role": "user", "content": json.dumps(items)}])
    rated = {int(s["id"]): s for s in json.loads(resp.choices[0].message.content)["scores"]}
    out = []
    for k, i in enumerate(best):
        r = rated.get(k, {})
        out.append(show(i, score=r.get("score", 0), why=r.get("why", ""), k=k, same=r.get("same_as")))
# best score first
    out.sort(key=lambda d: -d["score"])
# keep only interesting stories and skip repeats of the same event
    final = []
    for d in out:
        if d["score"] < 4:
            continue
        repeat = any(str(d["same"]) == str(f["k"]) or str(f["same"]) == str(d["k"]) for f in final)
        if not repeat:
            final.append(d)
    return final


TOOLS = [
    {"type": "function", "function": {
        "name": "search_news", "description": "Find news about a company or topic.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "most_interesting", "description": "Get the most interesting news, ranked best first.",
        "parameters": {"type": "object", "properties": {"company": {"type": "string"}}}}},
]

SYSTEM = ("You are a news assistant for a law firm. Only use what the tools return. "
          "If a tool returns no results, say there is no coverage. "
          "Show each story as a link like [title](url) with its date. "
          "For ranked news keep the order and say why each story matters. Do not make things up.")


def ask(model, history, question):
# send the question to the model and run the tools it picks
    msgs = [{"role": "system", "content": SYSTEM}] + history + [{"role": "user", "content": question}]
    for _ in range(4):
        reply = client.chat.completions.create(model=model, messages=msgs, tools=TOOLS, temperature=0)
        msg = reply.choices[0].message
        if not msg.tool_calls:
            return msg.content
        msgs.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            if call.function.name == "search_news":
                result = search_news(args["query"])
            else:
                company = find_company(question) or ""
                result = {"results": rank_interesting(model, company)[:10]}
            msgs.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    return "Sorry, that did not work."


st.title("Client News Bot")
st.caption("Answers only from the news articles in the dataset.")
model = MODELS[st.sidebar.radio("Model", list(MODELS))]

if "chat" not in st.session_state:
    st.session_state.chat = []
for m in st.session_state.chat:
    st.chat_message(m["role"]).write(m["content"])

question = st.chat_input("Ask about the news...")
if question:
    st.chat_message("user").write(question)
    answer = ask(model, st.session_state.chat[-6:], question)
    st.chat_message("assistant").write(answer)
    st.session_state.chat += [{"role": "user", "content": question},
                              {"role": "assistant", "content": answer}]