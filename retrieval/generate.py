import re 
import time
from groq import APIConnectionError , BadRequestError , Groq , RateLimitError

from config import LLM_MODEL , NOT_FOUND , REWRITE_MODEL
from ingestion.clean import load_and_clean
from retrieval.retrieve import Retriever
from prompts import load_prompt


ANSWER_PROMPT = load_prompt("answer", not_found=NOT_FOUND)
REWRITE_PROMPT = load_prompt("rewrite")

client = Groq()
retriver = Retriever()
SCHEME_NAMES = load_and_clean()['Scheme Name'].tolist()


def norm(s): 
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
def chat(**kwargs): 
    for attempt in range(3): 
        try: 
            return client.chat.completions.create(**kwargs)
        except (BadRequestError , RateLimitError , APIConnectionError): 
            if attempt == 2 : 
                raise
            time.sleep(1.5 * (attempt + 1))


def build_context(hits): 
    return "\n\n".join(f"[{i}]\n{h['text']}" for i, h in enumerate(hits, 1))


def rewrite_question(question , history): 
    if not history : 
        return question

    chat_text = "\n".join(f"{m['role']}: {m['content'][:1200]}" for m in history)
    try: 
        response = chat(
            model=REWRITE_MODEL,
            temperature=0,
            max_completion_tokens=800,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user",
                 "content": f"Chat history:\n{chat_text}\n\nLast message: {question}"},
            ],
        )
        return (response.choices[0].message.content or "").strip() or question
    except Exception as e:
        print("REWRITE FAILED:", repr(e))
        return question


def search_hits(query , state): 
    q = norm(query)
    names = [n for n in SCHEME_NAMES  if norm(n) in q]
    names = [n for n in names
             if not any(n != m and norm(n) in norm(m) for m in names)]
    if len(names) < 2 : 
        return retriver.search(query , k = 5 , state = state)
    
    hits , seen = [] , set()
    for name in names[:3]:
        for h in retriver.search(query, k=3, state=state , scheme = name):
            key = (h["scheme_name"], h["section"], h["text"][:60])
            if key not in seen:
                seen.add(key)
                hits.append(h)
    return hits


def pick_sources(text , hits): 
    if text.strip() == NOT_FOUND: 
        return []
    answer_norm = norm(text)
    sources , seen = [] , set()
    for h in hits: 
        name = h["scheme_name"]
        if name not in seen and norm(name) in answer_norm: 
            seen.add(name)
            sources.append({"scheme" : name , "url" : h["url"]})
    if not sources:
        for h in hits:
            if h["scheme_name"] not in seen:
                seen.add(h["scheme_name"])
                sources.append({"scheme": h["scheme_name"], "url": h["url"]})
    return sources

def answer(question , state = None , history = None): 
    stand = rewrite_question(question , history)
    hits = search_hits(stand , state)
    context = build_context(hits)

    response = chat(
        model = LLM_MODEL , 
        temperature = 0 , 
        max_completion_tokens = 2000 , 
        reasoning_effort = "low" , 
        messages = [
            {"role" : "system" , "content" : ANSWER_PROMPT} , 
            {"role" : "user" , "content" : f"Context:\n{context}\n\nQuestion: {stand}"}
        ]
    )
    text = (response.choices[0].message.content or "").strip() or NOT_FOUND
    text = re.sub(r"\s*[【\[]\d+[】\]]", "", text)
    
    return {"answer": text, "sources": pick_sources(text, hits),
            "context": context, "query": stand}