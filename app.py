
import gradio as gr

from ingestion.clean import load_and_clean
from retrieval.generate import answer
states = ["ALL"] + sorted(load_and_clean()["State / UT / Ministry"].dropna().unique())

EXAMPLES = [
    "What financial assistance is available for journalists or their families in India who are facing severe hardship?",
    "Which pension schemes are there for widows?",
    "Loan schemes for farmers",
    "Scholarships for engineering students",
]


def text(content): 
    if isinstance(content , str): 
        return content
    if isinstance(content , list): 
        return " ".join(p.get("text" , "") if isinstance(p , dict) else str(p) for p in content)


def to_history(history):

    out  = []
    for m in history or []: 
        if isinstance(m , dict): 
            out.append({"role" : m["role"] , "content" : text(m["content"])})
        else: 
            user , bot = m 
            if user: 
                out.append({"role" : "user" , "content" : text(user)})
            if bot: 
                out.append({"role" : "assistant" , "content" : text(bot)})

    for m in out:
        if m["role"] == "assistant":
            m["content"] = (m["content"].split("\n\n**Sources**")[0]
                                        .split("\n\n*Searched for:")[0])
    return out[-6:]

def respond(message , history , state): 
    message = message.strip().strip('“”"')
    state = None if state == "ALL" else state
    try: 
        result = answer(message , state = state , history = to_history(history))

    except Exception as e:
        print("ERROR:", repr(e))
        return "Something went wrong. Please try again in a moment."

    text = result["answer"]
    query = result.get("query")
    if query and query.strip().lower() != message.strip().lower(): 
        text += f"\n\n*Searched for: {query}*"

    if result["sources"]:
        links = "\n".join(f"- [{s['scheme']}]({s['url']})" for s in result["sources"])
        text += f"\n\n**Sources**\n{links}"
    return text

demo = gr.ChatInterface(
    fn = respond , 
    additional_inputs=[gr.Dropdown(states , value="ALL" , label="State / UT / Ministry")] , 
    examples=[[e , "ALL"] for e in EXAMPLES] , 
    cache_examples=False , 
    title= "Government Scheme Assistant" , 
    description="Answers come only from MyScheme data. Always confirm details on the official page.",
)
if __name__ == "__main__":
    demo.launch()