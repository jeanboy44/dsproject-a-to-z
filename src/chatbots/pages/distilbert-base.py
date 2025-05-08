import streamlit as st
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def chatbot(text: str, context: str) -> str:
    inputs = tokenizer(text, context, return_tensors="pt")
    inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs_on_device)

    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    answer_span = inputs_on_device["input_ids"][0][start:end]
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(answer_span)
    )
    return answer


model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(
    model_name, low_cpu_mem_usage=False
)
model.to(device)

if "messages_bert" not in st.session_state:
    st.session_state.messages_bert = []

st.title("Question Answering with DistilBERT")

st.markdown("---")
context = st.text_area(
    "add context here",
    value="France is a country in Europe. Its capital is Paris. It is known for wine and cheese.",
)
st.info("example: what is the capital of France?")
st.markdown("---")

if prompt := st.chat_input("Ask me a question!"):
    answer = chatbot(prompt, context)
    st.write(answer)
    st.session_state.messages_bert.append({"role": "user", "content": prompt})
    st.session_state.messages_bert.append({"role": "assistant", "content": answer})
    for message in st.session_state.messages_bert:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
