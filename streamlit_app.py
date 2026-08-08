import json
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    with open("config.json") as f:
        config = json.load(f)

    class ElectraMCQ(nn.Module):
        def __init__(self, model_name, dropout):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(hidden_size, 128),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1),
            )
        def forward(self, input_ids, attention_mask):
            batch_size, num_opts, max_len = input_ids.shape
            input_ids = input_ids.view(batch_size * num_opts, max_len)
            attention_mask = attention_mask.view(batch_size * num_opts, max_len)
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state
            mask_exp = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
            return self.classifier(pooled).view(batch_size, num_opts)

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    model = ElectraMCQ(config["base_model"], dropout=config["dropout"])
    model.load_state_dict(torch.load("electra_base_best.pt", map_location="cpu"))
    model.eval()
    return model, tokenizer, config

model, tokenizer, config = load_model()
OPTIONS = config["options"]
MAX_LEN = config["max_len"]

st.title("MCQ Solver — Fine-Tuned ELECTRA")

if "clear_flag" not in st.session_state:
    st.session_state.clear_flag = False

left, right = st.columns([1.3, 1])

with left:
    prompt = st.text_area("Question / Prompt", height=90,
                           placeholder="e.g. Which of the following best describes...",
                           key="prompt_input")
    opt_values = []
    for letter in OPTIONS:
        val = st.text_input(f"Option {letter}", key=f"opt_{letter}")
        opt_values.append(val)

    btn_col1, btn_col2 = st.columns(2)
    clear_clicked = btn_col1.button("Clear", use_container_width=True)
    submit_clicked = btn_col2.button("Submit", use_container_width=True, type="primary")

with right:
    result_box = st.empty()
    breakdown_box = st.empty()
    result_box.text_area("Top-3 Predicted Answers", value="", height=90, disabled=True)

if clear_clicked:
    for letter in OPTIONS:
        st.session_state[f"opt_{letter}"] = ""
    st.session_state["prompt_input"] = ""
    st.rerun()

if submit_clicked:
    if not prompt.strip() or any(not o.strip() for o in opt_values):
        with right:
            st.warning("Please fill in the question and all 5 options.")
    else:
        input_ids_list, mask_list = [], []
        for opt in opt_values:
            enc = tokenizer(prompt, opt, truncation=True, max_length=MAX_LEN,
                             padding='max_length', return_tensors='pt')
            input_ids_list.append(enc['input_ids'].squeeze(0))
            mask_list.append(enc['attention_mask'].squeeze(0))

        X = torch.stack(input_ids_list).unsqueeze(0)
        mask = torch.stack(mask_list).unsqueeze(0)

        with torch.no_grad():
            logits = model(X, mask)
            probs = torch.softmax(logits, dim=-1).numpy()[0]

        ranked = sorted(zip(OPTIONS, probs), key=lambda x: -x[1])
        top3_text = " ".join(o for o, _ in ranked[:3])

        with right:
            result_box.text_area("Top-3 Predicted Answers", value=top3_text, height=90, disabled=True)
            breakdown_text = "\n".join(f"{o}: {p:.1%}" for o, p in ranked)
            st.text_area("Full Probability Breakdown", value=breakdown_text, height=160, disabled=True)