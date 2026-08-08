# streamlit_app.py
import json
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

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
prompt = st.text_area("Question / Prompt")
cols = st.columns(5)
opts = [cols[i].text_input(f"Option {OPTIONS[i]}") for i in range(5)]

if st.button("Predict") and prompt and all(opts):
    input_ids_list, mask_list = [], []
    for opt in opts:
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
    st.subheader(f"Top-3: {' '.join(o for o, _ in ranked[:3])}")
    for o, p in ranked:
        st.write(f"{o}: {p:.1%}")