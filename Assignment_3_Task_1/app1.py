import streamlit as st
import pickle
import torch
import torch.nn as nn
import re
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity


class NextTokenGen(nn.Module):
    def __init__(self, context_len, vocab_size, emb_dim, hidden_layer_size):
        super(NextTokenGen, self).__init__()
        self.context_len = context_len
        self.emb_dim = emb_dim
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.layer0 = nn.Linear(context_len * emb_dim, hidden_layer_size)
        self.layer1 = nn.Linear(hidden_layer_size, vocab_size)

    def forward(self, X, activation=None):
        X = self.embed(X)
        X = X.view(X.shape[0], self.context_len * self.emb_dim)
        if activation == 'relu':
            X = F.relu(self.layer0(X))
        elif activation == 'tanh':
            X = torch.tanh(self.layer0(X))
        else:
            X = self.layer0(X)

        X = self.layer1(X)
        return X

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\.{3,}', '.', text)
    text = re.sub(r'\n\s*\n', ' ' + '.' * 5 + ' ', text)
    text = re.sub(r'(\w)\n(\w)', r'\1 \2', text)
    text = re.sub(r'[^a-zA-Z0-9 \'\.]', ' ', text)
    text = re.sub(r'[\']', '', text)
    text = text.replace('\n', ' ')
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text

def paragraph_processing(text, context_len):
    context_padding = '.' * context_len
    paragraphs = text.split(".....")
    processed_paragraphs = [context_padding + para.strip() for para in paragraphs]
    return processed_paragraphs

def tokenization(paragraphs_txt, context_len):
    tokens = []
    for para in paragraphs_txt:
        para_tokens = re.findall(r'\b\w+\b|\.{' + str(context_len) + r'}|[.]', para)
        para_tokens = [token for token in para_tokens if token != '.' * context_len]
        tokens.extend(para_tokens)
    return tokens

def create_vocab(tokens):
    token_to_index = {
        '.': 0,
        ' ': 1,
    }
    unique_tokens = sorted(list(set(token for token in tokens if token not in token_to_index)))
    token_to_index.update({token: idx + 2 for idx, token in enumerate(unique_tokens)})
    index_to_token = {idx: token for token, idx in token_to_index.items()}
    return token_to_index, index_to_token, unique_tokens

def load_model_from_drive(model_name: str):
    model_file_path = os.path.join(os.path.dirname(__file__), f'model_name.pkl')
    st.write("Full path to model file:", model_file_path)
    with open(model_file_path, 'rb') as f:
        model_loaded = pickle.load(f)
    print('Model loaded successfully!')
    return model_loaded

def get_embedding(context_tokens, token_to_index, embeddings):

    known_embeds=[]

    for token in context_tokens:
        if token in list(token_to_index.keys()):
            token_idx = token_to_index[token]
            known_embeds.append(embeddings[token_idx])

    if known_embeds:
        return np.mean(known_embeds, axis=0).reshape(1, -1) 
    else:
        return None 
    
def find_closest_word(avg_embed, vocab_embeds, vocab_words):
    avg_embed=avg_embed.reshape(1,-1)
    vocab_embeds = np.array(vocab_embeds).reshape(-1, avg_embed.shape[1])
    similarities = cosine_similarity(avg_embed, vocab_embeds)
    closest_idx = np.argmax(similarities)
    closest_word = vocab_words[closest_idx]

    return closest_word

def predict_next_k_words(context, k, token_to_index, index_to_token, model):

    context_tokens = re.findall(r'\b\w+\b|[.]', context)
    context_indices = []

    avg_embedding = get_embedding(context_tokens, token_to_index, np.array(list(token_to_index.values())))

    for token in context_tokens:

        if token in list(token_to_index.keys()):
            context_indices.append(token_to_index[token])

        elif avg_embedding is not None:
            closest_word=find_closest_word(avg_embedding, list(token_to_index.values()), list(token_to_index.keys()))
            context_indices.append(token_to_index.get(closest_word, token_to_index[' ']))

        else:
            context_indices.append(token_to_index[' ']) 
            

    if len(context_indices)<context_len:
        context_indices=[1]*(context_len-len(context_indices))+context_indices
    else:
        context_indices=context_indices[-context_len:]

    predicted_words = []

    model.eval()
    with torch.no_grad():
        for _ in range(k):

            context_tensor=torch.tensor(context_indices, dtype=torch.int64).unsqueeze(0)

            output=model(context_tensor)

            next_word_index = torch.argmax(output, dim=1).item()
            next_word = index_to_token[next_word_index]

            predicted_words.append(next_word)

            context_indices.append(next_word_index)
            context_indices = context_indices[-context_len:]

    return ' '.join(predicted_words)

# Streamlit setup:

st.title("Next Word Prediction App")
# vocab_file_path = 'vocab.pkl'  # Update this path as necessary
vocab_file_path = os.path.join(os.path.dirname(__file__), 'vocab.pkl')
st.write("Full path to vocab file:", vocab_file_path)

# Directly load the vocabulary
with open(vocab_file_path, 'rb') as f:
    vocab = pickle.load(f)
    token_to_index = vocab
    index_to_token = {idx: token for token, idx in token_to_index.items()}


if 'model' not in st.session_state:
    st.session_state.model = None

emb_dim = st.selectbox("Embedding Dimension", ["Select", 64, 128], index=0)
context_len = st.selectbox("Context Length", ["Select", 5, 10], index=0)
activation = st.selectbox("Activation Function", ["Select", 'relu', 'tanh'], index=0)

if emb_dim != "Select" and context_len != "Select" and activation != "Select":
    if st.button("Load Model"):
        st.session_state.model = load_model_from_drive(f'emb{emb_dim}_context{context_len}_{activation}')
        if st.session_state.model is not None:
            st.success("Model loaded successfully!")
        else:
            st.error("Failed to load model.")
else:
    st.warning("Please select valid model parameters.")

context = st.text_input("Enter Prompt", value="")
num_words = st.number_input("Number of Words to Generate", min_value=1, max_value=100, value=10)

if st.button("Generate Text"):
    if st.session_state.model is None:
        st.warning("Please load a model first.")
    elif context:
        generated_text = predict_next_k_words(context, num_words, token_to_index, index_to_token, st.session_state.model)
        st.write("Generated Text:", generated_text)
    else:
        st.warning("Please enter a prompt to generate text.")
