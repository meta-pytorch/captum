#!/usr/bin/env python
# coding: utf-8

# # Interpreting text models: IMDB sentiment analysis

# This notebook trains a small CNN sentiment classifier on a subset of the IMDB dataset and interprets predictions with `LayerIntegratedGradients`.
# 
# The vocabulary is built with the current TorchText APIs and is used before the model is initialized, so the embedding rows and token indices are always aligned. The notebook does not depend on the removed `torchtext.data.Field` / `LabelField` APIs or on a separately serialized model without its vocabulary.
# 
# **Note:** Before running this tutorial, please install torchtext. The raw ACL IMDB files are downloaded directly to avoid TorchText dataset / TorchData DataPipe version mismatches.

# In[ ]:


import os
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import captum
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

# In[2]:


for package in (captum, torch, torchtext):
    print(package.__name__, package.__version__)

# ## Load a small IMDB subset

# The full IMDB dataset can be used by increasing the subset limits. A smaller subset keeps this tutorial quick enough to run interactively. The notebook reads the raw ACL IMDB directory layout directly instead of `torchtext.datasets.IMDB`, since TorchText 0.18 expects an older TorchData DataPipe namespace that newer TorchData versions no longer expose.

# In[3]:


tokenizer = get_tokenizer("basic_english")
label_names = ["negative", "positive"]
DATA_ROOT = Path("data")
IMDB_DIR = DATA_ROOT / "aclImdb"
IMDB_ARCHIVE = DATA_ROOT / "aclImdb_v1.tar.gz"
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

def extract_archive_safely(archive_path, destination):
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if os.path.commonpath([destination, member_path]) != str(destination):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        archive.extractall(destination)

def ensure_imdb_dataset():
    if IMDB_DIR.exists():
        return
    DATA_ROOT.mkdir(exist_ok=True)
    if not IMDB_ARCHIVE.exists():
        urlretrieve(IMDB_URL, IMDB_ARCHIVE)
    extract_archive_safely(IMDB_ARCHIVE, DATA_ROOT)

def load_rows(split, limit):
    ensure_imdb_dataset()
    rows = []
    per_label_limit = limit // 2
    extra_positive = limit % 2
    for label_index, label_dir in enumerate(("neg", "pos")):
        label_limit = per_label_limit + (extra_positive if label_dir == "pos" else 0)
        paths = sorted((IMDB_DIR / split / label_dir).glob("*.txt"))[:label_limit]
        rows.extend((label_index, path.read_text(encoding="utf-8")) for path in paths)
    return rows

train_rows = load_rows("train", 4000)
test_rows = load_rows("test", 500)

print("Train examples:", len(train_rows))
print("Test examples:", len(test_rows))

# ## Build the vocabulary before constructing the model

# In[4]:


UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
MAX_TOKENS = 20000
MAX_LEN = 256
MIN_LEN = 5

def yield_tokens(rows):
    for _, text in rows:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(
    yield_tokens(train_rows),
    specials=[UNK_TOKEN, PAD_TOKEN],
    max_tokens=MAX_TOKENS,
)
vocab.set_default_index(vocab[UNK_TOKEN])
PAD_IDX = vocab[PAD_TOKEN]

print("Vocabulary size:", len(vocab))
print("Padding index:", PAD_IDX)

# In[5]:


def encode_text(text):
    token_ids = vocab(tokenizer(text))[:MAX_LEN]
    if len(token_ids) < MIN_LEN:
        token_ids += [PAD_IDX] * (MIN_LEN - len(token_ids))
    return torch.tensor(token_ids, dtype=torch.long)

def collate_batch(batch):
    labels = torch.tensor([label for label, _ in batch], dtype=torch.long)
    texts = [encode_text(text) for _, text in batch]
    texts = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    return texts, labels

train_loader = DataLoader(train_rows, batch_size=64, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_rows, batch_size=128, shuffle=False, collate_fn=collate_batch)

# ## Define and train a CNN sentiment classifier

# In[6]:


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=n_filters,
                kernel_size=(filter_size, embedding_dim),
            )
            for filter_size in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# In[7]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(
    vocab_size=len(vocab),
    embedding_dim=100,
    n_filters=100,
    filter_sizes=[3, 4, 5],
    output_dim=2,
    dropout=0.5,
    pad_idx=PAD_IDX,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    model.train()
    total_loss = 0.0
    for text, labels in train_loader:
        text = text.to(device)
        labels = labels.to(device)

        logits = model(text)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: loss = {total_loss / len(train_loader):.4f}")

# In[8]:


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for text, labels in test_loader:
        text = text.to(device)
        labels = labels.to(device)
        preds = model(text).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

print("Test accuracy:", correct / total)

# ## Attribute sentiment predictions

# Token ids are discrete, so gradients are computed with respect to the embedding layer output rather than the token id tensor. The padding token provides a natural reference token for the baseline sentence.

# In[9]:


def forward_with_softmax(input_indices):
    return F.softmax(model(input_indices), dim=1)

lig = LayerIntegratedGradients(forward_with_softmax, model.embedding)
token_reference = TokenReferenceBase(reference_token_idx=PAD_IDX)
vis_data_records_ig = []

# In[10]:


def add_attributions_to_visualizer(attributions, tokens, pred_prob, pred_idx, true_label, delta):
    attributions = attributions.sum(dim=2).squeeze(0)
    norm = torch.norm(attributions)
    if norm > 0:
        attributions = attributions / norm

    attributions = attributions.detach().cpu()
    true_class = label_names[true_label] if true_label is not None else "unknown"

    vis_data_records_ig.append(
        visualization.VisualizationDataRecord(
            attributions,
            pred_prob,
            label_names[pred_idx],
            true_class,
            label_names[pred_idx],
            attributions.sum().item(),
            tokens,
            delta,
        )
    )

def interpret_sentence(sentence, true_label=None):
    model.zero_grad()

    tokens = tokenizer(sentence)[:MAX_LEN]
    if len(tokens) < MIN_LEN:
        tokens += [PAD_TOKEN] * (MIN_LEN - len(tokens))

    input_indices = torch.tensor([vocab(tokens)], dtype=torch.long, device=device)
    reference_indices = token_reference.generate_reference(
        input_indices.shape[1], device=device
    ).unsqueeze(0)

    probs = forward_with_softmax(input_indices)
    pred_prob, pred_idx_tensor = probs.max(dim=1)
    pred_idx = pred_idx_tensor.item()

    attributions_ig, delta = lig.attribute(
        input_indices,
        baselines=reference_indices,
        target=pred_idx,
        n_steps=50,
        return_convergence_delta=True,
    )

    print(
        "pred:",
        label_names[pred_idx],
        f"({pred_prob.item():.2f})",
        "delta:",
        abs(delta.item()),
    )
    add_attributions_to_visualizer(
        attributions_ig,
        tokens,
        pred_prob.item(),
        pred_idx,
        true_label,
        delta.item(),
    )

# Below cells interpret a few handcrafted review phrases.

# In[11]:


interpret_sentence("It was a fantastic performance!", true_label=1)
interpret_sentence("Best film ever", true_label=1)
interpret_sentence("Such a great show!", true_label=1)
interpret_sentence("It was a horrible movie", true_label=0)
interpret_sentence("I have never watched something as bad", true_label=0)
interpret_sentence("That is a terrible movie.", true_label=0)

# ## Visualize token attributions

# In[12]:


visualization.visualize_text(vis_data_records_ig)
