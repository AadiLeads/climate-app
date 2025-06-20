from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import os
import torch
import pickle
from datasets import load_from_disk
from django.http import HttpResponse
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
dataset = load_from_disk(os.path.join(MODEL_DIR, 'dataset_with_embeddings.h5'))
dataset.load_faiss_index("embeddings", os.path.join(MODEL_DIR, "faiss_index.faiss"))

with open(os.path.join(MODEL_DIR, 'model_and_tokenizer.pkl'), 'rb') as f:
    obj = pickle.load(f)
    model = obj['model']
    model = model.to('cpu')
    tokenizer = obj['tokenizer']
    device = torch.device('cpu')

model.to(device)

def home(request):
    return HttpResponse("Welcome to the Recommender Home Page!")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask'])

@api_view(['POST'])
def recommend(request):
    try:
        query = request.data.get('query', '')
        k = int(request.data.get('k', 5))

        if not query.strip():
            return Response({"message": "Query cannot be empty."})

        # Generate query embedding
        query_embedding = get_embeddings([query]).cpu().numpy()

        # Retrieve from FAISS
        scores, samples = dataset.get_nearest_examples("embeddings", query_embedding, k=k * 10)

        # Calculate similarity scores
        scored_samples = [
            {
                "title": t,
                "description": d,
                "score": float(1 / (1 + s))
            }
            for t, d, s in zip(samples['title'], samples['cleaned_desc'], scores)
            if t.strip() and d.strip()
        ]

        # Remove duplicates by (title + description)
        seen = set()
        unique_samples = []
        for sample in scored_samples:
            key = (sample['title'].strip().lower(), sample['description'].strip().lower())
            if key not in seen:
                seen.add(key)
                unique_samples.append(sample)

        # Keyword + score filtering
        keyword = query.lower().strip()
        MIN_SCORE_THRESHOLD = 0.015

        filtered_samples = [
            sample for sample in unique_samples
            if sample["score"] >= MIN_SCORE_THRESHOLD and keyword in sample["description"].lower()
        ]

        if not filtered_samples:
            return Response({"message": "No relevant products found. Try searching something else."})

        # Sort and return top-k
        filtered_samples.sort(key=lambda x: x["score"], reverse=True)
        return Response(filtered_samples[:k])

    except Exception as e:
        from rest_framework import status
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)