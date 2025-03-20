# summarization_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import nltk
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import heapq
import re

# Initialize the FastAPI app
app = FastAPI()

# Download necessary NLTK data (can be done once, but included here for completeness)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the input data model for the API
class Article(BaseModel):
    text: str

@app.post("/summarize/")
async def summarize_article(article: Article):
    article_text = article.text.lower()
    
    # Tokenize the sentences
    sentence_list = nltk.sent_tokenize(article_text)

    # Load stopwords
    stopwords = [word for word in nltk.corpus.stopwords.words('english') if word not in ['not', 'but', 'while']]
    
    # Calculate word frequencies
    word_frequencies = defaultdict(int)
    for sentence in sentence_list:
        for word in nltk.word_tokenize(sentence):
            if word not in stopwords:  # Exclude stop words
                word_frequencies[word] += 1

    # Calculate sentence scores based on word frequencies
    sentence_scores = {}
    for sentence in sentence_list:
        for word in nltk.word_tokenize(sentence):
            if word in word_frequencies and len(sentence.split(' ')) < 30:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequencies[word]
                else:
                    sentence_scores[sentence] += word_frequencies[word]

    # Normalize sentence scores based on the sentence length
    for sentence in sentence_scores:
        sentence_scores[sentence] /= len(sentence.split(' '))

    # Use sentence transformers to get embeddings for sentences
    sentence_embeddings = model.encode(sentence_list)

    # Remove similar sentences based on cosine similarity threshold
    final_sentences = []
    used_indices = set()

    for idx, sentence in enumerate(sentence_list):
        if idx not in used_indices:
            final_sentences.append(sentence)
            similarities = util.pytorch_cos_sim(sentence_embeddings[idx], sentence_embeddings)
            for sim_idx, sim_score in enumerate(similarities[0]):
                if sim_score > 0.8:  # Similarity threshold
                    used_indices.add(sim_idx)

    # Get the top 5 sentences based on scores
    num_sentences = max(1, len(sentence_list) // 3)  # At least 1 sentence or 1/3 of the total
    summary = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    # Return the summary
    return {"summary": " ".join(summary)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)