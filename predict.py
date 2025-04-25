from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization

app = FastAPI()

# Load model and artifacts
model = tf.keras.models.load_model('movie_genre_model.keras')
mlb_classes = np.load('mlb_classes.npy', allow_pickle=True)
vectorizer_config = np.load('vectorizer_config.npy', allow_pickle=True).item()

# Rebuild vectorizer
vectorizer = TextVectorization(
    max_tokens=vectorizer_config['max_tokens'],
    output_sequence_length=vectorizer_config['output_sequence_length']
)
vectorizer.set_vocabulary(vectorizer_config['vocab'])

class MovieInput(BaseModel):
    title: str
    description: str

def predict_genres(title, description, threshold=0.4):
    text = (title + " " + description).lower()
    text_vectorized = vectorizer(np.array([text], dtype=str)).numpy()
    pred = model.predict(text_vectorized)
    pred_binary = (pred > threshold).astype(int)[0]
    genres = [mlb_classes[i] for i, val in enumerate(pred_binary) if val == 1]
    return genres

@app.post("/predict")
async def predict(movie: MovieInput):
    genres = predict_genres(movie.title, movie.description)
    return {"genres": genres}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)