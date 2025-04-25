# Movie Genre Classification

This project implements a multi-label movie genre classification system using a TensorFlow neural network. It predicts multiple genres (e.g., Action, Sci-Fi) for movies based on their titles and descriptions. The project includes a Jupyter Notebook for training and evaluation, a FastAPI backend for predictions, and visualizations to analyze model performance.

## Features
- **Data Preprocessing**: Cleans and tokenizes movie titles and descriptions.
- **Model**: Uses TextVectorization, Embedding, and Dense layers with dropout for robust predictions.
- **Training**: Trains on a dataset of 54,214 movies with 27 genres.
- **Evaluation**: Computes precision, recall, and F1-score with confusion matrix visualizations.
- **API**: FastAPI backend (`predict.py`) for real-time genre predictions.
- **Visualizations**: Includes genre distribution, training loss/precision plots, and per-genre confusion matrices.

## Dataset
- **Training Data**: `train_data.txt` (54,214 samples, format: `ID ::: TITLE ::: GENRE ::: DESCRIPTION`)
- **Test Data**: `test_data.txt` (54,200 samples, format: `ID ::: TITLE ::: DESCRIPTION`)
- **Source**: Not specified (replace with dataset source if applicable, e.g., IMDb or Kaggle).

## Requirements
- Python 3.9+
- Dependencies listed in `requirements.txt`:
  ```
  pandas>=1.5.0
  numpy>=1.23.0
  scikit-learn>=1.2.0
  tensorflow==2.14.0
  fastapi>=0.85.0
  uvicorn>=0.18.0
  matplotlib>=3.5.0
  seaborn>=0.11.0
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/movie-genre-classification.git
   cd movie-genre-classification
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage
### Running the Notebook
1. Open `movie_genre_classification_notebook.ipynb` in Jupyter Notebook or Google Colab.
2. Ensure `train_data.txt` and `test_data.txt` are in the same directory.
3. Run all cells to:
   - Load and preprocess data
   - Train the model
   - Visualize training progress and genre distribution
   - Evaluate performance
   - Test predictions (e.g., for "The Matrix")
4. Artifacts (`movie_genre_model.keras`, `mlb_classes.npy`, `vectorizer_config.npy`) are saved for deployment.

### Running the API
1. Run the FastAPI backend:
   ```bash
   python predict.py
   ```
2. Test the API with a POST request:
   ```bash
   curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"title":"The Matrix","description":"A hacker discovers a mysterious reality."}'
   ```
   Expected output: `{"genres":["action","sci-fi"]}`

## Files
- `movie_genre_classification_notebook.ipynb`: Jupyter Notebook for training and evaluation.
- `predict.py`: FastAPI backend for genre predictions.
- `train_data.txt`, `test_data.txt`: Training and test datasets.
- `movie_genre_model.keras`: Trained model.
- `mlb_classes.npy`: Genre encoder.
- `vectorizer_config.npy`: Text vectorizer configuration.
- `requirements.txt`: Python dependencies.

## Performance
- Validation Precision: ~0.70-0.80
- Validation Recall: ~0.60-0.70
- Validation F1-Score: ~0.65-0.75
- Note: Performance may vary based on dataset quality and training duration.

## Visualizations
- **Genre Distribution**: Bar plot of genre frequencies.
- **Training Plots**: Loss and precision over epochs.
- **Confusion Matrix**: Per-genre performance heatmaps.

## Author
- **Name**: Janapati Venkata Sriveda Manaswi
- **Contact**: (Add your email or GitHub profile if desired)

## License
This project is licensed under the MIT License - see below for details.

```
MIT License

Copyright (c) 2025 Janapati Venkata Sriveda Manaswi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## Acknowledgments
- Built with TensorFlow, FastAPI, and scikit-learn.
- Inspired by multi-label text classification tasks.