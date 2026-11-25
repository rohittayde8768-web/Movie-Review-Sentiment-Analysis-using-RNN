# 🎬 Movie Review Sentiment Analysis Using Simple RNN

An AI-powered web application that analyzes movie reviews and predicts whether the expressed sentiment is **Positive** or **Negative**. The project uses Natural Language Processing (NLP) and a Simple Recurrent Neural Network (RNN) built with TensorFlow/Keras, with an interactive interface developed using Streamlit.

---

## 📌 Project Overview

Understanding customer opinions from written reviews is an important NLP task. In this project, a deep learning model learns patterns in movie-review text and predicts the sentiment of a new review.

The model is trained using the **IMDB Movie Reviews dataset** available through Keras. Users can enter a review in the Streamlit application and receive a sentiment prediction along with the model's output score.

## ✨ Features

- Positive and negative movie-review classification.
- Text preprocessing and integer sequence encoding.
- Word Embedding layer to learn word representations.
- Simple RNN architecture for sequential text processing.
- Interactive Streamlit web application.
- Displays positive and negative prediction scores.
- Supports custom user-entered reviews.

## 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Building, training, and loading the deep learning model |
| Simple RNN | Learning sequential patterns in review text |
| Embedding | Learning dense word representations |
| NLP | Text preprocessing and sentiment analysis |
| IMDB dataset | Movie-review training and testing data |
| NumPy | Numerical operations |
| Streamlit | Interactive web application |

## 🧠 Model Architecture

The neural network uses the following layers:

| Layer | Configuration | Purpose |
|---|---|---|
| Input | 500 tokens | Receives padded review sequences |
| Embedding | 10,000 vocabulary entries, 64 dimensions | Converts token IDs into dense vectors |
| SimpleRNN | 64 units | Learns sequential information from the review |
| Dense | 1 unit, sigmoid activation | Produces a sentiment score |

**Compilation configuration**
- Optimizer: Adam
- Loss function: Binary Crossentropy
- Metric: Accuracy
- Early stopping: Monitors validation loss, with patience of 5 and best-weight restoration.
- Maximum training epochs in the supplied training script: 5
- Batch size: 128

The sigmoid output is interpreted using a threshold of `0.5`:
- Score **≥ 0.5** → Positive
- Score **< 0.5** → Negative

The output score is a model score, not a guarantee that the prediction is correct.

## 📂 Project Structure

Keep the following files in the same project directory:

```text
Movie-Review-Sentiment-Analysis-Using-RNN/
│
├── app.py              # Streamlit application
├── train.py            # Model training script
├── prediction.py       # Prediction utility script
├── rnn_model.keras     # Saved trained RNN model
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

> If your training file has a different name, update the command below to match its actual filename.

## 📊 Dataset

This project uses the **IMDB Movie Reviews dataset** provided by `tensorflow.keras.datasets.imdb`.

The dataset contains movie reviews labeled as positive or negative. Keras can download and cache the dataset automatically the first time the training script is run.

Dataset loader used in the project:

```python
from tensorflow.keras.datasets import imdb

max_features = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features
)
```

- Vocabulary limit: 10,000 most frequent words.
- Labels: `0` = Negative, `1` = Positive.
- Reviews are padded/truncated to 500 tokens before being passed to the model.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Movie-Review-Sentiment-Analysis-Using-RNN.git
```

### 2. Open the project directory

```bash
cd Movie-Review-Sentiment-Analysis-Using-RNN
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Run the Project

### Option A: Run the Streamlit application

Make sure `rnn_model.keras` is in the same directory as `app.py`, then run:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser, enter a movie review, and click the analysis button.

### Option B: Train the model again

If you want to train the model from scratch, run:

```bash
python train.py
```

The training script saves the trained model as:

```text
rnn_model.keras
```

After training finishes, run the Streamlit application.

> The supplied training script uses the IMDB dataset loader, so an internet connection may be needed the first time the dataset is downloaded.

## 📝 Example

**Input review:**

```text
This movie was absolutely amazing. The acting was brilliant
and the story kept me interested throughout.
```

**Possible output:**

```text
Predicted Sentiment: Positive
Positive score: Model-dependent
Negative score: Model-dependent
```

Predictions depend on the trained model and the text supplied. The example above illustrates the application format, not a guaranteed model result.

## 🔍 How It Works

1. The user enters a movie review.
2. The text is converted into tokens using the IMDB word index.
3. Tokens are converted to integer IDs.
4. The sequence is padded or truncated to the model's expected length of 500.
5. The Embedding layer converts token IDs into dense vectors.
6. The Simple RNN processes the sequence.
7. The sigmoid output is converted into a Positive or Negative label.
8. The Streamlit interface displays the prediction and scores.

## ⚠️ Important Notes

- The model expects reviews to be preprocessed using the same IMDB vocabulary and token-ID convention used during training.
- The sequence length at inference must match the training input length: **500 tokens**.
- Keep `rnn_model.keras` in the location expected by `app.py`.
- Model performance depends on training, preprocessing, and the quality of the input review.
- This project is intended for learning and demonstration; predictions should not be treated as definitive judgments.

## 🔮 Future Improvements

- Add model evaluation metrics such as precision, recall, F1-score, and a confusion matrix.
- Compare Simple RNN with LSTM and GRU architectures.
- Improve text preprocessing and handle negation and out-of-vocabulary words more robustly.
- Add review history and downloadable prediction reports.
- Deploy the Streamlit application online.

## 👨‍💻 Author

**Rohit Tayde**

Data Science | Machine Learning | Deep Learning | NLP

---

## ⭐ Support

If you find this project useful, consider giving the repository a **star** ⭐ on GitHub.

---

*Built with Python, TensorFlow/Keras, NLP, and Streamlit.*
