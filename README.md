# 🚦 Harmful Prompt Classification with Sentence Transformers and SGDClassifier

This project builds a machine learning pipeline to classify prompts as either **allowed** or **blocked** based on their content. It uses sentence embeddings from a multilingual transformer model and trains a logistic regression classifier using stochastic gradient descent (SGD). The goal is to evaluate and fine-tune a model that can help identify harmful prompts in AI-generated content.

---

## 📁 Dataset

The dataset is loaded from Hugging Face's `onepaneai/harmful-prompts-gpt-after-guardrail-evaluation` repository. It contains prompts labeled as either:

- `allowed` (safe)
- `blocked` (harmful)

These labels are mapped to binary values:
- `allowed` → `0`
- `blocked` → `1`

---

## 🧪 Pipeline Overview

### 1. **Data Preparation**
- Filters out invalid labels.
- Maps labels to binary values.
- Uses only the `prompt` field for classification.

### 2. **Data Splitting**
- Splits the dataset into training, validation, and test sets using stratified sampling to preserve label distribution.

### 3. **Text Embedding**
- Uses the `paraphrase-multilingual-MiniLM-L12-v2` model from `sentence-transformers` to convert prompts into dense vector representations.

### 4. **Model Training**
- Defines a pipeline with `SGDClassifier` using logistic loss.
- Performs hyperparameter tuning using `GridSearchCV` with cross-validation.

### 5. **Evaluation**
- Evaluates the best model on both validation and test sets.
- Reports metrics: Accuracy, Precision, Recall, F1 Score.
- Displays classification reports.

### 6. **Visualization**
- Generates and saves:
  - Confusion matrix
  - Classification report heatmap

### 7. **Example Predictions**
- Shows a few sample predictions with true and predicted labels.

### 8. **Model Saving**
- Saves the best classifier and the sentence transformer model to the `outputs/` directory.

---

## 📊 Metrics Used

- **Accuracy**: Overall correctness.
- **Precision**: Correctness of positive predictions.
- **Recall**: Coverage of actual positives.
- **F1 Score**: Harmonic mean of precision and recall.

---

## 📦 Dependencies

Make sure to install the following Python packages:

```bash
pip install pandas numpy scikit-learn sentence-transformers matplotlib seaborn joblib

## 🗂 Output Files
- `outputs/confusion_matrix.png`: Confusion matrix visualization.
- `outputs/classification_report.png`: Heatmap of classification metrics.
- `outputs/fine_tuned_classifier_best.joblib`: Trained classifier.
- `outputs/sentence_transformer_model`: Saved transformer model.

## 🚀 How to Run
- Ensure the dataset is accessible via Hugging Face's `hf://` protocol.
- Run the script in a Python environment with the required packages.
- Check the `outputs/` folder for visualizations and saved models.


## 📌 Notes
- The script uses early stopping and validation fraction to prevent overfitting.
- The transformer model supports multilingual prompts.
- The classifier is tuned for performance using grid search.




# 🚀 **Chatbot with Classification and Personalized Responses** 🚀

This repository contains a script that implements an interactive **Chatbot** system using machine learning and language generation techniques. The script loads a fine-tuned classification model to determine if a prompt is allowed, loads a sentence transformer for encoding prompts, and uses a causal language model to generate responses based on a selected personality type.

---

## 🌟 Overview 🌟
- **Purpose**: Classify prompts as **allowed** or **blocked**, then generate responses.
- **Features**: Interactive session, personality choices, conversation history, ratings.

---

## 📦 Dependencies 📦
To run the script, ensure you have the following Python packages:

- **joblib**
- **sentence_transformers**
- **transformers**
- **torch**
- **time** (standard library)

---

## 🛠️ Instructions 🛠️
1. **Load Models**: The classifier and transformer models are loaded from the `outputs/` directory.
2. **Select Personality**: Choose from friendly, formal, sarcastic, or enthusiastic.
3. **Interactive Chat**: Input prompts, get classifications and responses, and rate them.
4. **Exit**: Type `exit` or `quit` to end session.

---

## 🗂 Output Files 🗂
- `outputs/fine_tuned_classifier_best.joblib`: Trained classifier model.
- `outputs/sentence_transformer_model`: Sentence transformer model directory.

---

## 🚀 How to Run 🚀
- Ensure dependencies are installed.
- Have required model files in the `outputs/` folder.
- Run the script in a suitable Python environment.
- Follow on-screen prompts.

---

## 📌 Notes 📌
- The script simulates a chat with personality-driven responses.
- Ratings are stored in memory.
- Use GPU for better LLM performance.
- Assumes pre-trained classifier and encoder models are available.




# Chatbot System Project Api server 

## 🧠 Overview
This project is a prompt moderation and chatbot system built using machine learning and modern web frameworks. It includes:

1. **Training Pipeline**: Trains a classifier to detect harmful prompts.
2. **CLI Chatbot**: A command-line interface for chatting.
3. **FastAPI Backend**: Provides API endpoints for classification and chat.
4. **Streamlit Frontend**: A web interface for interacting with the chatbot.

---

## 🏗️ Architecture

![Architecture Diagram](blob:https://outlook.office.com/2f76ef44-f847-4cae-8c94-a8ed8d6daf0f)

### Components:
- **SentenceTransformer Encoder**: Converts prompts into embeddings.
- **SGDClassifier**: Classifies prompts as "allowed" or "blocked".
- **LLM (SmolLM3-3B)**: Generates responses for allowed prompts.
- **FastAPI**: Hosts endpoints for classification and chat.
- **Streamlit**: Provides a user-friendly chat interface.
- **Training Pipeline**: Prepares data, trains, and saves models.
- **Storage**: Stores trained models and user feedback.

---

## ⚙️ Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download or place the dataset in the correct path.
4. Run the training pipeline:
   ```bash
   uv run main2.py
   ```
5. Start the FastAPI server:
   ```bash
   python -m ensurepip --upgrade
   python -m pip install fastapi uvicorn
    python -m uvicorn api_server:app --reload
   ```
6. Start the Streamlit app:
   ```bash
   sourve .venv/bin/activate
   python -m pip install streamlit
   streamlit run streamlit_app.py
---

## 💬 Usage

- **Train**: Run the training pipeline to generate the classifier and encoder.
- **CLI Chat**: Interact with the chatbot via terminal.
- **API**: Use `/classify`, `/respond`, or `/chat` endpoints.
- **Web UI**: Use the Streamlit interface to chat with the bot.

---

## 📁 Outputs

- `outputs/fine_tuned_classifier_best.joblib`: Trained classifier.
- `outputs/sentence_transformer_model/`: Saved encoder.
- `outputs/confusion_matrix.png`: Evaluation plot.
- `outputs/classification_report.png`: Metrics heatmap.

---

## 🧪 Technologies

- Python, FastAPI, Streamlit
- scikit-learn, SentenceTransformers, Transformers
- Hugging Face LLM (SmolLM3-3B)

---

## 📌 Notes

- Prompts are filtered before generating responses.
- Only "allowed" prompts are passed to the LLM.
- You can extend this system with logging, analytics, or feedback loops.
