import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, make_scorer,
    confusion_matrix, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# 1. Load dataset
df = pd.read_parquet("hf://datasets/onepaneai/harmful-prompts-gpt-after-guardrail-evaluation/data/train-00000-of-00001.parquet")

# 2. Clean and prepare data
df = df[df['answer'].isin(['allowed', 'blocked'])]  # Only keep valid labels
df['answer'] = df['answer'].map({'allowed': 0, 'blocked': 1})  # Map to 0/1
df['text'] = df['prompt']  # Only use the prompt

# 3. Split data
train_val_texts, test_texts, y_train_val, y_test = train_test_split(
    df["text"], df["answer"], test_size=0.2, random_state=42, stratify=df["answer"]
)
train_texts, val_texts, y_train, y_val = train_test_split(
    train_val_texts, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

# 4. Encode prompts
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
X_train_embeddings = model.encode(train_texts.tolist(), convert_to_numpy=True, show_progress_bar=True)
X_val_embeddings = model.encode(val_texts.tolist(), convert_to_numpy=True, show_progress_bar=True)
X_test_embeddings = model.encode(test_texts.tolist(), convert_to_numpy=True, show_progress_bar=True)

# 5. Define pipeline and hyperparameter grid
pipeline = Pipeline([
    ("clf", SGDClassifier(loss="log_loss", random_state=42, early_stopping=True, n_iter_no_change=5, validation_fraction=0.1))
])

param_grid = {
    "clf__alpha": [1e-4, 1e-3, 1e-2],
    "clf__penalty": ["l2", "l1", "elasticnet"],
    "clf__max_iter": [1000, 2000, 3000],
    "clf__tol": [1e-3, 1e-4]
}

# 6. Grid search with cross-validation
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring=make_scorer(accuracy_score),
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train_embeddings, y_train)
best_model = grid_search.best_estimator_

# 7. Evaluate on validation set
print("\n🧪 Evaluating on validation set...")
y_val_pred = best_model.predict(X_val_embeddings)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred, average='weighted')
val_recall = recall_score(y_val, y_val_pred, average='weighted')
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

print(f"✅ Validation Accuracy: {val_accuracy:.4f}")
print(f"🎯 Precision: {val_precision:.4f}")
print(f"🔁 Recall: {val_recall:.4f}")
print(f"📊 F1 Score: {val_f1:.4f}")
print("\n📋 Validation Classification Report:")
print(classification_report(y_val, y_val_pred))

# 8. Evaluate on test set
print("\n🧪 Evaluating on test set...")
y_test_pred = best_model.predict(X_test_embeddings)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"✅ Test Accuracy: {test_accuracy:.4f}")
print(f"🎯 Precision: {test_precision:.4f}")
print(f"🔁 Recall: {test_recall:.4f}")
print(f"📊 F1 Score: {test_f1:.4f}")
print("\n📋 Test Classification Report:")
print(classification_report(y_test, y_test_pred))

# 9. Visualizations
os.makedirs("outputs", exist_ok=True)

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('outputs/confusion_matrix.png')
plt.close()

# Classification report heatmap
report = classification_report(y_test, y_test_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().iloc[:-1, :-1]
plt.figure(figsize=(8, 6))
sns.heatmap(report_df, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Classification Report')
plt.xlabel('Metrics')
plt.ylabel('Classes')
plt.savefig('outputs/classification_report.png')
plt.close()

# 10. Show example predictions
print("\n🔍 Example Predictions:")
for i in range(min(10, len(test_texts))):
    print(f"Prompt: {test_texts.iloc[i][:60]}…")
    print(f"True label: {'blocked' if y_test.iloc[i] == 1 else 'allowed'}")
    print(f"Predicted: {'blocked' if y_test_pred[i] == 1 else 'allowed'}")
    print("---")

# 11. Save model and encoder
joblib.dump(best_model, "outputs/fine_tuned_classifier_best.joblib")
model.save("outputs/sentence_transformer_model")
print("✅ Best model and encoder saved to outputs/")
