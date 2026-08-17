import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

# Load CSV dataset
df = pd.read_csv('/kaggle/input/final-dataset/final_dataset.csv').dropna()

# Load RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Define the RoBERTa model
model = TFRobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Tokenization function
def tokenize_text(text):
    return tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="np")

# Tokenize text inputs
input_ids, attention_masks = [], []
for text in df['headline']:
    encoded_text = tokenize_text(text)
    input_ids.append(encoded_text['input_ids'].squeeze())
    attention_masks.append(encoded_text['attention_mask'].squeeze())

input_ids = np.array(input_ids)
attention_masks = np.array(attention_masks)
labels = np.array(df['label'])

# Handle class imbalance using SMOTE
smote = SMOTE()
input_ids_res, labels_res = smote.fit_resample(input_ids, labels)
attention_masks_res, _ = smote.fit_resample(attention_masks, labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(input_ids_res, labels_res, test_size=0.2, random_state=42)
train_attention_masks, test_attention_masks = train_test_split(attention_masks_res, test_size=0.2, random_state=42)

# Define datasets
train_dataset = tf.data.Dataset.from_tensor_slices(((X_train, train_attention_masks), y_train)).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices(((X_test, test_attention_masks), y_test)).batch(16)

# Define optimizer with learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=2e-5, end_learning_rate=1e-6, decay_steps=10000
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Training with early stopping
early_stopping_patience = 2
best_loss = float('inf')
patience_counter = 0

epochs = 15  # Increased epochs for better training
max_steps = 100  # Steps per epoch

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    epoch_loss = 0
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        if step >= max_steps:
            break

        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True).logits
            loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        epoch_loss += loss_value.numpy()
        if step % 10 == 0:
            print(f"Step {step}: loss = {loss_value:.4f}")

    epoch_loss /= max_steps
    print(f"Epoch {epoch + 1} Average Loss: {epoch_loss:.4f}")

    # Early stopping check
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
        # Save model weights using pickle
        with open("/kaggle/working/roberta_model.pkl", "wb") as f:
            pickle.dump(model.get_weights(), f)
        # Save tokenizer
        with open("/kaggle/working/tokenizer_2.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# Evaluate the model
predictions = model.predict((X_test, test_attention_masks)).logits
pred_labels = np.argmax(predictions, axis=1)

# Display evaluation metrics
print("\nEvaluation Metrics:")
print("Accuracy:", accuracy_score(y_test, pred_labels))
print("Precision:", precision_score(y_test, pred_labels))
print("Recall:", recall_score(y_test, pred_labels))
print("F1 Score:", f1_score(y_test, pred_labels))
print("ROC-AUC Score:", roc_auc_score(y_test, predictions[:, 1]))
print("Classification Report:\n", classification_report(y_test, pred_labels))

# Confusion matrix
cm = confusion_matrix(y_test, pred_labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Offensive', 'Non-Offensive'], yticklabels=['Offensive', 'Non-Offensive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix')
plt.show()

# Ethical safeguards
print("\nEnsure ethical use: Model predictions are strictly for educational purposes.")
