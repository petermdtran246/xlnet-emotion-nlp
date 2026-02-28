# Emotion Transformer Classification

> Fine-tuning XLNet for multi-class emotion classification using HuggingFace Transformers and PyTorch.

---

## Overview

This project implements an end-to-end NLP pipeline to classify text into four emotional categories:

- Anger  
- Fear  
- Joy  
- Sadness  

The model is fine-tuned from a pretrained **XLNet Transformer** using the HuggingFace `Trainer` API.

This repository demonstrates:

- Modern Transformer fine-tuning workflow  
- Dataset preprocessing & balancing  
- Tokenization and encoding pipeline  
- Model evaluation  
- Inference deployment using `pipeline()`  

---

## Model Choice

Model used:

xlnet-base-cased

Why XLNet?

- Permutation-based language modeling  
- Strong contextual representation  
- Robust classification performance  
- Handles longer dependencies better than traditional RNN/LSTM  

The architecture is adapted for sequence classification with 4 output labels.

---

## Pipeline Architecture

Raw CSV Data  
      ↓  
Text Cleaning  
      ↓  
Label Encoding  
      ↓  
Train/Test Split  
      ↓  
Tokenization (XLNetTokenizer)  
      ↓  
HuggingFace Dataset  
      ↓  
Fine-Tuning (Trainer API)  
      ↓  
Evaluation  
      ↓  
Inference Pipeline  

---

## Dataset

The dataset consists of labeled emotion text samples combined from:

- emotion-labels-train.csv  
- emotion-labels-test.csv  
- emotion-labels-val.csv  

### Text Preprocessing

- Emoji removal  
- Username removal (@mentions)  
- Text normalization  

Example:

```python
data['text_clean'] = data['text'].apply(lambda x: clean(x, no_emoji=True))
data['text_clean'] = data['text_clean'].apply(lambda x: re.sub(r'@[^\s]+', '', x))
```

---

## Class Balancing

To prevent class imbalance bias:

```python
g = data.groupby('label')
data = g.apply(lambda x: x.sample(g.size().min())).reset_index()
```

All classes are downsampled to the smallest class size.

---

## Label Encoding

| ID | Label   |
|----|----------|
| 0  | anger    |
| 1  | fear     |
| 2  | joy      |
| 3  | sadness  |

Encoded using `LabelEncoder`.

---

## Tokenization

Tokenizer:

```python
XLNetTokenizer.from_pretrained("xlnet-base-cased")
```

Configuration:

- max_length = 128  
- padding = "max_length"  
- truncation = True  

Outputs:

- input_ids  
- token_type_ids  
- attention_mask  

---

## Training

Training configuration:

```python
TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    num_train_epochs=3
)
```

Training handled via:

```python
Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics
)
```

Metric used: Accuracy

---

## Results

| Metric | Value |
|--------|--------|
| Validation Accuracy | 0.34 |
| Training Loss | 1.316 |
| Epochs | 3 |

Note: Training was performed on a reduced dataset (100 samples) for demonstration purposes.  
Full dataset training improves performance significantly.

---

## Inference Example

```python
clf = pipeline("text-classification", fine_tuned_model, tokenizer=tokenizer)

test_text = "I finally solved the problem!"
clf(test_text, top_k=None)
```

Example Output:

```json
[
  {"label": "joy", "score": 0.42},
  {"label": "fear", "score": 0.22},
  {"label": "anger", "score": 0.19},
  {"label": "sadness", "score": 0.16}
]
```

---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- HuggingFace Transformers  
- HuggingFace Datasets  
- PyTorch  
- Scikit-learn  
- Evaluate  

---

## Future Improvements

- Train on full dataset  
- Add F1-score, precision, recall  
- Hyperparameter tuning  
- Early stopping  
- Add experiment tracking (Weights & Biases)  
- Export to ONNX for optimized inference  
- Compare against BERT / RoBERTa  
