# BERT-QA-SQuAD

**Fine-tuning a pre-trained BERT model on the SQuAD v1.1 dataset for Question Answering tasks using Hugging Face Transformers.**

This project demonstrates how to preprocess data, fine-tune a Transformer model, and evaluate it on custom questions using Python and Hugging Face libraries.

---

## 📌 Project Overview

**Objective:**

1. Understand the difference between Question Answering (QA) and classification tasks.  
2. Fine-tune a pre-trained BERT-based model on the SQuAD v1.1 dataset.  
3. Evaluate and test the model on sample questions.  
4. Gain hands-on experience with Hugging Face Transformers and Trainer API.

**Dataset:**  
- SQuAD v1.1 (Stanford Question Answering Dataset)  
- Contains context passages, questions, and corresponding answers.

---

## 🛠️ Installation

Install the required libraries:

```bash
pip install transformers datasets evaluate
```

---

## 📑 Usage

1. **Load Dataset:**

```python
from datasets import load_dataset

dataset = load_dataset("squad")
print(dataset['train'][0])
```

2. **Tokenization:**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Preprocess question-context pairs
```

3. **Model Setup:**

```python
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
```

4. **Fine-tuning:**

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    "bert-qa",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    num_train_epochs=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)
trainer.train()
```

5. **Evaluation & Testing:**

```python
question = "Who developed the theory of relativity?"
context = "Albert Einstein developed the theory of relativity in the early 20th century."

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits
```

---

## 📊 Results

- Metrics used: **Exact Match (EM)**, **F1 Score**  
- Tested the model on custom questions to verify predictions.

---

## 📑 Introduction & Reflection

**Introduction:**  
Question Answering (QA) requires the model to find the exact answer to a question within a context passage, unlike classification which assigns predefined labels. QA needs deeper understanding and reasoning to align questions with context spans.

**Reflection:**  
Through this project, I learned how QA differs from classification tasks. Fine-tuning BERT on SQuAD taught me about tokenization, context-question alignment, and span prediction. I also gained practical experience with the Hugging Face Trainer API and evaluating model predictions. Testing custom questions helped me understand model strengths and limitations. Overall, this improved my applied NLP skills using Transformers.

---

## 💾 Folder Structure (Recommended)

```
bert-qa-squad/
│
├─ README.md
├─ requirements.txt
├─ QA_FineTuning_SQuAD_YourName.ipynb
├─ dataset/  (optional local dataset storage)
└─ outputs/  (model checkpoints & evaluation results)
```

---

## 🔗 References

1. [Hugging Face Transformers](https://huggingface.co/transformers/)  
2. [SQuAD v1.1 Dataset](https://rajpurkar.github.io/SQuAD-explorer/)  
3. [Hugging Face Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer)

---

## 📜 License

This project is open-source and available for educational purposes.

