
# ---------- library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import evaluate
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, load_metric


#--------- split data set and data preprocessing
df = pd.read_csv("/Users/ruochentan1/Downloads/test.csv")
def convert(row):
    if row == "neutral":
        result = '0'
    elif row == "positive":
        result = '2'
    else:
        result = '1'
    return result

df["sentiment"] = df["sentiment"].apply(lambda x: convert(x))
train, test = train_test_split(df, test_size=0.2, random_state=123)
train,valid = train_test_split(train,test_size = 0.25, random_state = 123 )


train.to_csv("/Users/ruochentan1/PycharmProjects/Sentiment/train.csv", index=False)
test.to_csv("/Users/ruochentan1/PycharmProjects/Sentiment/test.csv",index =False)
valid.to_csv("/Users/ruochentan1/PycharmProjects/Sentiment/valid.csv",index =False)


# ----- import data set in DatasetDict format
df_train= load_dataset("csv", data_files="/Users/ruochentan1/PycharmProjects/Sentiment/train.csv",split = "train")
df_test = load_dataset("csv", data_files="/Users/ruochentan1/PycharmProjects/Sentiment/test.csv", split = "train")
df_valid = load_dataset("csv", data_files= "/Users/ruochentan1/PycharmProjects/Sentiment/valid.csv", split = "train")
print(df_train)

# ------- encoding
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
train_tokenized = df_train.map(lambda batch: tokenizer(batch['text'], padding='max_length', truncation=True, max_length=32))
test_tokenized = df_test.map(lambda batch: tokenizer(batch['text'], padding='max_length', truncation=True, max_length=32))
valid_tokenized = df_valid.map(lambda batch: tokenizer(batch["text"], padding='max_length', truncation=True, max_length = 32))

train_tokenized.set_format("torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
test_tokenized.set_format("torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])


model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)


training_args = TrainingArguments(
    output_dir='/Users/ruochentan1/PycharmProjects/Sentiment',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=0,                # number of warmup steps for learning rate scheduler
    learning_rate=5e-5,               # learning rate
    logging_dir='./logs',            # directory for storing logs
    logging_steps=1000,
)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=train_tokenized,
   eval_dataset=valid_tokenized,
   tokenizer=tokenizer,
   compute_metrics=compute_metrics,
)

trainer.train()

#----------- save the fine-tuned model
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
output_dir = "/Users/ruochentan1/PycharmProjects/sentiment"
model_to_save = model.module if hasattr(model, 'module') else model

import os
import torch
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(output_dir)

# --------- reload the model
output_dir = "/Users/ruochentan1/PycharmProjects/sentiment"
model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)


# ---------  get prediction for test data set
pred = trainer.predict(test_tokenized) # get prediction output
prediction = pred[0].argmax(axis=1) # transform logits to compare with the original label
original = pred[1]
accuracy = evaluate.load('accuracy')
f1 = evaluate.load('f1')
accuracy.compute(predictions=prediction, references=original) # 0.744
f1.compute(predictions=prediction, references=original, average='weighted') # 0.744
