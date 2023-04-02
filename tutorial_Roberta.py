# ---------- library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import evaluate
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
from datasets import load_from_disk, load_dataset

# ----- import data set in DatasetDict format
df_train= load_dataset("csv", data_files="train.csv",split = "train")
df_test = load_dataset("csv", data_files="test.csv", split = "train")
df_valid = load_dataset("csv", data_files= "valid.csv", split = "train")
print(df_train)

# ------- encoding with Roberta Tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
train_tokenized = df_train.map(lambda batch: tokenizer(batch['text'], padding='max_length', truncation=True, max_length=32))
test_tokenized = df_test.map(lambda batch: tokenizer(batch['text'], padding='max_length', truncation=True, max_length=32))
valid_tokenized = df_valid.map(lambda batch: tokenizer(batch["text"], padding='max_length', truncation=True, max_length = 32))

train_tokenized = train_tokenized.rename_column("sentiment", "labels")
test_tokenized = test_tokenized.rename_column("sentiment", "labels")
valid_tokenized = valid_tokenized.rename_column("sentiment", "labels")

train_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
valid_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


# -------- load model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# dynmaic padding
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ------- train the model
training_args = TrainingArguments(
    output_dir='/path',          # output directory
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
trainer.evaluate() # get results from validation data set

# ---------  get prediction for test data set
pred = trainer.predict(test_tokenized) # get prediction output
prediction = pred[0].argmax(axis=1) # transform logits to compare with the original label
original = pred[1]
accuracy = evaluate.load('accuracy')
f1 = evaluate.load('f1')
accuracy.compute(predictions=prediction, references=original) #0.757
f1.compute(predictions=prediction, references=original, average='weighted') # 0.755

# ----- compare with the original result

df = pd.read_csv("test.csv")
df["tuned_sentiment"] = prediction
def convert(x):
    if x == 0:
        return 'neutral'
    elif x == 2:
        return'positive'
    else:
        return "negative"

df[["sentiment", "tuned_sentiment"]] = df[["sentiment", "tuned_sentiment"]].applymap(convert)

result = df.to_csv("result.csv") # save the data 



