# Sentiment analysis with multi-labels classification

BERT and Roberta are both pre-trained transformer models that use bidirectional encoder representations from transformers. BERT, developed by Google in 2018, uses a masked language modeling objective and next sentence prediction task, while Roberta, developed by Facebook AI Research in 2019, uses a larger training corpus, a longer training time, and a dynamic masking approach. Roberta uses a byte-level byte-pair encoding for subword tokenization, while BERT uses WordPiece tokenization. Roberta has been shown to outperform BERT on natural language understanding tasks such as GLUE and SuperGLUE, due to its larger size and more extensive pre-training. Both BERT and Roberta are available in pre-trained form and can be fine-tuned on specific natural language processing tasks.

This project uses BERT and RoBERTa from hugging face transformers library to analyze whether each tweet is neutral, positive, or negative.

## Data
The data has 3 columns which are "textID", "text", and "sentiment" and 3534 rows with no missing values. The important data preprocessing before using BERT is to encode sentiment variable (neutral, negative, positive) into numeric variable (0,1,2).

## Method
The main process for using BERT, RoBERTa for multi-label classification involves the following steps:

Data preprocessing: The input data must be preprocessed to fit the format expected by the BERT model. This involves tokenizing the text into subword units, adding special tokens to mark the beginning and end of the text, and padding or truncating the input sequences to a fixed length.

Fine-tuning the pre-trained BERT model: The pre-trained BERT model is fine-tuned on the specific multi-label classification task using a labeled training dataset (num_labels = 3). This involves training the model on the input text sequences and their corresponding multi-label outputs, optimizing the model parameters using a suitable loss function, and evaluating the model performance on a validation dataset.

Model evaluation: The performance of the fine-tuned BERT model is evaluated on a test dataset using suitable metrics which this project uses accuracy and F1 score. 

Inference with test data set: Once the model has been trained and evaluated, it can be used for inference on new, unseen text data to predict the corresponding multi-label outputs.


## Result with test data set 
| Metrics | BERT | RoBERTa |
| --- | --- | --- |
| Accuracy |0.744 | 0.757 |
| F1 | 0.744| 0.755 |

## References 
Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.
