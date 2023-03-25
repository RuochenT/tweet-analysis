# Sentiment analysis with multi-labels classification
This project uses BERT from hugging face transformers library to analyze whether each tweet is neutral, positive, or negative.

## Data
The data has 3 columns which are "textID", "text", and "sentiment" and 3534 rows with no missing values. The important data preprocessing before using BERT is to encode sentiment variable (neutral, negative, positive) into numeric variable (0,1,2).

## Method
The main process is composed of 

1.) split the data into train , valid, test data set. And turn them into DataDict form. 

2.) processing the data ( encoding by tokenzing the data and pad all the samples to have the same length) 

3.) load pre-trained BERT model, specify num_labels = 3, and fine-tuning the model with trainer API

4.) test the fine-tuning model on the test set. 

## Result
The result from the test set shows that both the accuracy and F1 score are 0.744.

