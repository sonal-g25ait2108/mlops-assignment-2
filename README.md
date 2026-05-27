# mlops-assignment-2
DistilBERT Goodreads Genre Classifier
This project fine-tunes distilbert-base-cased for multi-class Goodreads book genre classification. The model was trained in a Kaggle Notebook using GPU acceleration, tracked with Weights & Biases, evaluated with accuracy, weighted F1 score, and evaluation loss, and prepared for deployment through Hugging Face Hub. The goal of this assignment is to demonstrate a complete MLOps workflow rather than maximize model accuracy.

Project Workflow
Load and sample UCSD Goodreads review data by genre.
Encode review text using DistilBertTokenizerFast.
Fine-tune DistilBertForSequenceClassification using Hugging Face Trainer.
Track training and evaluation metrics with Weights & Biases.
Save final evaluation metrics and classification report.
Upload the evaluation report as a W&B artifact.
Push the trained model and tokenizer to Hugging Face Hub.
Model
The model used is:

distilbert-base-cased
DistilBERT was chosen because it is smaller and faster than full BERT while still providing strong language understanding performance. It is suitable for Kaggle GPU training and integrates well with Hugging Face Transformers, W&B, and Hugging Face Hub.

Setup
Install the required packages:

pip install -U transformers scikit-learn wandb huggingface_hub pandas torch
For Kaggle, enable:

Internet
GPU accelerator
Kaggle Secrets for WANDB_API_KEY
Kaggle Secrets for HF_TOKEN
Running the Notebook
Run the notebook from top to bottom in Kaggle. The notebook will:

load the Goodreads review dataset
tokenize the text
train DistilBERT
log metrics to W&B
generate eval_report.json
generate eval_metrics.json
upload the evaluation report as a W&B artifact
push the model/tokenizer to Hugging Face Hub
Results
Metric	Score
Accuracy	0.6019
Weighted F1 Score	0.5992
Eval Loss	2.4617
Links
Kaggle Notebook: TODO - add public Kaggle notebook link
Hugging Face Model: 
https://huggingface.co/arcsaber2302/distilbert-goodreads-genres
W&B Run: 
https://wandb.ai/g25ait2108-prom-iit-rajasthan/mlops-assignment2/runs/up1av12k
GitHub Repository:
Files
File	Description
g25ait2108-mlops-assignment.ipynb	Final training and deployment notebook
eval_report.json	Classification report saved after evaluation
eval_metrics.json	Final evaluation metrics
README.md	Project documentation
Notes
The assignment focuses on the MLOps pipeline: secure token handling, reproducible training, experiment tracking, artifact logging, and model deployment. The achieved accuracy and F1 score are sufficient for demonstrating the workflow, but future improvements could include using more training samples, tuning hyperparameters, and comparing additional transformer models.
