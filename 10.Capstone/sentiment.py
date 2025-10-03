# Import libraries
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr # import gradio

import warnings
warnings.filterwarnings('ignore')

#save_path = "./models/finbert-financial-sentiment"
save_path = 'yiyanghkust/finbert-tone'
#LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative

# Load Model
tokenizer = AutoTokenizer.from_pretrained(save_path)
model = AutoModelForSequenceClassification.from_pretrained(save_path)

# Example inference with loaded model
text = "The bank announced late but fine  financial results this quarter."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
pred = torch.argmax(outputs.logits).item()

#labels = ["negative", "neutral", "positive"]
labels = ['neutral','positive','negative']
print("Predicted Sentiment:", labels[pred])

def response(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits).item()

    return f"Predicted Sentiment: {labels[pred]}" 

demo = gr.Interface(response,'text','text')
demo.launch()