import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import gradio as gr
from explainers import SentimentExplainer
import shap

# ---------- Load Models ----------
sent_model_path = "./models/distilbert-financial-sentiment"
sent_tok = AutoTokenizer.from_pretrained(sent_model_path)
sent_model = AutoModelForSequenceClassification.from_pretrained(sent_model_path)
labels = ["negative", "neutral", "positive"]

explainer = SentimentExplainer(sent_model, sent_tok, class_names=labels)

sum_name = "t5-small"
sum_tok = AutoTokenizer.from_pretrained(sum_name)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_name)


# ---------- Functions ----------
def predict_sentiment(text):
    if not text.strip():
        return {}
    inputs = sent_tok(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = sent_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


def summarize(text, max_len=80):
    if not text.strip():
        return ""
    inputs = sum_tok("summarize: " + text, return_tensors="pt", truncation=True, max_length=512)
    ids = sum_model.generate(**inputs, max_length=max_len, num_beams=4, early_stopping=True)
    return sum_tok.decode(ids[0], skip_special_tokens=True)


def explain_text(text):
    if not text.strip():
        return "<p style='color:red;'>⚠️ No input text provided.</p>", None

    probs = explainer.predict_proba(text)
    pred_class = np.argmax(probs, axis=1)[0]
    pred_label = labels[pred_class]
    pred_prob = probs[0][pred_class]

    shap_values = explainer.explain_shap(text)
    shap_html = shap.plots.text(shap_values[0], display=False)
    fig = explainer.shap_contribution_plot(shap_values, pred_label, pred_prob)

    return f"<h3>Prediction → {pred_label} ({pred_prob:.2f})</h3>" + shap_html, fig


# ---------- Gradio UI ----------
with gr.Blocks(title="Banking NLP Dashboard") as demo:
    gr.Markdown("# Banking NLP Dashboard\nAnalyze financial text with **Sentiment, Summarization & Explanations**")

    with gr.Tab("Sentiment Analysis"):
        text_in = gr.Textbox(label="Enter financial sentence", lines=4)
        text_out = gr.Label(label="Predicted Sentiment")
        analyze_btn = gr.Button("Analyze", variant="primary")
        analyze_btn.click(predict_sentiment, inputs=text_in, outputs=text_out)

    with gr.Tab("Summarization"):
        sum_in = gr.Textbox(label="Enter text to summarize", lines=8)
        max_len = gr.Slider(30, 200, value=80, step=5, label="Max length")
        sum_out = gr.Textbox(label="Summary", lines=6)
        sum_btn = gr.Button("Summarize", variant="primary")
        sum_btn.click(summarize, inputs=[sum_in, max_len], outputs=sum_out)

    with gr.Tab("Sentiment Explanation"):
        exp_in = gr.Textbox(label="Enter sentence", lines=4)
        exp_html = gr.HTML(label="Explanation")
        exp_fig = gr.Plot(label="Contribution Plot")
        exp_btn = gr.Button("Explain", variant="primary")
        exp_btn.click(explain_text, inputs=exp_in, outputs=[exp_html, exp_fig])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
