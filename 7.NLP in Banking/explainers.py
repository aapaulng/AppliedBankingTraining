import torch
import numpy as np
import shap
import matplotlib.pyplot as plt


class SentimentExplainer:
    def __init__(self, model, tokenizer, class_names):
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.model.eval()

    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, (np.ndarray, list)):
            texts = [str(t) for t in texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        return probs

    # ---------- SHAP ----------
    def explain_shap(self, text):
        masker = shap.maskers.Text(self.tokenizer)
        explainer = shap.Explainer(self.predict_proba, masker)
        shap_values = explainer([text])
        return shap_values

    def shap_contribution_plot(self, shap_values, pred_label, pred_prob):
        tokens = shap_values[0].data
        values = shap_values[0].values  # (tokens, classes)

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, cls in enumerate(self.class_names):
            contribs = values[:, i]
            ax.plot(range(len(tokens)), contribs, marker="o", label=cls)
            for j, val in enumerate(contribs):
                ax.text(j, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        ax.axhline(0, color="black", linestyle="--")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45)
        ax.set_ylabel("Contribution")
        ax.set_title(f"SHAP Token Contributions\nPrediction: {pred_label} ({pred_prob:.2f})")
        ax.legend()
        return fig
