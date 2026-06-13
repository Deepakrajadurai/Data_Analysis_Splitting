import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.utils import clean_template_artifacts

# Page Configuration
st.set_page_config(
    page_title="German AI vs. Human Text Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism Theme)
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1rem;
        color: #888;
    }
    .status-box {
        border-radius: 10px;
        padding: 15px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .ai-box {
        background-color: rgba(244, 67, 54, 0.15);
        border: 1px solid #f44336;
        color: #ff5252;
    }
    .human-box {
        background-color: rgba(76, 175, 80, 0.15);
        border: 1px solid #4caf50;
        color: #69f0ae;
    }
    .highlight-card {
        background: rgba(30, 41, 59, 0.5);
        border-left: 5px solid #3b82f6;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 0 10px 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Example Texts
HUMAN_EXAMPLES = [
    "Sehr geehrte Kolleginnen und Kollegen, wir stehen heute vor einer wegweisenden Entscheidung für die Zukunft unserer Energieinfrastruktur. Die Bürgerinnen und Bürger erwarten von uns eine zukunftsorientierte, stabile und bezahlbare Lösung. Daher müssen wir den Ausbau der erneuerbaren Energien entschlossen vorantreiben.",
    "Meine Damen und Herren, Bildung darf keine Frage des Geldbeutels sein. Wir müssen jetzt flächendeckend in unsere Schulen investieren, die Digitalisierung anpassen und Lehrkräfte entlasten. Nur so schaffen wir echte Chancengleichheit für kommende Generationen."
]

AI_EXAMPLES = [
    "Wir stehen vor der Aufgabe, die innere Sicherheit unseres Landes an die Anforderungen der modernen Zeit anzupassen. Es bedarf einer konsequenten Stärkung unserer Sicherheitsorgane und einer klaren gesetzlichen Grundlage, um den Schutz der Bürgerinnen und Bürger nachhaltig zu gewährleisten.",
    "Für die Weiterentwicklung der regionalen Verkehrsinfrastruktur sind zusätzliche Bundesmittel bereitzustellen. Eine verlässliche Anbindung des ländlichen Raums an die Ballungszentren sichert Arbeitsplätze und stärkt den gesellschaftlichen Zusammenhalt."
]

@st.cache_resource
def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

# Main App Logic
def main():
    st.title("🛡️ German AI vs. Human Text Detector")
    st.markdown("An advanced context-aware text classifier powered by **XLM-RoBERTa** trained on leak-free, length-balanced German speech and document datasets.")

    model_dir = "best_model"
    
    # Check if final model files are ready
    if not (os.path.exists(os.path.join(model_dir, "config.json")) and os.path.exists(os.path.join(model_dir, "metrics.json"))):
        st.warning("🔄 **Final XLM-RoBERTa Model is currently training in the background on the CPU.**")
        st.info("The training script runs 10,000 samples. The app will automatically enable once training completes. Please wait a moment...")
        
        # Display live training progress from logs if available
        log_path = r"C:\Users\vijayakr\.gemini\antigravity-ide\brain\1f6870d8-f380-49bb-a6e2-c5021e445d30\.system_generated\tasks\task-256.log"
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                logs = f.readlines()
            # Show last 10 lines of progress
            st.code("".join(logs[-10:]), language="text")
            
        if st.button("Refresh Page"):
            st.rerun()
        return

    # Load Model Assets
    tokenizer, model = load_model_and_tokenizer(model_dir)
    
    # Load JSON metrics
    with open(os.path.join(model_dir, "metrics.json"), "r") as f:
        metrics = json.load(f)
    with open(os.path.join(model_dir, "confusion_matrix.json"), "r") as f:
        cm = json.load(f)
    with open(os.path.join(model_dir, "dataset_stats.json"), "r") as f:
        stats = json.load(f)
    with open(os.path.join(model_dir, "misclassifications.json"), "r") as f:
        misclass = json.load(f)

    # Sidebar
    st.sidebar.title("Model Selection")
    st.sidebar.markdown("Currently Active:")
    st.sidebar.success("XLM-RoBERTa (Primary Model)")
    st.sidebar.markdown("""
    ---
    ### Model Rationale
    **XLM-RoBERTa** is used as our primary model instead of the **TF-IDF + LR baseline** because:
    1. **Context-Aware**: Understands word meanings in sentence contexts, preventing shortcut learning.
    2. **Syntactic Generalization**: Outperforms lexical models on unseen writing styles.
    3. **Robustness**: Shows far higher resilience against adversarial paraphrasing.
    """)

    # Tabs
    tab1, tab2 = st.tabs(["🖥️ Interactive Detector", "📊 Evaluation Dashboard"])

    with tab1:
        st.subheader("Analyze Text for AI Generation")
        
        # Text Loading Helpers
        col_buttons = st.columns(4)
        with col_buttons[0]:
            if st.button("Load Human Example 1"):
                st.session_state.text_input = HUMAN_EXAMPLES[0]
        with col_buttons[1]:
            if st.button("Load Human Example 2"):
                st.session_state.text_input = HUMAN_EXAMPLES[1]
        with col_buttons[2]:
            if st.button("Load AI Example 1"):
                st.session_state.text_input = AI_EXAMPLES[0]
        with col_buttons[3]:
            if st.button("Load AI Example 2"):
                st.session_state.text_input = AI_EXAMPLES[1]
                
        # Initialize session state for text input if not present
        if "text_input" not in st.session_state:
            st.session_state.text_input = ""

        user_input = st.text_area(
            "Paste German text sentence or paragraph below:",
            value=st.session_state.text_input,
            height=150,
            key="input_area"
        )
        # Link text area value to session state
        st.session_state.text_input = user_input

        if st.button("Run Detector", type="primary"):
            if not user_input.strip():
                st.warning("Please paste some text to analyze.")
            else:
                # Preprocess: Clean template artifacts
                cleaned_text = clean_template_artifacts(user_input)
                
                # Classify
                with st.spinner("Analyzing style and structure..."):
                    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=128)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1).squeeze().numpy()
                        
                    prob_human = float(probs[0])
                    prob_ai = float(probs[1])
                    
                # Determine Prediction & Confidence
                if prob_ai > prob_human:
                    prediction = "AI Generated"
                    pred_prob = prob_ai
                    color_class = "ai-box"
                else:
                    prediction = "Human Written"
                    pred_prob = prob_human
                    color_class = "human-box"
                    
                if pred_prob > 0.90:
                    confidence = "High"
                    conf_color = "green"
                elif pred_prob > 0.70:
                    confidence = "Medium"
                    conf_color = "orange"
                else:
                    confidence = "Low"
                    conf_color = "red"
                    
                # Display Results
                st.markdown(f'<div class="status-box {color_class}">Prediction: {prediction}</div>', unsafe_allow_html=True)
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric("AI Probability", f"{prob_ai*100:.1f}%")
                    st.progress(prob_ai)
                with col_res2:
                    st.metric("Human Probability", f"{prob_human*100:.1f}%")
                    st.progress(prob_human)
                    
                st.markdown(f"**Confidence Level**: :{conf_color}[**{confidence}**] (Based on prediction probability)")
                
                with st.expander("Show Preprocessed Text fed to the model"):
                    st.write(cleaned_text)

    with tab2:
        st.subheader("Model Metrics & Dataset Quality Dashboard")
        
        # Row 1: Key Metrics Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['f1']*100:.2f}%</div>
                <div class="metric-label">F1-Score</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['accuracy']*100:.2f}%</div>
                <div class="metric-label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['precision']*100:.2f}%</div>
                <div class="metric-label">Precision</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['recall']*100:.2f}%</div>
                <div class="metric-label">Recall</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        # Row 2: Confusion Matrix & Dataset Quality
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown("### 🧩 Confusion Matrix")
            # Build HTML table for Confusion Matrix
            st.markdown(f"""
            | | Predicted Human (Class 0) | Predicted AI (Class 1) |
            | :--- | :---: | :---: |
            | **True Human (Class 0)** | **TN: {cm['tn']}** | FP: {cm['fp']} |
            | **True AI (Class 1)** | FN: {cm['fn']} | **TP: {cm['tp']}** |
            """)
            st.info("The confusion matrix is computed on a balanced validation split containing 1,000 Human and 1,000 AI leak-free sentences.")
            
        with col_d2:
            st.markdown("### 📊 Dataset Quality Statistics")
            quality_df = pd.DataFrame([
                {"Dataset Quality Parameter": "Human Samples", "Value": f"{stats['human_samples']:,}"},
                {"Dataset Quality Parameter": "AI Samples", "Value": f"{stats['ai_samples']:,}"},
                {"Dataset Quality Parameter": "Unique AI Models", "Value": str(stats['ai_models'])},
                {"Dataset Quality Parameter": "Unique Domains", "Value": str(stats['domains'])},
                {"Dataset Quality Parameter": "Duplicate Records Removed", "Value": str(stats['duplicates'])},
                {"Dataset Quality Parameter": "Missing/Empty Texts", "Value": str(stats['missing_text'])}
            ])
            st.table(quality_df)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 3: Domain / Model Distributions & Misclassifications
        col_e1, col_e2 = st.columns(2)
        
        with col_e1:
            st.markdown("### ⚙️ Distributions in AI Dataset")
            
            st.write("**AI Model Generation Breakdown:**")
            model_df = pd.DataFrame.from_dict(stats['model_distribution'], orient='index', columns=['Count'])
            st.bar_chart(model_df)
            
            st.write("**AI Domain Breakdown:**")
            domain_df = pd.DataFrame.from_dict(stats['domain_distribution'], orient='index', columns=['Count'])
            st.bar_chart(domain_df)
            
        with col_e2:
            st.markdown("### ⚠️ Model Failures: Interesting Misclassifications")
            st.markdown("These are real examples where the trained XLM-RoBERTa model failed to predict the correct label during testing, highlighting the limitations of vocabulary-independent style matching.")
            
            st.markdown("#### False Positives (Human classified as AI)")
            if misclass.get('false_positives'):
                for i, fp_text in enumerate(misclass['false_positives']):
                    st.markdown(f'<div class="highlight-card">"{fp_text}"</div>', unsafe_allow_html=True)
            else:
                st.write("*No false positives in evaluation subset.*")
                
            st.markdown("#### False Negatives (AI classified as Human)")
            if misclass.get('false_negatives'):
                for i, fn_text in enumerate(misclass['false_negatives']):
                    st.markdown(f'<div class="highlight-card">"{fn_text}"</div>', unsafe_allow_html=True)
            else:
                st.write("*No false negatives in evaluation subset.*")

if __name__ == '__main__':
    main()
