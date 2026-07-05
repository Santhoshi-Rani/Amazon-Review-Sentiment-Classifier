import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from scipy.sparse import hstack, csr_matrix
import plotly.express as px
import time
from datetime import datetime

# --- MUST BE FIRST ---
st.set_page_config(
    page_title="Customer Review Sentiment Classifier", 
    layout="wide",
    page_icon="📝",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Inter',sans-serif;
}

.stApp{
    background:#F8FAFC;
}

section[data-testid="stSidebar"]{
    background:#FFFFFF;
    border-right:1px solid #E5E7EB;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1200px;
}

h1{
    color:#111827;
    font-weight:700;
}

h2,h3{
    color:#1F2937;
}

.stTextArea textarea{

    border-radius:14px;

    border:1px solid #D1D5DB;

    background:#FFFFFF;

    color:#111827;

    font-size:16px;

}

.stTextArea textarea:focus{

    border:2px solid #2563EB;

}

.stButton>button{

    width:100%;

    background:#2563EB;

    color:white;

    border:none;

    border-radius:12px;

    height:52px;

    font-size:17px;

    font-weight:600;

}

.stButton>button:hover{

    background:#1D4ED8;

}

[data-testid="metric-container"]{

    background:white;

    border-radius:16px;

    border:1px solid #E5E7EB;

    padding:20px;

    box-shadow:0 8px 25px rgba(0,0,0,.05);

}

div[data-testid="stExpander"]{

    background:white;

    border-radius:12px;

    border:1px solid #E5E7EB;

}

.stTabs [data-baseweb="tab-list"]{

    gap:20px;

}

.stTabs [data-baseweb="tab"]{

    font-size:16px;

    font-weight:600;

}
html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"]{
    background:#F8FAFC !important;
    color:#111827 !important;
}

/* Main page text */
p,
span,
div,
label{
    color:#111827 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] *{
    color:#111827 !important;
}

/* Markdown */
[data-testid="stMarkdownContainer"]{
    color:#111827 !important;
}

/* Tabs */
button[data-baseweb="tab"]{
    color:#111827 !important;
}

/* Checkbox text */
.stCheckbox label{
    color:#111827 !important;
}

/* Success / Warning / Info */
[data-testid="stAlert"]{
    color:#111827 !important;
}

/* TextArea */
textarea{
    color:#111827 !important;
    background:white !important;
}

/* Placeholder */
textarea::placeholder{
    color:#6B7280 !important;
}

/* Text Input */
input{
    color:#111827 !important;
    background:white !important;
}

/* Expander */
details{
    color:#111827 !important;
}

/* Metrics */
[data-testid="metric-container"] label,
[data-testid="metric-container"] div{
    color:#111827 !important;
}

</style>
""",unsafe_allow_html=True)
# --- File paths ---
MODEL_PATH = 'final_logistic_model.pkl'
VECTORIZER_PATH = 'fitted_tfidf_vectorizer.pkl'

# --- Download NLTK data ---
@st.cache_resource
def download_nltk_data():
    with st.spinner("Downloading NLTK data..."):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            pass

download_nltk_data()

# --- Load model and vectorizer ---
@st.cache_resource
def load_assets():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# --- Enhanced preprocessing function ---
@st.cache_data
def preprocess_text(text):
    """Cleans, tokenizes, POS tags, and lemmatizes a raw text string."""
    # Clean text
    text = text.lower()
    text = re.sub(r'http\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization and stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    if not filtered_tokens:
        return ""
    
    # POS Tagging and Lemmatization
    pos_tags = pos_tag(filtered_tokens)
    lemmatizer = WordNetLemmatizer()
    
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return 'a'
        elif treebank_tag.startswith('V'):
            return 'v'
        elif treebank_tag.startswith('N'):
            return 'n'
        elif treebank_tag.startswith('R'):
            return 'r'
        else:
            return 'n'
    
    lemmas = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) 
              for word, tag in pos_tags]
    
    return ' '.join(lemmas)

# --- Sentiment analysis function ---
def analyze_sentiment(text, model, vectorizer):
    processed_text = preprocess_text(text)
    if not processed_text:
        return None, None
    
    text_vector = vectorizer.transform([processed_text])
    structured_features = csr_matrix([[0.0, 2010]])
    final_features = hstack([structured_features, text_vector])
    
    prediction = model.predict(final_features)[0]
    probabilities = model.predict_proba(final_features)[0]
    
    return prediction, probabilities

# --- Sidebar with information ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/google-reviews.png", width=80)
    st.title("ℹ️ About")
    st.markdown("""
    This app uses **Machine Learning** to analyze customer reviews and predict sentiment.
    
    ### 🎯 Model Details
    - **Algorithm:** Logistic Regression
    - **Accuracy:** 78%
    - **Training Data:** Amazon Reviews
    
    ### 📊 Sentiment Classes
    - 😞 **Unsatisfied** (0) - Negative feedback
    - 😐 **Neutral** (1) - Mixed or neutral opinions  
    - 😊 **Satisfied** (2) - Positive feedback
    
    ### 🔧 Features
    - Real-time sentiment prediction
    - Confidence scores for each class
    - Text preprocessing visualization
    - Batch analysis for multiple reviews
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# --- Main App ---
st.title("📝 Customer Review Sentiment Analysis")
st.markdown("**Understand what your customers are really saying**")

# Load model
try:
    model, vectorizer = load_assets()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.info("Please ensure model files are in the correct directory")
    st.stop()

# --- Tabs for different features ---
tab1, tab2, tab3 = st.tabs(["🔍 Single Review", "📊 Batch Analysis", "📈 About Model"])

# --- TAB 1: Single Review Analysis ---
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Customer Review")
        user_input = st.text_area("Customer Review",
                                  height=200,
                                  label_visibility="collapsed",
                                  placeholder="Example: 'This product exceeded my expectations! The quality is outstanding and delivery was fast.'",
                                  key="review_input")
        
        # Example reviews
        with st.expander("📋 Try these example reviews"):
            col_ex1, col_ex2, col_ex3 = st.columns(3)
            with col_ex1:
                if st.button("😞 Negative Example"):
                    user_input = "This product is terrible! Complete waste of money. It broke after one use."
            with col_ex2:
                if st.button("😐 Neutral Example"):
                    user_input = "The product is okay. Does what it says but nothing special. Average quality."
            with col_ex3:
                if st.button("😊 Positive Example"):
                    user_input = "Absolutely love this! Best purchase I've made. Highly recommended!"
    
    with col2:
        st.subheader("Analysis Options")
        show_processed = st.checkbox("Show processed text", value=False)
        show_word_count = st.checkbox("Show word statistics", value=True)
    
    # Predict button
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if user_input and user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                time.sleep(0.5)  # Smooth animation
                prediction, probabilities = analyze_sentiment(user_input, model, vectorizer)
                
                if prediction is None:
                    st.warning("⚠️ Unable to process text. Please enter a valid review.")
                else:
                    # Display main result
                    sentiment_map = {0: ("Unsatisfied", "😞", "#f44336"), 
                                     1: ("Neutral", "😐", "#ff9800"), 
                                     2: ("Satisfied", "😊", "#4caf50")}
                    
                    sentiment_text, emoji, color = sentiment_map[prediction]
                    
                    # Big result card
                    st.markdown(f"""
                    <div style="background-color: {color}10; padding: 30px; border-radius: 15px; text-align: center; border: 2px solid {color};">
                        <h1 style="font-size: 48px; margin: 0;">{emoji}</h1>
                        <h2 style="color: {color}; margin: 10px 0;">{sentiment_text}</h2>
                        <p style="font-size: 18px;">Confidence: {probabilities[prediction]*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence scores with gauge charts
                    st.subheader("📊 Confidence Breakdown")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Unsatisfied", f"{probabilities[0]*100:.1f}%", 
                                 delta=None, delta_color="inverse")
                        st.progress(probabilities[0])
                    
                    with col2:
                        st.metric("Neutral", f"{probabilities[1]*100:.1f}%")
                        st.progress(probabilities[1])
                    
                    with col3:
                        st.metric("Satisfied", f"{probabilities[2]*100:.1f}%")
                        st.progress(probabilities[2])
                    
                    # Additional analysis
                    if show_word_count:
                        words = user_input.split()
                        st.info(f"📝 Word count: {len(words)} characters | {len(words)} words")
                    
                    if show_processed:
                        processed = preprocess_text(user_input)
                        with st.expander("🔧 View Processed Text"):
                            st.code(processed, language="text")
        else:
            st.warning("⚠️ Please enter a review to analyze.")

# --- TAB 2: Batch Analysis ---
with tab2:
    st.subheader("Batch Review Analysis")
    st.markdown("Analyze multiple reviews at once (one per line)")
    
    batch_input = st.text_area("Enter multiple reviews", height=300,
                               placeholder="Review 1: This product is amazing!\nReview 2: Not worth the money.\nReview 3: It's okay, nothing special.",
                               key="batch_input")
    
    if st.button("📊 Analyze Batch", type="primary"):
        if batch_input.strip():
            reviews = [r.strip() for r in batch_input.split('\n') if r.strip()]
            
            with st.spinner(f"Analyzing {len(reviews)} reviews..."):
                results = []
                for review in reviews:
                    pred, probs = analyze_sentiment(review, model, vectorizer)
                    if pred is not None:
                        sentiment = ["Unsatisfied", "Neutral", "Satisfied"][pred]
                        results.append({
                            "Review": review[:100] + "..." if len(review) > 100 else review,
                            "Sentiment": sentiment,
                            "Confidence": f"{probs[pred]*100:.1f}%",
                            "Unsatisfied": f"{probs[0]*100:.1f}%",
                            "Neutral": f"{probs[1]*100:.1f}%",
                            "Satisfied": f"{probs[2]*100:.1f}%"
                        })
                
                if results:
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True, hide_index=True)
                    
                    # Summary statistics
                    st.subheader("📈 Summary Statistics")
                    sentiment_counts = pd.DataFrame(results)['Sentiment'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.bar_chart(sentiment_counts)
                    with col2:
                        st.metric("Total Reviews", len(results))
                        st.metric("Unique Reviews", len(set([r['Review'] for r in results])))
        else:
            st.warning("Please enter at least one review")

# --- TAB 3: Model Information ---
with tab3:
    st.subheader("📊 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "78%", delta="+2%", delta_color="normal")
    with col2:
        st.metric("Precision (Satisfied)", "90%")
    with col3:
        st.metric("Recall (Unsatisfied)", "73%")
    
    st.markdown("---")
    st.subheader("🎯 Feature Importance")
    st.markdown("""
    ### Top Positive Indicators
    - great, best, delicious, love, perfect, excellent
    
    ### Top Negative Indicators  
    - horrible, waste, terrible, disappointed, awful
    
    ### Neutral Indicators
    - ok, okay, however, decent, probably
    """)
    
    st.markdown("---")
    st.subheader("💡 Tips for Better Results")
    st.info("""
    - Write detailed reviews (50+ characters work best)
    - Be specific about product features
    - Avoid overly short reviews like "good" or "bad"
    - Include both positive and negative aspects for nuanced analysis
    """)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    Powered by Logistic Regression | Trained on Amazon Reviews Dataset | Real-time sentiment analysis
</div>
""", unsafe_allow_html=True)
    
