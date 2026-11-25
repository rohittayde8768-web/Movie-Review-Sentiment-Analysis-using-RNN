import streamlit as st
import numpy as np
from pathlib import Path
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.models import load_model

st.set_page_config(page_title="ReviewSense | Sentiment AI", page_icon="🎬", layout="wide")
MODEL_PATH = Path(__file__).parent / "rnn_model.keras"
MAX_FEATURES = 10_000
MAX_LEN = 500  # Must match the training script.

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp{background:radial-gradient(circle at 8% 0%,rgba(91,76,190,.22),transparent 32%),radial-gradient(circle at 95% 10%,rgba(35,177,193,.13),transparent 28%),#0b1020;color:#f4f7ff;font-family:'DM Sans',sans-serif}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#141d35,#10172a);border-right:1px solid #ffffff18}
h1,h2,h3{font-family:'Space Grotesk',sans-serif}
.hero{padding:2rem 2.2rem;border:1px solid #ffffff20;border-radius:24px;background:linear-gradient(115deg,#7e6cff33,#32c6d21a);box-shadow:0 18px 55px #0005;margin-bottom:1.5rem}
.eyebrow{color:#a69cff;text-transform:uppercase;letter-spacing:.16em;font-size:.76rem;font-weight:700}
.hero-title{color:white;font:700 clamp(2rem,4vw,3.25rem) 'Space Grotesk';line-height:1.1;margin:.55rem 0}
.hero-subtitle,.small-note{color:#b8c4df}
.glass-card{background:#121a2eee;border:1px solid #ffffff1a;border-radius:20px;padding:1.25rem 1.4rem;min-height:120px}
.metric-label{color:#9aa8c7;font-size:.85rem}.metric-value,.result-title{color:white;font:700 1.7rem 'Space Grotesk';margin-top:.35rem}
.result-positive{border-color:#36d39966;background:linear-gradient(135deg,#14785b38,#121a2ef0)}
.result-negative{border-color:#ff697d66;background:linear-gradient(135deg,#962d4538,#121a2ef0)}
div.stButton>button{border:0;border-radius:12px;color:white;font-weight:700;min-height:2.8rem;background:linear-gradient(90deg,#7667f5,#3bbdce)}
textarea{background:#111a2e!important;color:#f4f7ff!important;border-radius:12px!important}
[data-testid="stMetric"]{background:#121a2ecc;border:1px solid #ffffff18;padding:1rem;border-radius:16px}
hr{border-color:#ffffff1a} footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("rnn_model.keras must be in the same folder as app.py.")
    return load_model(MODEL_PATH), imdb.get_word_index()

def preprocess_review(text, word_index):
    words = text_to_word_sequence(text)
    encoded = []
    for word in words:
        index = word_index.get(word)
        # IMDB uses 0 for padding and 2 for out-of-vocabulary words.
        token = index + 3 if index is not None and index + 3 < MAX_FEATURES else 2
        encoded.append(token)
    return pad_sequences([encoded], maxlen=MAX_LEN, padding="pre", truncating="pre")

def predict_sentiment(text, model, word_index):
    x = preprocess_review(text, word_index)
    score = float(model.predict(x, verbose=0)[0][0])
    sentiment = "Positive" if score >= 0.5 else "Negative"
    confidence = score if sentiment == "Positive" else 1 - score
    return sentiment, score, confidence

with st.sidebar:
    st.markdown("# 🎬 ReviewSense")
    st.caption("AI-powered movie review analysis")
    st.markdown("---")
    st.markdown("### About this project")
    st.write("A Simple RNN model that classifies movie reviews as positive or negative.")
    st.markdown("### Model architecture")
    st.markdown("- Embedding — 64 dimensions\n- SimpleRNN — 64 units\n- Dense — Sigmoid output")
    st.markdown("---")
    st.caption("Dataset: IMDB Movie Reviews")
    st.caption("Framework: TensorFlow / Keras")
    st.caption("Input sequence length: 500")

st.markdown('''<div class="hero"><div class="eyebrow">NLP · DEEP LEARNING</div><div class="hero-title">Understand the feeling<br>behind every review.</div><div class="hero-subtitle">Enter a movie review and let the recurrent neural network identify whether its sentiment is positive or negative.</div></div>''', unsafe_allow_html=True)

c1,c2,c3 = st.columns(3)
for col, label, value, note in [(c1,"MODEL","Simple RNN","Sequential text learning"),(c2,"DATASET","IMDB","Movie review vocabulary"),(c3,"OUTPUT","2 Classes","Positive or negative")]:
    with col:
        st.markdown(f'<div class="glass-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="small-note">{note}</div></div>',unsafe_allow_html=True)

st.write("")
left,right = st.columns([1.15,.85],gap="large")
examples = {
    "Choose an example…":"",
    "Positive review":"This movie was absolutely amazing. The acting was brilliant, the story was beautiful, and I enjoyed every minute.",
    "Negative review":"This was a terrible movie. The story was boring, the acting was weak, and I regret watching it.",
    "Mixed review":"The visuals were impressive, but the story was slow and the characters were not very interesting."
}
with left:
    st.markdown("### ✍️ Write your movie review")
    selected = st.selectbox("Try a sample review",list(examples))
    review = st.text_area("Your review",value=examples[selected],height=210,placeholder="Write or paste a movie review here…",label_visibility="collapsed")
    analyze = st.button("✨ Analyze sentiment",use_container_width=True)
with right:
    st.markdown("### 📊 How to read the result")
    st.markdown('''<div class="glass-card"><div class="result-title">Positive 😊</div><p class="small-note">Model score is 0.50 or higher.</p><hr><div class="result-title">Negative ☹️</div><p class="small-note">Model score is below 0.50.</p><hr><p class="small-note">The score is the model's sigmoid output, not a guarantee of correctness.</p></div>''',unsafe_allow_html=True)

if analyze:
    if not review.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        try:
            with st.spinner("Analyzing your review…"):
                model, word_index = load_assets()
                sentiment, score, confidence = predict_sentiment(review,model,word_index)
            st.markdown("---\n## Analysis result")
            css_class = "result-positive" if sentiment == "Positive" else "result-negative"
            emoji = "😊" if sentiment == "Positive" else "☹️"
            st.markdown(f'<div class="glass-card {css_class}"><div class="metric-label">PREDICTED SENTIMENT</div><div class="result-title">{emoji} {sentiment}</div><div class="small-note">Confidence indicator: {confidence:.1%}</div></div>',unsafe_allow_html=True)
            a,b = st.columns(2)
            a.metric("Positive score",f"{score:.1%}")
            b.metric("Negative score",f"{1-score:.1%}")
            st.progress(min(max(score,0.0),1.0),text="Positive sentiment score")
            st.caption("Scores closer to 1 indicate positive sentiment; scores closer to 0 indicate negative sentiment.")
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Could not analyze this review: {e}")

st.markdown("---")
st.markdown('<p class="small-note" style="text-align:center">ReviewSense · Movie Review Sentiment Analysis · Powered by TensorFlow and Streamlit</p>',unsafe_allow_html=True)
