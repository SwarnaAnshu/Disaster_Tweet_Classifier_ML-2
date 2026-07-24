import streamlit as st
import joblib
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


@st.cache_resource
def load_artifacts():
    model = joblib.load("trained_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_artifacts()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


st.set_page_config(page_title="DisasterAlert: Disaster Tweet Classifier", page_icon="🌪️")

st.markdown(
    """
    <style>
    div[data-testid="stButton"] button,
    .stButton > button,
    button[kind="primary"],
    button[kind="secondary"] {
        max-width: 140px !important;
        min-width: 100px !important;
        width: auto !important;
        margin: 0 auto !important;
        display: block !important;
        white-space: nowrap !important;
    }
    @media (max-width: 640px) {
        div[data-testid="stButton"] button {
            max-width: 100px !important;
            min-width: 80px !important;
            font-size: 12px !important;
            padding: 5px 8px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("ℹ️ About This App")
    st.write(
        "**DisasterAlert** predicts whether a tweet refers to a **real disaster** "
        "or is using disaster-related language casually/metaphorically."
    )

    st.subheader("🧠 Model")
    st.write("Multinomial Naive Bayes (tuned via GridSearchCV)")

    st.subheader("🛠️ Tech Stack")
    st.markdown(
        "- Streamlit\n"
        "- Scikit-Learn\n"
        "- NLTK\n"
        "- TF-IDF Vectorization\n"
        "- imbalanced-learn\n"
        "- Joblib"
    )

    st.subheader("🧩 Built with")
    st.write("Text-Based Machine Learning Capstone Project")

    st.markdown(
        """
        <div style="margin-top:16px; background-color:#1f2937; border:1px solid #4a5568;
                    border-radius:8px; padding:12px 16px; text-align:center;">
            <span style="font-size:14px; color:#cbd5e0;">🚀 Deployed by</span><br>
            <span style="font-size:16px; font-weight:700; color:#ffffff;">Anshu Swarna</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

if "tweet_text" not in st.session_state:
    st.session_state.tweet_text = ""


def clear_input():
    st.session_state.tweet_text = ""
    st.session_state.pop("result", None)


st.title("🌪️ DisasterAlert: Disaster Tweet Classifier")

st.write("Predict whether a tweet refers to a **real disaster** or **not**, using Machine Learning.")

with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
    st.subheader("💬 Enter Tweet Text")
    st.button("❌ Clear", on_click=clear_input)

tweet_input = st.text_area(
    label="Tweet",
    key="tweet_text",
    height=150,
    placeholder="Type or paste your tweet here...",
    label_visibility="collapsed",
)

with st.container(horizontal=True, horizontal_alignment="center"):
    predict_clicked = st.button("🔍 Predict", type="primary")

if predict_clicked:
    tweet = st.session_state.tweet_text

    if not tweet or not tweet.strip():
        st.warning("Please enter a tweet to classify.")
        st.session_state.pop("result", None)
    else:
        cleaned = clean_text(tweet)

        if not cleaned:
            st.warning("Tweet contains no usable text after cleaning — please enter a valid tweet.")
            st.session_state.pop("result", None)
        else:
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0][1]

            if pred == 1:
                confidence = proba
                confidence_label = "Real Disaster Confidence"
                st.session_state["result"] = {
                    "label": "🚨 Real Disaster",
                    "kind": "error",
                    "confidence": confidence,
                    "confidence_label": confidence_label,
                }
            else:
                confidence = 1 - proba
                confidence_label = "Not a Disaster Confidence"
                st.session_state["result"] = {
                    "label": "✅ Not a Disaster",
                    "kind": "success",
                    "confidence": confidence,
                    "confidence_label": confidence_label,
                }

if "result" in st.session_state:
    result = st.session_state["result"]

    st.subheader("🎯 Prediction Result")

    if result["kind"] == "error":
        st.error(result["label"])
    else:
        st.success(result["label"])

    pct = result["confidence"] * 100
    bar_color = "#ff4d4d" if result["kind"] == "error" else "#4caf50"
    bar_text = f"{result['confidence_label']}: {pct:.2f}%"

    st.markdown(
        f"""
        <div style="margin-top:6px;">
            <div style="background-color:#333333; border-radius:8px; height:38px; width:100%;
                        overflow:hidden; position:relative;">
                <div style="background-color:{bar_color}; height:100%; width:{pct}%;
                            transition:width 0.3s;">
                </div>
                <div style="position:absolute; top:0; left:0; width:100%; height:100%;
                            display:flex; align-items:center; justify-content:center;
                            font-size:15px; font-weight:700; color:white; text-shadow:0 1px 2px rgba(0,0,0,0.6);">
                    {bar_text}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")