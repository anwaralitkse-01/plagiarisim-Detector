import streamlit as st
import pickle
import string
import re
import nltk

from nltk.corpus import stopwords


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Plagiarism Detector",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 50%,
            #f8fafc 100%
        );
    }

    /* Header */
    .title {
        text-align: center;
        font-size: 45px;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #64748b;
        margin-bottom: 35px;
    }

    /* Cards */
    .card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0px 8px 25px rgba(15, 23, 42, 0.08);
        margin-bottom: 20px;
    }

    /* Result */
    .result-card {
        background: white;
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0px 10px 30px rgba(15, 23, 42, 0.10);
        margin-top: 25px;
    }

    .result-title {
        font-size: 28px;
        font-weight: 700;
        color: #1e293b;
    }

    .result-text {
        font-size: 24px;
        font-weight: 800;
        margin-top: 15px;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 12px;
        border: none;
        background: #4f46e5;
        color: white;
        font-size: 18px;
        font-weight: 700;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background: #3730a3;
        transform: translateY(-2px);
    }

    /* Text area */
    textarea {
        border-radius: 12px !important;
        border: 1px solid #cbd5e1 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e2e8f0;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: white;
        padding: 18px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0px 5px 15px rgba(15, 23, 42, 0.05);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 14px;
        margin-top: 45px;
        padding: 20px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# NLTK STOPWORDS
# ============================================================

try:
    stop_words = set(stopwords.words("english"))

except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


# ============================================================
# LOAD SVM MODEL AND VECTORIZER
# ============================================================

@st.cache_resource
def load_models():

    with open("svm.pkl", "rb") as model_file:
        svm_model = pickle.load(model_file)

    with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return svm_model, vectorizer


try:

    svm_model, vectorizer = load_models()

except Exception as e:

    st.error(
        "Could not load the model or vectorizer."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    if text is None:
        return ""

    if len(text.strip()) == 0:
        return ""

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Remove stopwords
    text = " ".join(
        word
        for word in text.split()
        if word not in stop_words
    )

    return text


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🔍 AI Plagiarism Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'NLP-based Plagiarism Detection using TF-IDF + SVM'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 About")

    st.write(
        """
        This application uses a trained Support Vector Machine
        (SVM) model to classify whether two pieces of text
        are likely to belong to the plagiarism class used
        during training.
        """
    )

    st.markdown("---")

    st.subheader("🧠 Model")

    st.write("**Algorithm:** Support Vector Machine")
    st.write("**Features:** TF-IDF")
    st.write("**Task:** Plagiarism Classification")

    st.markdown("---")

    st.subheader("⚙️ Pipeline")

    st.write("1. Text Input")
    st.write("2. Text Preprocessing")
    st.write("3. Text Combination")
    st.write("4. TF-IDF Transformation")
    st.write("5. SVM Prediction")
    st.write("6. Result")


# ============================================================
# INPUT SECTION
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# SOURCE TEXT
# ============================================================

with col1:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📄 Source / Original Text")

    source_text = st.text_area(
        "Source Text",
        height=300,
        placeholder=(
            "Paste the original/source text here..."
        ),
        label_visibility="collapsed"
    )

    source_word_count = len(
        source_text.split()
    )

    st.caption(
        f"Words: {source_word_count}"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# STUDENT TEXT
# ============================================================

with col2:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📝 Student / Suspected Text")

    plagiarized_text = st.text_area(
        "Student Text",
        height=300,
        placeholder=(
            "Paste the student/suspected text here..."
        ),
        label_visibility="collapsed"
    )

    student_word_count = len(
        plagiarized_text.split()
    )

    st.caption(
        f"Words: {student_word_count}"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# CHECK BUTTON
# ============================================================

st.write("")

check_button = st.button(
    "🔍 CHECK FOR PLAGIARISM"
)


# ============================================================
# PREDICTION
# ============================================================

if check_button:

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not source_text.strip():

        st.warning(
            "⚠️ Please enter the source/original text."
        )

        st.stop()


    if not plagiarized_text.strip():

        st.warning(
            "⚠️ Please enter the student/suspected text."
        )

        st.stop()


    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing text..."
    ):

        source_clean = preprocess_text(
            source_text
        )

        plagiarized_clean = preprocess_text(
            plagiarized_text
        )


        # ----------------------------------------------------
        # COMBINE TEXT
        # ----------------------------------------------------

        combined_text = (
            source_clean
            + " "
            + plagiarized_clean
        )


        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        text_vector = vectorizer.transform(
            [combined_text]
        )


        # ----------------------------------------------------
        # SVM PREDICTION
        # ----------------------------------------------------

        prediction = svm_model.predict(
            text_vector
        )[0]


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="result-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="result-title">'
        '📊 Analysis Result'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # HANDLE LABEL
    # --------------------------------------------------------

    # Assumes:
    # 1 = Plagiarism
    # 0 = No Plagiarism

    if prediction == 1:

        st.markdown(
            '<div class="result-text">'
            '⚠️ PLAGIARISM DETECTED'
            '</div>',
            unsafe_allow_html=True
        )

        st.error(
            "The SVM model classified this text pair "
            "as plagiarism."
        )

    else:

        st.markdown(
            '<div class="result-text">'
            '✅ NO PLAGIARISM DETECTED'
            '</div>',
            unsafe_allow_html=True
        )

        st.success(
            "The SVM model classified this text pair "
            "as non-plagiarism."
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # ========================================================
    # TEXT STATISTICS
    # ========================================================

    st.markdown("### 📈 Text Statistics")

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Source Words",
            source_word_count
        )


    with col2:

        st.metric(
            "Student Words",
            student_word_count
        )


    with col3:

        st.metric(
            "Prediction",
            "Plagiarism"
            if prediction == 1
            else "No Plagiarism"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <b>AI Plagiarism Detection System</b>
        <br>
        Built with Python • NLP • TF-IDF • SVM • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

