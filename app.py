import streamlit as st
import pandas as pd
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Download NLTK data (if not already downloaded) ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

download_nltk_data()

# --- Load Models and Encoder ---
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load("random_forest_model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        return rf_model, label_encoder
    except FileNotFoundError:
        st.error("Error: Model or label encoder file not found.")
        st.write("Please ensure 'random_forest_model.pkl' and 'label_encoder.pkl' are in the same directory.")
        st.stop()

rf_model, label_encoder = load_models()

# --- Initialize NLP Tools ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
vader = SentimentIntensityAnalyzer()

# Load spaCy model (download it first using: python -m spacy download en_core_web_sm)
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy model 'en_core_web_sm' not found.")
        st.write("Please install it by running: `python -m spacy download en_core_web_sm` in your terminal.")
        st.stop()

nlp = load_spacy_model()


# --- Feature Extraction Functions (Matching the notebook) ---

# Define common negation terms
negations = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor', 'nowhere',
             'hardly', 'scarcely', 'barely', 'without', 'lack', 'failed'}

# Modified clean_text function to include negation handling
def clean_text_with_negation(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)

    new_tokens = []
    negate = False
    window = 3  # check next 3 words after negation

    for i, token in enumerate(tokens):
        if token in negations:
            negate = True
            count = 0
            continue

        if negate:
            if count < window:
                new_tokens.append("NEG_" + token)
                count += 1
            else:
                negate = False
                new_tokens.append(token)
        else:
            new_tokens.append(token)

    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in new_tokens if word not in stop_words]
    return ' '.join(tokens)

# Define strong financial sentiment words
strong_words = ['soars', 'slumps', 'plunges', 'surges', 'crashes', 'rallies', 'jumps', 'tumbles', 'skyrockets']

# Financial keywords to track position
financial_keywords = ['profit', 'loss', 'revenue', 'growth', 'decline', 'market', 'stock', 'investment']


# Example subset of Loughran-McDonald sentiment words
lm_positive = {'profit', 'gain', 'growth', 'improved', 'success', 'exceed', 'outperform', 'surged'}
lm_negative = {'loss', 'decline', 'lawsuit', 'drop', 'fall', 'underperform', 'layoff', 'debt', 'bankruptcy'}

# Define event phrases and their associated scores
event_phrases_dict = {
    "filed for bankruptcy": -2,
    "announced layoffs": -1,
    "cut jobs": -1,
    "beat expectations": 1,
    "exceeded expectations": 1,
    "missed expectations": -1,
    "reported strong earnings": 1,
    "warned of losses": -1,
    "strong revenue": 1,
    "decline in revenue": -1,
    "record profits": 1,
    "record losses": -1,
    "laid off": -1,
    "hired more staff": 1,
    "raised forecast": 1,
    "lowered forecast": -1,
    "acquired": 1,
    "invested in": 1,
    "fined": -1,
    "sued": -1,
    "won lawsuit": 1,
    "lost lawsuit": -1
}

# Define positive and negative (entity type, verb) sentiment pairs
positive_patterns = {
    ('ORG', 'acquired'), ('ORG', 'invested'), ('ORG', 'announced'), ('GPE', 'invested'),
    ('ORG', 'expanded'), ('ORG', 'reported'), ('ORG', 'launched')
}
negative_patterns = {
    ('ORG', 'sued'), ('ORG', 'fined'), ('ORG', 'laid'), ('ORG', 'cut'),
    ('GPE', 'banned'), ('ORG', 'filed'), ('ORG', 'failed'), ('ORG', 'halted')
}

# Financial nouns typically tied to sentiment
financial_nouns = {'profit', 'loss', 'growth', 'decline', 'earnings', 'forecast', 'revenue', 'expenses', 'sale', 'layoffs'}

# Define SVO patterns with sentiment weights
positive_svo_patterns = {
    ('announced', 'profits'), ('reported', 'earnings'), ('raised', 'forecast'),
    ('acquired', 'company'), ('invested', 'funds'), ('won', 'lawsuit'), ('beat', 'expectations')
}

negative_svo_patterns = {
    ('filed', 'bankruptcy'), ('announced', 'layoffs'), ('reported', 'loss'),
    ('cut', 'jobs'), ('faced', 'lawsuit'), ('missed', 'expectations'), ('failed', 'target')
}


# Re-implement feature extraction functions based on your notebook
def get_vader_score(text):
    return vader.polarity_scores(text)['compound']

def get_strong_word_count(text):
    tokens = text.lower().split()
    return sum(1 for word in tokens if word in strong_words)

def get_keyword_position_score(text):
    tokens = text.lower().split()
    position_scores = []
    for kw in financial_keywords:
        if kw in tokens:
            position_scores.append(1 - tokens.index(kw) / len(tokens))
    return round(sum(position_scores), 3) if position_scores else 0.0

def get_named_entity_count(text):
    doc = nlp(text)
    return len([ent for ent in doc.ents if ent.label_ in ['ORG', 'GPE', 'PERSON']])

def get_lm_score(text, pos=True):
    tokens = text.lower().split()
    if pos:
        return sum(1 for word in tokens if word in lm_positive)
    else:
        return sum(1 for word in tokens if word in lm_negative)

def get_negation_flip_count(text):
    tokens = text.lower().split()
    neg_flip_count = 0
    for i, word in enumerate(tokens[:-1]):
        if word in negations and tokens[i+1] in lm_positive.union(lm_negative):
            neg_flip_count += 1
    return neg_flip_count

def get_event_phrase_score(text):
    text = text.lower()
    score = 0
    for phrase, value in event_phrases_dict.items():
        if phrase in text:
            score += value
    return score

def get_verb_near_org_count(text):
    doc = nlp(text)
    verb_near_org_count = 0
    org_indices = [ent.start for ent in doc.ents if ent.label_ == "ORG"]

    for i, token in enumerate(doc):
        if token.pos_ == "VERB":
            if any(abs(i - org_i) <= 3 for org_i in org_indices):  # within 3-token window
                verb_near_org_count += 1
    return verb_near_org_count

def get_adj_noun_combo_count(text):
    doc = nlp(text)
    adj_noun_combo_count = 0
    for i in range(len(doc)-1):
        if doc[i].pos_ == "ADJ" and doc[i+1].pos_ == "NOUN":
            if doc[i+1].lemma_.lower() in financial_nouns:
                adj_noun_combo_count += 1
    return adj_noun_combo_count

def get_entity_verb_sentiment_score(text):
    doc = nlp(text)
    score = 0
    for ent in doc.ents:
        if ent.label_ in {'ORG', 'GPE'}:
            ent_index = ent.start
            # Search for verbs within a window of 3 tokens
            for i in range(max(0, ent_index - 3), min(len(doc), ent_index + 4)):
                token = doc[i]
                if token.pos_ == 'VERB':
                    pattern = (ent.label_, token.lemma_)
                    if pattern in positive_patterns:
                        score += 1
                    elif pattern in negative_patterns:
                        score -= 1
    return score


def polarity_window_scan(text):
    text = text.lower()
    words = text.split()
    if len(words) < 4:
        return 0, 0, 0  # Not enough words

    # Split in two halves
    mid = len(words) // 2
    first_half = ' '.join(words[:mid])
    second_half = ' '.join(words[mid:])

    # Compute sentiment scores
    first_score = vader.polarity_scores(first_half)['compound']
    second_score = vader.polarity_scores(second_half)['compound']

    # Detect reversal
    reversal = int((first_score * second_score) < 0)  # flip in polarity

    return first_score, second_score, reversal

def get_first_half_sentiment(text):
    return polarity_window_scan(text)[0]

def get_second_half_sentiment(text):
    return polarity_window_scan(text)[1]

def get_sentiment_reversal_flag(text):
    return polarity_window_scan(text)[2]

def extract_svo_sentiment_score(text):
    doc = nlp(text)
    score = 0
    for token in doc:
        if token.pos_ == "VERB":
            subj = [w.text.lower() for w in token.lefts if w.dep_ in {"nsubj", "nsubjpass"}]
            obj = [w.text.lower() for w in token.rights if w.dep_ in {"dobj", "pobj"}]

            if subj and obj:
                verb = token.lemma_.lower()
                if (verb, obj[0]) in positive_svo_patterns:
                    score += 1
                elif (verb, obj[0]) in negative_svo_patterns:
                    score -= 1
    return score


# --- Combined Feature Extraction Function (for single text) ---
def extract_features_from_text(text):
    # Ensure text is not empty before processing
    if not text:
        return {
            'sentiment_score_vader': 0.0,
            'strong_word_count': 0,
            'keyword_position_score': 0.0,
            'named_entity_count': 0,
            'lm_positive_score': 0,
            'lm_negative_score': 0,
            'negation_flip_count': 0,
            'event_phrase_score': 0,
            'verb_near_org_count': 0,
            'adj_noun_combo_count': 0,
            'entity_verb_sentiment_score': 0,
            'first_half_sentiment': 0.0,
            'second_half_sentiment': 0.0,
            'sentiment_reversal': 0,
            'svo_sentiment_score': 0
        }

    first_half_sent, second_half_sent, reversal_flag = polarity_window_scan(text)

    return {
        'sentiment_score_vader': get_vader_score(text),
        'strong_word_count': get_strong_word_count(text),
        'keyword_position_score': get_keyword_position_score(text),
        'named_entity_count': get_named_entity_count(text),
        'lm_positive_score': get_lm_score(text, pos=True),
        'lm_negative_score': get_lm_score(text, pos=False),
        'negation_flip_count': get_negation_flip_count(text),
        'event_phrase_score': get_event_phrase_score(text),
        'verb_near_org_count': get_verb_near_org_count(text),
        'adj_noun_combo_count': get_adj_noun_combo_count(text),
        'entity_verb_sentiment_score': get_entity_verb_sentiment_score(text),
        'first_half_sentiment': first_half_sent,
        'second_half_sentiment': second_half_sent,
        'sentiment_reversal': reversal_flag,
        'svo_sentiment_score': extract_svo_sentiment_score(text)
    }

# --- Function to extract features and predict for multiple texts ---
def analyze_multiple_texts(texts_list):
    if not texts_list:
        return []

    all_features = []
    for text in texts_list:
        features = extract_features_from_text(text)
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)

    feature_columns = [
        'sentiment_score_vader',
        'strong_word_count',
        'keyword_position_score',
        'named_entity_count',
        'lm_positive_score',
        'lm_negative_score',
        'negation_flip_count',
        'event_phrase_score',
        'verb_near_org_count',
        'adj_noun_combo_count',
        'entity_verb_sentiment_score',
        'first_half_sentiment',
        'second_half_sentiment',
        'sentiment_reversal',
        'svo_sentiment_score'
    ]
    # Reindex to ensure consistent column order, filling missing with 0
    features_df = features_df.reindex(columns=feature_columns, fill_value=0)

    predictions = rf_model.predict(features_df)
    probabilities = rf_model.predict_proba(features_df)

    results = []
    for i, text in enumerate(texts_list):
        predicted_label_encoded = predictions[i]
        predicted_sentiment = label_encoder.inverse_transform([predicted_label_encoded])[0]
        prob_dict = dict(zip(label_encoder.classes_, probabilities[i]))
        
        results.append({
            'text': text,
            'sentiment': predicted_sentiment,
            'confidence': prob_dict,
            'features': all_features[i]
        })
    return results


# --- Streamlit App ---
st.set_page_config(
    page_title="Financial News Sentiment Analyzer",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom CSS for a modern, dark theme with background image
st.markdown("""
<style>
    /* Main App Container */
    .stApp {
        background-color: #be2fed; /* Fallback color */
        background-image: url("https://images.unsplash.com/photo-1543286386-713bdd59760f?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"); /* Placeholder image URL */
        background-size: cover; /* Cover the entire container */
        background-repeat: no-repeat; /* Do not repeat the image */
        background-attachment: fixed; /* Fix the background image when scrolling */
        color: #e0e0e0;
        font-family: 'Segoe UI', 'Roboto', 'Helvetica', sans-serif;
    }
    
    /* Optional: Add a semi-transparent overlay to improve text readability */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.5); /* Dark overlay with 50% opacity */
        z-index: -1; /* Place behind content */
    }

    /* Headers and Titles */
    h1 {
        font-size: 2.5em;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        margin-bottom: 20px;
    }
    h2 {
        font-size: 1.8em;
        font-weight: 600;
        color: #ffffff;
        border-bottom: 2px solid #282828;
        padding-bottom: 5px;
        margin-top: 30px;
    }

    /* Text Area */
    .stTextArea label {
        font-size: 1.2em;
        font-weight: bold;
        color: #ffffff;
    }
    .stTextArea textarea {
        background-color: #1e1e1e;
        color: #e0e0e0;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 10px;
        font-size: 1.1em;
        transition: border-color 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #00a884; /* Accent color on focus */
        outline: none;
        box-shadow: 0 0 0 2px rgba(0, 168, 132, 0.5);
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(45deg, #00a884, #00c79f);
        color: white;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 10px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.3);
    }

    /* Result Containers */
    .stSuccess, .stError, .stWarning {
        padding: 20px;
        border-radius: 12px;
        font-size: 1.3em;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        border-left-width: 8px;
        border-left-style: solid;
        background-color: #1e1e1e;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .stSuccess {
        border-color: #00c79f;
        color: #00c79f;
    }
    .stError {
        border-color: #ff4c4c;
        color: #ff4c4c;
    }
    .stWarning {
        border-color: #ffcc00;
        color: #ffcc00;
    }

    /* Metrics (Confidence Scores) */
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        text-align: center;
        border-top: 4px solid #333333;
    }
    .stMetric label {
        color: #999999;
        font-size: 0.9em;
    }
    .stMetric .st-bd {
        font-size: 1.5em;
        color: #e0e0e0;
        font-weight: 600;
    }
    
    /* Expander */
    .stExpander {
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 10px;
        background-color: #1e1e1e;
    }
    .stExpander > div > div > button {
        color: #e0e0e0;
        font-weight: bold;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] h2 {
        color: #ffffff;
        border-bottom: none;
    }
    /* Checkbox styling for news selection */
    .stCheckbox label {
        color: #e0e0e0;
    }
    .stCheckbox > label > div {
        background-color: #282828;
        border-radius: 5px;
        padding: 8px;
        margin-bottom: 5px;
        border: 1px solid #333333;
    }
    .stCheckbox > label > div:hover {
        background-color: #3a3a3a;
    }
    .stCheckbox > label > div[aria-checked="true"] {
        border-color: #00a884;
        background-color: #00a88420;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Financial News Sentiment Analyzer")

st.write(
    "Welcome to the Financial News Sentiment Analyzer! "
    "This tool uses a machine learning model to predict the sentiment "
    "(Positive, Negative, or Neutral) of financial news headlines or short texts."
)
st.markdown("---")

# --- Single Text Input Section ---
st.header("✍️ Enter Your Text (Single Prediction)")
user_input = st.text_area(
    "Type or paste a financial news headline here:",
    "", # No pre-population from API anymore
    height=150,
    placeholder="e.g., 'Company X reports record profits amid strong market performance'"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Predict Sentiment (Single Text)"):
        if user_input:
            with st.spinner("Analyzing sentiment..."):
                # Extract features
                features = extract_features_from_text(user_input)
                features_df = pd.DataFrame([features])

                # Ensure column order matches training data
                feature_columns = [
                    'sentiment_score_vader',
                    'strong_word_count',
                    'keyword_position_score',
                    'named_entity_count',
                    'lm_positive_score',
                    'lm_negative_score',
                    'negation_flip_count',
                    'event_phrase_score',
                    'verb_near_org_count',
                    'adj_noun_combo_count',
                    'entity_verb_sentiment_score',
                    'first_half_sentiment',
                    'second_half_sentiment',
                    'sentiment_reversal',
                    'svo_sentiment_score'
                ]
                # Reindex to ensure consistent column order, filling missing with 0
                features_df = features_df.reindex(columns=feature_columns, fill_value=0)


                # Predict
                try:
                    prediction = rf_model.predict(features_df)
                    predicted_label_encoded = prediction[0]
                    predicted_sentiment = label_encoder.inverse_transform([predicted_label_encoded])[0]

                    st.markdown("---")
                    st.header("📊 Prediction Result (Single Text)")
                    if predicted_sentiment == 'positive':
                        st.success(f"Sentiment: {predicted_sentiment.upper()} 🎉")
                    elif predicted_sentiment == 'negative':
                        st.error(f"Sentiment: {predicted_sentiment.upper()} 📉")
                    else:
                        st.warning(f"Sentiment: {predicted_sentiment.upper()} ⚖️")

                    # Display probabilities in a more visual way
                    st.subheader("Confidence Scores:")
                    probs = rf_model.predict_proba(features_df)[0]
                    prob_dict = dict(zip(label_encoder.classes_, probs))

                    prob_cols = st.columns(len(prob_dict))
                    for i, (label, prob) in enumerate(prob_dict.items()):
                        with prob_cols[i]:
                            st.metric(label=label.capitalize(), value=f"{prob*100:.2f}%")

                    # Display extracted features in an expander
                    with st.expander("🔍 View Extracted Features"):
                        st.json(features)

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

        else:
            st.warning("Please enter some text to predict.")

st.markdown("---")

# --- Multiple Manual Input Section ---
st.header("📝 Analyze Multiple Texts (Manual Input)")
multi_user_input = st.text_area(
    "Enter multiple financial news headlines (one per line):",
    "",
    height=200,
    placeholder="e.g.,\nCompany A announces record profits.\nStock market experiences a sharp decline.\nNew policy has neutral impact on economy."
)

if st.button("🚀 Predict Sentiments for Multiple Texts"):
    if multi_user_input:
        texts_to_analyze = [line.strip() for line in multi_user_input.split('\n') if line.strip()]
        if texts_to_analyze:
            with st.spinner("Analyzing multiple texts..."):
                multi_results = analyze_multiple_texts(texts_to_analyze)
                st.markdown("---")
                st.header("📈 Multi-Text Sentiment Results")
                if multi_results:
                    for i, result in enumerate(multi_results):
                        st.subheader(f"Text {i+1}: {result['sentiment'].upper()}")
                        st.write(f"**Text:** {result['text']}")
                        
                        prob_cols = st.columns(len(result['confidence']))
                        for j, (label, prob) in enumerate(result['confidence'].items()):
                            with prob_cols[j]:
                                st.metric(label=label.capitalize(), value=f"{prob*100:.2f}%")
                        
                        with st.expander(f"🔍 View Extracted Features for Text {i+1}"):
                            st.json(result['features'])
                        st.markdown("---")
                else:
                    st.info("No valid texts entered for analysis.")
        else:
            st.warning("Please enter at least one headline for multi-text analysis.")
    else:
        st.warning("Please enter some text for multi-text analysis.")


st.markdown("---")
st.info(
    "**How it works:** This app extracts various linguistic features from your text, "
    "such as sentiment scores, keyword positions, named entity counts, and "
    "subject-verb-object patterns. These features are then fed into a "
    "pre-trained Random Forest Classifier to determine the overall sentiment."
)

st.sidebar.header("About This App")
st.sidebar.write(
    "This application demonstrates a simple Natural Language Processing (NLP) "
    "pipeline for financial news sentiment analysis. It leverages NLTK and spaCy "
    "for text processing and a scikit-learn Random Forest model for classification."
)
st.sidebar.markdown("---")
st.sidebar.write("Developed by: \n* Ganapathi \n* sriram \n* sivasai \n* shivathmika \n* varsha")
