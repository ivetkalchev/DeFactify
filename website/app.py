import joblib
import numpy 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request

# NLTK data setup
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# load vectorizers and model
vectorizer_title = joblib.load("website/model/vectorizer_title.pkl")
vectorizer_content = joblib.load("website/model/vectorizer_content.pkl")
model = joblib.load("website/model/stacking_model.pkl")

# preprocessing setup
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_and_stem(text):
    tokens = word_tokenize(text.lower())
    stemmed_tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(stemmed_tokens)

@app.route("/predict", methods=["POST"])
def index():
    prediction = None
    if request.method == "POST":
        title = request.form.get("title")
        content = request.form.get("content")

        # preprocess
        preprocessed_title = preprocess_and_stem(title)
        preprocessed_content = preprocess_and_stem(content)

        # vectorize
        title_vec = vectorizer_title.transform([preprocessed_title]).toarray()
        content_vec = vectorizer_content.transform([preprocessed_content]).toarray()

        # combine
        combined_vec = numpy.hstack((title_vec, content_vec))

        # predict
        pred = model.predict(combined_vec)
        prediction = "Real News" if pred[0] == 1 else "Fake News"

    return render_template('result.html', prediction=prediction)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detector')
def detector():
    return render_template('detector.html')

if __name__ == "__main__":
    app.run(debug=True)