import numpy
import joblib
from text_processing import preprocess_and_stem
from flask import Flask, request, render_template

app = Flask(__name__)

# models
nb_model = joblib.load('website/models/naive_bayes_model.pkl')
vectorizer_title = joblib.load('website/models/vectorizer_title.pkl')
vectorizer_content = joblib.load('website/models/vectorizer_content.pkl')
dt_model = joblib.load('website/models/decision_tree_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detector')
def detector():
    return render_template('detector.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    content = request.form['content']
    author_listed = request.form.get('author_listed') == 'True'

    # preprocess inputs
    processed_title = preprocess_and_stem(title)
    processed_content = preprocess_and_stem(content)

    # transform inputs using vectorizers
    title_features = vectorizer_title.transform([processed_title]).toarray()
    content_features = vectorizer_content.transform([processed_content]).toarray()

    # combine features for Naive Bayes
    combined_features = numpy.hstack((title_features, content_features))
    author_feature = numpy.array([[1 if author_listed else 0]])
    combined_features = numpy.hstack((combined_features, author_feature))

    # input size matches the Naive Bayes' expected size
    nb_feature_size = nb_model.n_features_in_
    if combined_features.shape[1] != nb_feature_size:
        padding = numpy.zeros((combined_features.shape[0], nb_feature_size - combined_features.shape[1]))
        combined_features = numpy.hstack((combined_features, padding))

    # predict with Naive Bayes
    nb_prediction_prob = nb_model.predict_proba(combined_features)[0]
    fake_news_prob = nb_prediction_prob[0]
    real_news_prob = nb_prediction_prob[1]

    # prepare features for Decision Tree
    dt_features = author_feature
    dt_prediction = dt_model.predict(dt_features)[0]

    # adjust probabilities based on the Decision Tree output
    if dt_prediction == 0: 
        fake_news_prob += 0.2  # increase chance of Fake News
        real_news_prob -= 0.1  # decrease chance of Real News

        total = fake_news_prob + real_news_prob
        fake_news_prob /= total
        real_news_prob /= total

    # final prediction
    final_prediction = "Real News" if real_news_prob > fake_news_prob else "Fake News"

    # text output
    final_prediction_text = f"{final_prediction}"

    return render_template('result.html', prediction=final_prediction_text)

if __name__ == '__main__':
    app.run(debug=True)