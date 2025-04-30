from flask import Flask, request, render_template
import joblib
import numpy as np
from text_processing import preprocess_and_stem 

app = Flask(__name__)

# Load models
nb_model = joblib.load('website/models/naive_bayes_model.pkl')
tree_model = joblib.load('website/models/decision_tree_model.pkl')
vectorizer_title = joblib.load('website/models/vectorizer_title.pkl')
vectorizer_content = joblib.load('website/models/vectorizer_content.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detector')
def detector():
    return render_template('detector.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    content = request.form['content']
    author_listed = request.form.get('author_listed') == 'True'

    # Preprocess inputs
    processed_title = preprocess_and_stem(title)
    processed_content = preprocess_and_stem(content)

    # Transform inputs
    title_features = vectorizer_title.transform([processed_title]).toarray()
    content_features = vectorizer_content.transform([processed_content]).toarray()

    # Combine features for Naive Bayes (assuming both title + content)
    combined_features = np.hstack((title_features, content_features))

    # Ensure the input size matches the model's expected size for Naive Bayes
    nb_feature_size = nb_model.n_features_in_
    if combined_features.shape[1] != nb_feature_size:
        padding = np.zeros((combined_features.shape[0], nb_feature_size - combined_features.shape[1]))
        combined_features = np.hstack((combined_features, padding))

    # Predict for Naive Bayes
    nb_prediction = nb_model.predict(combined_features)

    # Decision Tree: Use author_listed status
    tree_features = np.array([[int(author_listed)]])  # Shape (1,1)
    tree_prediction = tree_model.predict(tree_features)

    # Naive Bayes: Keep existing text feature processing
    nb_prediction = nb_model.predict(combined_features)

    # Combine predictions
    final_prediction = (nb_prediction + tree_prediction) > 1

    # Display the author listed status in the result page
    author_status = "Yes" if author_listed else "No"

    return render_template('result.html', prediction=final_prediction[0], author_status=author_status)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)