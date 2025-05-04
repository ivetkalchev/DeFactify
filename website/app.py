import joblib
from scipy.sparse import hstack
from flask import Flask, request, render_template
from text_preprocessing import preprocess_and_stem

app = Flask(__name__)

stacking_clf = joblib.load('website/models/stacking_svm.pkl')
vectorizer_title = joblib.load('website/models/vectorizer_title.pkl')
vectorizer_content = joblib.load('website/models/vectorizer_content.pkl')

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
    title   = request.form['title']
    content = request.form['content']

    # apply cleaning/stemming here
    clean_title   = preprocess_and_stem(title)
    clean_content = preprocess_and_stem(content)

    # transform — this will always yield shapes (1, 3000) and (1, 20000)
    ft_title   = vectorizer_title.transform([clean_title])
    ft_content = vectorizer_content.transform([clean_content])

    # stack to (1, 20000)
    X_pred = hstack([ft_title, ft_content])

    # predict
    pred_label = stacking_clf.predict(X_pred)[0]

    return render_template('result.html', prediction=pred_label)

if __name__ == '__main__':
    app.run(debug=True)