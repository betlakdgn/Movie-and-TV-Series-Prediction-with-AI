import joblib


def predict_genre(overview):

    model = joblib.load('model/genre_model.pkl')
    vectorizer = joblib.load('model/vectorizer.pkl')


    vectorized_input = vectorizer.transform([overview])
    prediction = model.predict(vectorized_input)

    return prediction[0]

