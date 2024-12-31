import joblib


def predict_genre(overview):
    # Model ve vektörizeri yükle
    model = joblib.load('model/genre_model.pkl')
    vectorizer = joblib.load('model/vectorizer.pkl')

    # Açıklama üzerinden tür tahmini
    vectorized_input = vectorizer.transform([overview])
    prediction = model.predict(vectorized_input)

    return prediction[0]

