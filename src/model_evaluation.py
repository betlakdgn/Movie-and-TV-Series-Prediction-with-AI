from sklearn.metrics import classification_report, accuracy_score
import joblib


def evaluate_model(X_test, y_test):
    # Model ve vektörizeri yükle
    model = joblib.load('model/genre_model.pkl')

    # Tahminler
    y_pred = model.predict(X_test)

    # Raporlama
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))
