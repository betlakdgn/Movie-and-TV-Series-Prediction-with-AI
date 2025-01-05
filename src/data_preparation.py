import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_prepare_data(file_path):

    data = pd.read_csv(file_path)


    data = data[['DESCRIPTION', 'GENRE']]


    data.dropna(subset=['DESCRIPTION', 'GENRE'], inplace=True)


    smote = SMOTE(random_state=42)
    X = data['DESCRIPTION']
    y = data['GENRE']


    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(X).toarray()

    X_resampled, y_resampled = smote.fit_resample(X, y)


    joblib.dump(vectorizer, 'model/vectorizer.pkl')

    return X_resampled, y_resampled, X, y