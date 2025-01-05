from sklearn.model_selection import  GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy')


    grid_search.fit(X_train, y_train)


    print("En iyi parametreler:", grid_search.best_params_)


    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'model/genre_model.pkl')

    return X_test, y_test