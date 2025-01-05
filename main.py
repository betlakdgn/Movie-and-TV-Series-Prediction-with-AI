from src.data_preparation import load_and_prepare_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.predict import predict_genre
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"


if __name__ == "__main__":

    X_resampled, y_resampled, X, y = load_and_prepare_data('data/imdb_data.csv')


    X_test, y_test = train_model(X_resampled, y_resampled)


    evaluate_model(X_test, y_test)


    example = "A thrilling adventure of a superhero saving the world."
    print("Predicted Genre:", predict_genre(example))