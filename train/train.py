# import argparse
# import os
# import pickle

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
# from sklearn.pipeline import Pipeline


# def train(train_path: str, model_output: str) -> None:
#     df = pd.read_csv(train_path)
#     X, y = df["text"], df["label"]

#     pipeline = Pipeline([
#         ("tfidf", TfidfVectorizer(max_features=5000)),
#         ("clf", LogisticRegression(max_iter=1000)),
#     ])

#     pipeline.fit(X, y)

#     os.makedirs(os.path.dirname(model_output), exist_ok=True)
#     with open(model_output, "wb") as f:
#         pickle.dump(pipeline, f)

#     print(classification_report(y, pipeline.predict(X)))
#     print(f"Model saved to {model_output}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train-path", default="data/train.csv")
#     parser.add_argument("--model-output", default="model/model.pkl")
#     args = parser.parse_args()

#     train(args.train_path, args.model_output)
