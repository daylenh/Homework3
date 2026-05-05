from src.data_loader import load_data
from src.preprocessing import preprocess
from src.features import tfidf_features
from src.embeddings import train_word2vec, vectorize_text
from src.models import (
    train_nb, train_svm, train_rf, train_knn, train_dt, predict,
    train_nn, predict_nn
)
from src.evaluate import evaluate
from src.visualize import plot_accuracy, plot_speed, plot_test_speed, plot_tradeoff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main():
    # load data
    X, y = load_data()
    print("Unique labels:", y.unique())
    print("Label distribution:\n", y.value_counts())
    print("Sample texts:")
    print(X[:5].values)
    print("Sample labels:")
    print(y[:5].values)
    # preprocess
    X = preprocess(X)
    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    results = []

    # TF-IDF
    X_train_tfidf, X_test_tfidf = tfidf_features(X_train, X_test)

    # Naive Bayes
    model, train_time = train_nb(X_train_tfidf, y_train)
    y_pred, test_time = predict(model, X_test_tfidf)
    acc, p, r, f1 = evaluate(y_test, y_pred)
    results.append(("TFIDF+NB", acc, p, r, f1, train_time, test_time))
    # SVM
    model, train_time = train_svm(X_train_tfidf, y_train)
    y_pred, test_time = predict(model, X_test_tfidf)
    acc, p, r, f1 = evaluate(y_test, y_pred)
    results.append(("TFIDF+SVM", acc, p, r, f1, train_time, test_time))

    # Random Forest
    model, train_time = train_rf(X_train_tfidf, y_train)
    y_pred, test_time = predict(model, X_test_tfidf)
    acc, p, r, f1 = evaluate(y_test, y_pred)
    results.append(("TFIDF+RF", acc, p, r, f1, train_time, test_time))

    # KNN
    model, train_time = train_knn(X_train_tfidf, y_train)
    y_pred, test_time = predict(model, X_test_tfidf)
    acc, p, r, f1 = evaluate(y_test, y_pred)
    results.append(("TFIDF+KNN", acc, p, r, f1, train_time, test_time))

    # Decision Tree
    model, train_time = train_dt(X_train_tfidf, y_train)
    y_pred, test_time = predict(model, X_test_tfidf)
    acc, p, r, f1 = evaluate(y_test, y_pred)
    results.append(("TFIDF+DT", acc, p, r, f1, train_time, test_time))

    # Word2Vec + NN
    w2v = train_word2vec(X_train)

    X_train_vec = vectorize_text(w2v, X_train)
    X_test_vec = vectorize_text(w2v, X_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    model, train_time = train_nn(X_train_vec, y_train_enc)
    y_pred, test_time = predict_nn(model, X_test_vec)

    acc, p, r, f1 = evaluate(y_test_enc, y_pred)
    results.append(("W2V+NN", acc, p, r, f1, train_time, test_time))

    # ---------------- GloVe + NN ----------------
    # glove = load_glove()

    # X_train_glove = vectorize_glove(glove, X_train)
    # X_test_glove = vectorize_glove(glove, X_test)

    # model, train_time = train_nn(X_train_glove, y_train_enc)
    # y_pred, test_time = predict_nn(model, X_test_glove)

    # acc, p, r, f1 = evaluate(y_test_enc, y_pred)
    # results.append(("GloVe+NN", acc, p, r, f1, train_time, test_time))
    # ---------------- Visualization ----------------
    models = [r[0] for r in results]
    accuracy = [r[1] for r in results]
    train_times = [r[5] for r in results]
    test_times = [r[6] for r in results]
    plot_accuracy(models, accuracy)
    plot_speed(models, train_times)
    plot_test_speed(models, test_times)
    plot_tradeoff(train_times, accuracy, models)
    print("Results:")
    for r in results:
        print(f"{r[0]}: Acc={r[1]:.3f}, Prec={r[2]:.3f}, Rec={r[3]:.3f}, F1={r[4]:.3f}, TrainTime={r[5]:.3f}s, TestTime={r[6]:.3f}s")
    print("Graphs saved in /report folder")

if __name__ == "__main__":
    main()