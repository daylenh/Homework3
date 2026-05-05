import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras import Input

# traditional Models

def train_nb(X_train, y_train):
    start = time.time()
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model, time.time() - start

def train_svm(X_train, y_train):
    start = time.time()
    model = LinearSVC()
    model.fit(X_train, y_train)
    return model, time.time() - start

def train_rf(X_train, y_train):
    start = time.time()
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model, time.time() - start

def train_knn(X_train, y_train):
    start = time.time()
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model, time.time() - start

def train_dt(X_train, y_train):
    start = time.time()
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model, time.time() - start

def predict(model, X_test):
    start = time.time()
    y_pred = model.predict(X_test)
    return y_pred, time.time() - start

# neural network models
def train_nn(X_train, y_train):
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(set(y_train)), activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    start = time.time()
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model, time.time() - start

def predict_nn(model, X_test):
    start = time.time()
    y_pred = model.predict(X_test)
    y_pred = y_pred.argmax(axis=1)
    return y_pred, time.time() - start