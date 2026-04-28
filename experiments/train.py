from src.data.loader import load_data
from src.data.preprocessing import preprocess
from src.models.cnn import build_model

X_train, X_test, y_train, y_test = load_data("data/raw/dataset.csv")
X_train, X_test = preprocess(X_train, X_test)

model = build_model(X_train.shape[1])
model.fit(X_train, y_train, epochs=5)

print("Training Done")
