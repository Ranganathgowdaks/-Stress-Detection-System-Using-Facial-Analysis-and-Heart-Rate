# /src/evaluation/evaluate.py

from sklearn.metrics import classification_report

def evaluate_model(model, test_gen):
    loss, accuracy = model.evaluate(test_gen)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    y_pred = model.predict(test_gen)
    y_true = test_gen.classes

    print(classification_report(y_true, y_pred.argmax(axis=1)))
