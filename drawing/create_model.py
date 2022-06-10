from tensorflow.keras.metrics import top_k_categorical_accuracy, categorical_crossentropy
from drawing.hyperparameter import IMG_SIZE, NUM_CLASS
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import Adam


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_1_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)

def create_model():
    model = MobileNet(input_shape=(IMG_SIZE, IMG_SIZE, 1), alpha=1, weights=None, classes=NUM_CLASS)
    model.compile(optimizer=Adam(learning_rate=0.002), loss='categorical_crossentropy', metrics=[categorical_crossentropy, top_3_accuracy, top_1_accuracy])
    return model

