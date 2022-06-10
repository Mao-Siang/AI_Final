from keras.utils.np_utils import to_categorical
from drawing.hyperparameter import NUM_CLASS


def one_hot_encoder(word, wordEncoder):
    return to_categorical(wordEncoder.transform([word]), num_classes=NUM_CLASS).reshape((NUM_CLASS,1))

# test_y = oneHotEncoder('The Eiffel Tower')
# print(test_y)
# b = [5]
# print(to_categorical(b, 6).reshape(6))
# print(wordEncoder.transform(['arm', 'asparagus', 'axe']))