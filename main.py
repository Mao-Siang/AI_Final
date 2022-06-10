from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from drawing.valid_data import create_validation_set
from drawing.datagen import training_data_generator
from sklearn.preprocessing import LabelEncoder
from drawing.create_model import create_model
from drawing.curve import learning_curve
from drawing.hyperparameter import *
import matplotlib.pyplot as plt
from drawing import labels


def main():

    model = create_model()

    wordEncoder = LabelEncoder()
    wordEncoder.fit(labels)

    train_datagen = training_data_generator(batch_size = BATCH_SIZE, img_size = IMG_SIZE, wordEncoder=wordEncoder, num_class=340, time_color=False)

    x, y = next(train_datagen)
    n = 8
    fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
    for i in range(n**2):
        ax = axs[i // n, i % n]
        (-x[i]+1)/2
        ax.imshow((-x[i, :, :, 0] + 1)/2, cmap=plt.cm.gray)
        ax.axis('off')
    plt.tight_layout()
    fig.savefig('gs.png', dpi=300)
    # plt.show()


    x_valid, y_valid = create_validation_set(wordEncoder, NUM_CLASS, IMG_SIZE, 6, False)
    print(x_valid.shape, y_valid.shape) 

    callbacks = [
             ReduceLROnPlateau(monitor="val_top_3_accuracy", factor = 0.75, patience = 2, min_delta=0.001,mode='max', min_lr=1e-5, verbose=1),
             ModelCheckpoint('model.h5', monitor='val_top_3_accuracy', mode='max', save_best_only=True,
                    save_weights_only=True),
    ]

    histories = []
    history = model.fit(train_datagen, epochs = EPOCHS, steps_per_epoch=STEP_PER_EPOCH, verbose=1, validation_data=(x_valid, y_valid), callbacks=callbacks)
    histories.append(history)
    print(histories)

    learning_curve(histories)


if __name__ == '__main__':
    main()
