import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def learning_curve(histories):
    hist_df = pd.concat([pd.DataFrame(hist.history) for hist in histories], sort=True)
    hist_df.index = np.arange(1, len(hist_df)+1)
    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(16, 10))
    axs[0].plot(hist_df['val_top_3_accuracy'], lw=5, label='Validation Accuracy')
    axs[0].plot(hist_df['top_3_accuracy'], lw=5, label='Training Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].grid()
    axs[0].legend(loc=0)
    axs[1].plot(hist_df['val_loss'], lw=5, label='Validation Loss')
    axs[1].plot(hist_df['loss'], lw=5, label='Training Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].grid()
    axs[1].legend(loc=0)
    fig.savefig('hist.png', dpi=300)
    plt.show()