import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    rows = np.arange(len(y_true))
    correct_confidents = y_pred[rows,y_true]

    loss = -np.mean(np.log(correct_confidents))
    return loss
    pass