import numpy as np
def sigmoid_rampup(current_epoch, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current_epoch, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))
