import numpy as np

CLASS_NAMES = ["100", "1000", "200", "250", "50", "500"]
CONF_THRESHOLD = 0.9

def decode_prediction(preds):
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))

    if conf < CONF_THRESHOLD:
        return "unknown", conf

    return CLASS_NAMES[idx], conf
