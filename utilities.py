
import wfdb
import numpy as np


def load_ecg_signal(fname):

    try:  # ensure the file name given exists

        # load the ecg signal
        record = wfdb.rdrecord(fname)
        # load annotation
        annotation = wfdb.rdann(fname, 'atr')

    except:
        return [], [], [],False

    # asserting sampling frequency
    assert record.fs == 360, 'sample frequency is not 360Hz'

    # extract signal
    p_signal = record.p_signal

    # samples and labels
    labels = annotation.symbol
    samples = annotation.sample

    return p_signal, labels, samples, True
