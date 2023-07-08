'''

EEGNet-based deep learning classifier prediction script for iEEG Real-Time Closed-Loop Stim
Nebras Warsi & Simeon Wong
Ibrahim Lab
July 2021


'''

# Qt framework
from PyQt5.QtCore import pyqtSignal, QRunnable, QObject

# General imports
import tensorflow as tf
from tensorflow.keras.models import load_model
import pathos
import numpy as np
import os
import json
import datetime
from enum import IntEnum

# DSP
from scipy.signal import welch
from scipy.integrate import simpson

# Sets mode for the broadband EEG vs. PSD
mode = 'psd' # or 'bb'

def funcgen(Fs):

    def get_data(signal):

        freqs, dataPSD = welch(signal, fs=Fs, nperseg=500, nfft=Fs)
        freq_range = np.where((freqs >= 4) & (freqs <= 43))[0]

        dataPSD = 100 * (dataPSD[:, freq_range] / simpson(dataPSD[:, freq_range])[:, None])

        return dataPSD

    return get_data

class Intent(IntEnum):
    NO_STIM = 0
    STIM = 1
    SUPPRESSED = 2

class MLRunnerSignal(QObject):
    ''' QObject helper so MLRunner can send Qt signals '''
    done = pyqtSignal(Intent)


class MLRunner(QRunnable):

    def __init__(self, pool, data, channels, Fs, model, logfile):
        ''' Initialize the runnable.

        Parameters
        ----------
        data : np.ndarray
            EEG data that needs to go into the ML model

        Fs : float
            EEG data sampling rate

        '''
        super(MLRunner, self).__init__()

        self.data = data
        self.channels = channels
        self.Fs = Fs
        self.pool = pool
        self.model = model
        self.logfile = logfile
        self.signals = MLRunnerSignal()


    def run(self):
        ''' Run the model computations. Should emit done signal with a bool as a parameter. '''
       
        # Predicts trial reaction time for intracranial stimulation
        # Fast is 0, 1 is slow

        # def some functions
        get_data = funcgen(self.Fs)
        signal = np.asarray(get_data(self.data[:len(self.channels), :]))

        # Predict attention based on ML model and save output
        out = self.model.predict(np.expand_dims(np.expand_dims(signal, axis=-1), axis=0))
        pred = out > 0.5 # Get model prediction
        # pred = out < 0.5 # REVERSE model prediction for MISMATCHED STIM

        # Randomize 50% of the predicted slow trials to receive no stim (control)
        # This is the within-subject randomization for the experiment
        should_stim = pred * np.random.randint(low=0, high=2)

        # don't stim on half the trials
        if pred == 0:
            intent = Intent.NO_STIM

        elif should_stim > 0:
            intent = Intent.STIM

        else:
            intent = Intent.SUPPRESSED

        outputstr = "%.2f - %s - [%s]" % (out, pred[0],  str(intent).split('.')[-1])
        print(outputstr)

        # Write to log file
        self.logfile.write('%s\t%.5f\t%d\t%s' %
                           (datetime.datetime.now().isoformat(timespec='microseconds'), pred, should_stim, str(intent).split('.')[-1] + '\n'))


        # Output trigger
        self.signals.done.emit(intent)


class MLModel():
    model = None

    def __init__(self, json_path, Fs):
        ''' 
        Load patient-specific data 
        '''

        try:
            with open(json_path, 'r') as datafile:
                self.info = json.load(datafile)
                print(self.info)
            self.path = self.info.get('path')
            self.model_dir = self.info.get('ml_model')
            self.channels = self.info.get('contacts')

        except:

            print('Error! Please check file path')
            exit()

        # Load EEGNet model for deployment
        self.model = load_model(self.model_dir, compile=False)

        ### Initialize parameters ###
        # EEG Parameters
        self.Fs = Fs

        ### Initialize persistent Process pool ###
        mp = pathos.helpers.mp
        self.pool = mp.Pool(5)  # brainzapper3000 has 6 cores, so we'll use up to 5.
        self.logfile = open(os.path.join(os.curdir, 'output', 'ml_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')), 'a')

    def close(self):
        self.pool.close()

    def __exit__(self):
        self.pool.close()

    def get_runner(self, data):
        return MLRunner(self.pool, data, self.channels, self.Fs, self.model, self.logfile)