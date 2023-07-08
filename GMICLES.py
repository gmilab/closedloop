# Qt Framework
import sys
import os
import os.path
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer, Qt, QThreadPool, QPoint
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QStaticText
from PyQt5.QtWidgets import QWidget

from psychopy import parallel  # requires inpoutx64.dll in working dir + system32

# Plot stuff
import matplotlib, matplotlib.figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# Math stuff
import numpy as np

# EEG streaming interface
from QCurryInterface import QCurryInterface

# ML Code
from QStimModels import MLModel, Intent

# Misc
import datetime


class GMICLES(QtWidgets.QMainWindow):
    ''' Qt-based GUI for closed loop electrical stim.

    Buffers and plots incoming EEG data for QC, listens for intent triggers on pin 8 (value=128) of the Synamps RT, 
    and runs an ML model to dichotomize this trial as stim or nostim, provides visual feedback to guide clinician
    in pressing the stim trigger at the target time.
    
    '''
    STRINGS = {
        'C': 'Connecting...',
        'RDY': 'Not streaming',
        'W': 'Waiting for intent input...',
        'I': 'Intent received!',
        'P': 'Model running',
        'ST': 'Prepare to STIM!',
        'NS': 'No stim...',
        'SS': 'Stim suppressed...'
    }

    TRIG_VAL = int(1)  # trigger port value as int in range [0, 255]
    TRIG_DUR = 1

    ### Configuration ###
    time_window = (-2, 2)  # progress bar

    intent_time = -0.8  # from Presentation code, at what time (s) wrt. stim will it send the trigger
    intent_timer_timeout = 0.05  # how often to update progress bar
    intent_received_time = None  # storing the time trigger is received
    intent_sample = 0  # the sample number of the most recent intent marker

    analyze_at_time = 0  # the time (s) wrt. stim when data is sent for analysis
    analysis_interval = 2  # the timewindow up until the analyze_at_time that is submitted for analysis

    ### EEG data items ###
    data_buffer = None  # rolling eeg data buffer
    latest_sample_num = 0  # CURRY-provided sample index for most recent sample in the buffer
    fsample = None  # sampling rate

    ### Tracking variables for intent ###
    collecting_after_intent = False  # has an intent been received?
    samples_since_intent = 0  # how many samples received since the intent?

    ### ML runners ###
    mlmodel = None
    pool = QThreadPool.globalInstance()

    ### stim trains ###
    train_num = 1
    train_isi = 0
    train_curr_count = 0
    train_timer = QTimer()

    ### stim cue timer ###
    stimcue_timer = QTimer()

    ### Trigger logger ###
    trig_log_file = None

    def __init__(self):
        super().__init__()

        # load the UI
        uic.loadUi('GMICLES.ui', self)

        # initialize CURRY
        self.eeg = QCurryInterface()
        self.eeg.dataReceived.connect(self.eeg_data_received)
        self.eeg.initialized.connect(self.eeg_connected)
        self.eeg.eventReceived.connect(self.eeg_event_received)

        self.eeg.connectToHost()
        self.txtStatus.setText(self.STRINGS['C'])

        # bind buttons
        self.btnStart.clicked.connect(self.start_streaming)
        self.btnStart.setEnabled(False)

        self.btnStop.clicked.connect(self.stop_streaming)
        self.btnIntent.clicked.connect(self.intent_received)
        self.btnSelectModelFile.clicked.connect(self.select_model_file)

        self.btnSendStim.clicked.connect(self.start_stim_train)
        self.btnSendStim.setStyleSheet("background-color : yellow")

        # plots
        self.tsplot = TSPlot(self.timeseries)

        self.prevplot = TriggerPlot(self.prevTargetBar)
        self.prevplot.minValue = self.time_window[0]
        self.prevplot.maxValue = self.time_window[1]

        self.currplot = TriggerPlot(self.targetBar)
        self.currplot.minValue = self.time_window[0]
        self.currplot.maxValue = self.time_window[1]
        self.set_target_time()

        # intent timer
        self.intent_timer = QTimer(self)
        self.intent_timer.setInterval(round(self.intent_timer_timeout * 1000))
        self.intent_timer.timeout.connect(self.move_bar)

        self.txtTargetTime.editingFinished.connect(self.set_target_time)

        # stim trains
        self.train_timer.setTimerType(Qt.PreciseTimer)
        self.train_timer.timeout.connect(self.stim_train_timeout)

        self.txt_train_isi.editingFinished.connect(self.update_stim_train)
        self.txt_train_num.editingFinished.connect(self.update_stim_train)

        self.stimcue_timer.setTimerType(Qt.PreciseTimer)
        self.stimcue_timer.setSingleShot(True)
        self.stimcue_timer.timeout.connect(self.start_stim_train)

        # parallel port
        self.port = parallel.ParallelPort(address=0x3ff8)

        # trig log file
        self.trig_log_file = open(
            os.path.join(os.curdir, 'output', 'trigs_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')), 'a')

        self.show()

    def start_streaming(self):
        ''' Initiate streaming when the button is pressed '''
        # tell CURRY NetStreamer to start sending data
        self.eeg.start_streaming()

        # # create reporting timer
        # self.reporting_timer = QTimer(self)
        # self.reporting_timer.timeout.connect(self.reporting_func)
        # self.reporting_timer.start(1000)

        # update status at the top of the window
        self.txtStatus.setText(self.STRINGS['W'])

    def plot_eeg_data(self):
        ''' Called when plot refresh timer times out. Tell the MPL subclass to update data and redraw plot. '''
        self.tsplot.update_data(self.data_buffer)

    def stop_streaming(self):
        ''' Called when stop streaming button is pressed. '''
        # Tell CURRY NetStreamer to stop sending data
        self.eeg.stop_streaming()

        # Stop refreshing the plot
        self.plot_timer.stop()

    def eeg_data_received(self, sample_start: int, data: np.ndarray):
        ''' Callback for dataReceived signal. 
        Add data to circular buffer, check if the data should be sent for analysis. 
        '''
        nsamples_received = data.shape[1]
        nsamples_remain = self.data_buffer.shape[1] - nsamples_received

        # store sample number of ending sample
        self.latest_sample_num = sample_start + nsamples_received

        # roll the array
        self.data_buffer[:, :nsamples_remain] = self.data_buffer[:, -nsamples_remain:]

        # add new data
        self.data_buffer[:, -nsamples_received:] = data

        # check if we're buffering data for analysis
        if self.collecting_after_intent:
            self.samples_since_intent += nsamples_received

            # if we've buffered enough
            if self.samples_since_intent >= self.samples_to_collect_after_intent:
                # reset data counter
                self.collecting_after_intent = False
                self.samples_since_intent = 0

                # send off for analysis!
                self.data_ready_for_analysis()

    def eeg_event_received(self, data):
        ''' Callback for the eventReceived signal.
        If this is the intent trigger, process it...
        '''
        elapsed_since_trigger = (self.latest_sample_num - data['start']) / self.fsample

        # if this is the intent trigger from presentation, do intent stuff
        if data['type'] == 128:
            self.intent_sample = data['start']
            self.intent_received(elapsed_since_trigger)
            print('Event received | %d - %s | %.4f secs ago' %
                  (data['type'], data['annotation'], elapsed_since_trigger))

        self.currplot.addTrigger((data['start'] - self.intent_sample) / self.fsample + self.intent_time, data['type'])

        self.trig_log_file.write('{:s}\t{:d}\t{:d}\n'.format(datetime.datetime.now().isoformat(timespec='microseconds'),
                                                             data['type'], data['start']))

    def eeg_connected(self):
        ''' Callback for eegConnected signal.
        Display active channel information, initialize timeseries plots, initialize buffers.
        '''

        # print channel information to the textbox
        info_list_as_text = ['%d - %s: %d' % (x['id'], x['chanLabel'], x['deviceType']) for x in self.eeg.info_list]
        info_list_as_text = '\n'.join(info_list_as_text)
        self.chInfo.setPlainText(info_list_as_text)

        # initialize circular buffer
        self.fsample = self.eeg.basic_info['sampleRate']
        nsamples = self.fsample * (self.time_window[1] - self.time_window[0])
        self.data_buffer = np.zeros((self.eeg.basic_info['eegChan'], nsamples))
        self.time_values = np.arange(-1 * nsamples, 0) / self.fsample

        self.samples_to_collect_after_intent = ((self.analyze_at_time - self.intent_time) * self.fsample)

        # initial plot
        # self.tsplot.data_to_show = [not ('Trigger' in x['chanLabel']) for x in self.eeg.info_list]
        self.tsplot.plot(self.time_values, self.data_buffer)

        # start a timer to refresh the plot
        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.plot_eeg_data)
        self.plot_timer.start(100)

        # update status at the top
        self.txtStatus.setText(self.STRINGS['RDY'])

    def intent_received(self, elapsed_since_trigger=0):
        # start a timer that moves the progress bar
        self.intent_timer.start()
        self.currplot.setStatus(0)

        self.intent_received_time = datetime.datetime.now() - datetime.timedelta(seconds=elapsed_since_trigger)
        self.currplot.value = self.intent_time + elapsed_since_trigger

        # update status at the top
        self.txtStatus.setText(self.STRINGS['I'])

        # reset counters
        self.samples_since_intent = 0
        self.collecting_after_intent = True

    def data_ready_for_analysis(self):
        # submit most recent block of data for analysis in separate thread
        if self.mlmodel is not None:
            task = self.mlmodel.get_runner(self.data_buffer[:, -1 * round(self.analysis_interval * self.fsample):])
            task.signals.done.connect(self.analysis_done)

            # run task using thread pool
            self.pool.start(task)

            # update status
            self.txtStatus.setText(self.STRINGS['P'])

    def analysis_done(self, stimgo):
        ''' Callback for analysis done signal. '''
        if stimgo == Intent.STIM:
            self.currplot.setStatus(2)
            self.txtStatus.setText(self.STRINGS['ST'])

            time_until_stim = max(
                self.currplot.targetTime -
                ((datetime.datetime.now() - self.intent_received_time).total_seconds() + self.intent_time), 0.001)

            self.stimcue_timer.setInterval(int(time_until_stim * 1000))
            self.stimcue_timer.start()

        elif stimgo == Intent.NO_STIM:
            self.currplot.setStatus(1)
            self.txtStatus.setText(self.STRINGS['NS'])

        elif stimgo == Intent.SUPPRESSED:
            self.currplot.setStatus(3)

    def update_stim_train(self):
        try:
            self.train_num = int(self.txt_train_num.text())
            self.train_isi = int(self.txt_train_isi.text())

            if self.train_isi > 0 and self.train_isi < (self.TRIG_DUR * 2):
                self.train_isi = self.TRIG_DUR * 2

            self.train_timer.setInterval(self.train_isi)

        except:
            # if parse failed, reset
            self.txt_train_num.setText("1")
            self.txt_train_isi.setText("0")
            self.train_num = 1
            self.train_isi = 0

    def start_stim_train(self):
        
        # setup repeating trains
        self.train_curr_count = 0
        self.train_timer.start()

        # send first stim immediately
        self.stim_train_timeout()

    def stim_train_timeout(self):
        if self.train_curr_count < self.train_num:
            # fire stim
            self.output_stim_trigger()
            self.train_curr_count = self.train_curr_count + 1
        else:
            self.train_timer.stop()
            self.train_curr_count = 0

    def output_stim_trigger(self):
        self.port.setData(self.TRIG_VAL)
        QTimer.singleShot(self.TRIG_DUR, self.output_reset)
        print('! STIM ' + datetime.datetime.now().isoformat(timespec='microseconds'))

    def output_reset(self):
        self.port.setData(int(0))

    def move_bar(self):
        self.currplot.value = (datetime.datetime.now() - self.intent_received_time).total_seconds() + self.intent_time
        self.currplot.update()

        # if we hit the end of the time window...
        if self.currplot.value >= self.time_window[1]:
            # stop bar motion
            self.txtStatus.setText(self.STRINGS['W'])
            self.intent_timer.stop()

            # copy current to previous
            self.prevplot.cloneState(self.currplot)

            # reset current
            self.currplot.reset()

    def set_target_time(self):
        ''' Callback for target time box edit. '''
        try:
            self.currplot.targetTime = float(self.txtTargetTime.text())
            self.currplot.update()
        except:
            self.txtTargetTime.setText("0")
            self.currplot.targetTime = 0

    def select_model_file(self):
        # cleanup existing
        if self.mlmodel is not None:
            self.mlmodel.close()

        # filepicker dialog
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Model descriptor file", "", "JSON (*.json)")
        if not os.path.exists(filepath):
            return

        self.txtModelFile.setText(filepath)

        # load the model
        self.mlmodel = MLModel(filepath, self.fsample)

        self.btnStart.setEnabled(True)

    def reporting_func(self):
        print('Buffer size: {:d}'.format(self.eeg.con.bytesAvailable()))


class TSPlot(FigureCanvasQTAgg):

    def __init__(self, parent=None, dpi=96):
        # initialize the figure
        self.fig = matplotlib.figure.Figure(figsize=(8.5, 2.3), dpi=96)
        self.axes = self.fig.add_subplot(111)
        self.axes.invert_yaxis()

        s = super(TSPlot, self)
        s.__init__(self.fig)
        self.setParent(parent)

        s.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        s.updateGeometry()

    def plot(self, time, data):
        self.plt = self.axes.plot(time, data.T)
        self.axes.set_ylim(-500, 500)
        self.draw()

    def update_data(self, data):
        for kk, ln in enumerate(self.plt):
            ln.set_ydata(data[kk, :])
        self.draw()


class TriggerPlot(QWidget):
    value = -100

    minValue = -2
    maxValue = 2

    targetTime = 0

    t0pen = None
    target_pen = None

    barcolor = None

    clr = []  # UNK, NOGO, GO

    status = 0

    # list of triggers
    triggers = []

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedSize(parent.width(), parent.height())

        # setup pens for drawing targets
        self.t0pen = QPen()
        self.t0pen.setColor(QColor(0, 0, 0))
        self.t0pen.setStyle(Qt.DotLine)

        self.target_pen = QPen()
        self.t0pen.setColor(QColor(0, 0, 0))
        self.target_pen.setWidth(2)

        self.clr = [QColor(182, 182, 224), QColor(181, 58, 58), QColor(0, 181, 0), QColor(252, 186, 3)]

    def getX(self, val):
        return int(round((val - self.minValue) / (self.maxValue - self.minValue) * self.width()))

    def setStatus(self, status):
        self.status = status
        self.update()

    def paintEvent(self, event):
        p = QPainter()
        p.begin(self)

        # progress bar
        p.setPen(QPen(Qt.NoPen))  # no pen
        p.setBrush(self.clr[self.status])
        p.drawRect(0, 0, self.getX(self.value), self.height())

        # t = 0
        p.setBrush(QBrush(Qt.NoBrush))  # no brush
        p.setPen(self.t0pen)

        for t in [-1.0, -0.5, 0, 0.5, 1.0]:
            xv = self.getX(t)
            p.drawLine(xv, 0, xv, self.height())

        # target
        p.setPen(self.target_pen)
        xv = self.getX(self.targetTime)
        p.drawLine(xv, 0, xv, self.height())

        # triggers
        yv = round(self.height() / 2)
        for t, v in self.triggers:
            xv = self.getX(t)
            p.drawStaticText(QPoint(xv, yv), QStaticText('%d' % v))

        p.end()

    def addTrigger(self, time, value):
        self.triggers.append((time, value))
        self.update()

    def reset(self):
        self.value = self.minValue
        self.setStatus(0)
        self.triggers = []
        self.update()

    def cloneState(self, src):
        self.value = src.value
        self.targetTime = src.targetTime
        self.setStatus(src.status)
        self.triggers = src.triggers
        self.update()


if __name__ == "__main__":
    App = QtWidgets.QApplication(sys.argv)
    window = GMICLES()
    sys.exit(App.exec())