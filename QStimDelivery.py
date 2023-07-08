from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, pyqtSlot

class QStimDelivery(QThread):
    def __init__(self, sig_start:pyqtSlot, sig_update:pyqtSlot):
        sig_start.connect(self.start_stim_train)
        sig_update.connect(self.update_stim_train)
    
    def update_stim_train(self):
        pass

    def start_stim_train(self):
        pass