'''
Simeon Wong
Ibrahim Lab
Hospital for Sick Children

Based on the CURRY MATLAB Interface.
'''

import struct
import array
import numpy as np
import re
import traceback
from enum import IntEnum
from typing import Union

from PyQt5.QtNetwork import QTcpSocket
from PyQt5.QtCore import pyqtSignal, QObject


class RequestType(IntEnum):
    VERSION = 1
    CHANNEL_INFO = 3
    BASIC_INFO = 6
    STREAMING_START = 8
    STREAMING_STOP = 9


class ControlCode(IntEnum):
    FROM_SERVER = 1
    FROM_CLIENT = 2


class InfoType(IntEnum):
    NOT_INFO = 0
    VERSION = 1
    BASIC_INFO = 2
    CHANNEL_INFO = 4


class DataType(IntEnum):
    INFO = 1
    EEG = 2
    EVENTS = 3
    IMPEDANCE = 4


class BlockType(IntEnum):
    # this type is not used in practice because the data is always float32
    FLOAT_32BIT = 1
    FLOAT_32BITZIP = 2
    EVENT_LIST = 3


class ConnectionState(IntEnum):
    DISCONNECTED = 1
    INIT_CONNECTING = 2
    INIT_BASIC = 3
    INIT_CHANNEL = 4
    READY = 5
    STREAMING = 6


class QCurryInterface(QObject):
    #################################
    # Decode / Encode packet header #
    #################################

    # connection state
    state = ConnectionState.DISCONNECTED

    # packet header
    phdr = None

    # basic info
    basic_info = {}
    info_list = []

    # Signals
    initialized = pyqtSignal()
    dataReceived = pyqtSignal(int, np.ndarray)
    eventReceived = pyqtSignal(object)

    # String cleaner
    strclean = re.compile('[^\w ]+')

    #####################
    # Functions n stuff #
    #####################
    def __init__(self, debug: bool = False, dump_data_path: Union[str, None] = None):
        # initialize TCP connection
        self.con = QTcpSocket()
        super().__init__()

        self.debug = debug

        if dump_data_path is not None:
            self.dump_data_path = dump_data_path
            self.dump_data_file = open(self.dump_data_path, 'wb')
        else:
            self.dump_data_path = None
            self.dump_data_file = None

    def connectToHost(self, ip='127.0.0.1', port=4455):
        print('Connecting to socket...\n')
        self.con.connected.connect(self.connected_handler)
        self.con.connectToHost(ip, port)
        self.state = ConnectionState.INIT_CONNECTING

    def data_handler(self):
        try:
            while self.con.bytesAvailable() > 0:
                # if we didn't receive a header (yet), read the CURRY header
                if self.phdr is None:
                    # continue waiting for header if the header is not present
                    if self.con.bytesAvailable() < 20:
                        return False

                    # if entire header is present, read it
                    header = self.con.read(20)
                    if self.dump_data_file is not None:
                        self.dump_data_file.write(header)

                    assert len(header) > 0, 'Unable to read packet header'

                    # print(header)

                    # parse the header
                    self.phdr = {
                        'data_type':
                        DataType(int.from_bytes(header[4:6], byteorder='big')),
                        'sample_start':
                        int.from_bytes(header[8:12], byteorder='big',
                                    signed=False),
                        'size':
                        int.from_bytes(header[12:16], byteorder='big')
                    }

                    data_info = int.from_bytes(header[6:8], byteorder='big')

                    if self.phdr['data_type'] == DataType.INFO:
                        self.phdr['info_type'] = InfoType(data_info)
                    elif self.phdr['data_type'] == DataType.EEG:
                        self.phdr['block_type'] = BlockType(data_info)
                        assert self.phdr['block_type'] == BlockType.FLOAT_32BIT
                    elif self.phdr['data_type'] == DataType.EVENTS:
                        self.phdr['block_type'] = BlockType(data_info)
                        assert self.phdr['block_type'] == BlockType.EVENT_LIST

                    assert self.phdr['size'] < 64000  #invalid size

                # if (not elif): because the packet might include both header and data all at once
                if self.phdr is not None:
                    if self.con.bytesAvailable() < self.phdr['size']:
                        return False

                    # if entire packet is present, read it
                    data = self.con.read(self.phdr['size'])
                    if self.dump_data_file is not None:
                        self.dump_data_file.write(data)


                    # based on the current status, do something
                    if self.phdr['data_type'] == DataType.INFO:
                        if self.phdr['info_type'] == InfoType.BASIC_INFO:
                            self.process_basic_info(data)

                            # getting basic info usually happens right at the beginning of connection
                            # once we've got the basic info, request channel info
                            if self.state == ConnectionState.INIT_BASIC:
                                self.state = ConnectionState.INIT_CHANNEL
                                self.send_ctrl_header(RequestType.CHANNEL_INFO)

                        elif self.phdr['info_type'] == InfoType.CHANNEL_INFO:
                            self.process_channel_info(data)

                            # getting basic info usually happens right at the beginning of connection
                            # once we've got the basic info, we're ready to request data streaming start
                            if self.state == ConnectionState.INIT_CHANNEL:
                                self.state = ConnectionState.READY
                                self.initialized.emit()  # signal that we're ready

                    elif self.phdr['data_type'] == DataType.EEG:
                        self.process_eeg(self.phdr['sample_start'],
                                        data)  # hand the data off to eeg function

                    elif self.phdr['data_type'] == DataType.EVENTS:
                        self.process_event(data)

                    else:
                        # other option is IMPEDANCE, which is not yet implemented
                        pass

                    # reset states
                    self.phdr = None

        except:
            if self.phdr and isinstance(self.phdr,
                                        dict) and self.phdr['sample_start']:
                lastsample = 'Last sample: {:d}\n'.format(
                    self.phdr['sample_start'])
            else:
                lastsample = ''
            print('>>>>> except in QCurryInterface.data_handler >>>>>\n' +
                  traceback.format_exc() + '\n' + lastsample +
                  '<<<<< /except <<<<<\n\n')

            # reset header
            self.phdr = None

            # flush until next packet
            bufdata = self.con.peek(64000).decode('ascii')
            nextpacket = bufdata.find('DATA')

            if nextpacket == -1:  # if no header in the buffer, clear the buffer
                self.con.readAll()
            else:  # otherwise clear only the corrupted data until next packet
                self.con.read(nextpacket)

        # if no errors, but we're at this point, it means that we're just waiting for data.
        return True

    def connected_handler(self):
        # once we're connected, retrieve basic info
        self.con.connected.disconnect()  # disconnect the connected signal
        self.con.readyRead.connect(
            self.data_handler)  # connect the data available signal

        # request basic info
        self.send_ctrl_header(RequestType.BASIC_INFO)
        self.state = ConnectionState.INIT_BASIC

    def start_streaming(self):
        if self.state == ConnectionState.STREAMING:
            return False
        self.send_ctrl_header(RequestType.STREAMING_START)
        self.state = ConnectionState.STREAMING

    def stop_streaming(self):
        if self.state != ConnectionState.STREAMING:
            return False
        self.send_ctrl_header(RequestType.STREAMING_STOP)
        self.con.readAll()  # clear the rest of the read buffer
        self.state = ConnectionState.READY

    def process_basic_info(self, data):
        """ Handle incoming basic configuration info from the Curry streaming server

        Parameters
        ----------
        data: array of bytes

        Sets
        ----
        basic_info : dict
            Contains information about basic configuration parameters

            size : int
            eegChan : int
            number of EEG channels
            sampleRate : int
            sampling rate in Hz
            dataSize : int
            bytes per record / data point / sample

        Returns
        -------
        None

        """

        # decode response
        basic_info = {
            'size': int.from_bytes(data[0:4], byteorder='little'),
            'eegChan': int.from_bytes(data[4:8], byteorder='little'),
            'sampleRate': int.from_bytes(data[8:12], byteorder='little'),
            'dataSize': int.from_bytes(data[12:16], byteorder='little')
        }

        # do some sanity checks
        assert (basic_info['eegChan'] > 0 and basic_info['eegChan'] < 300
                ), 'Invalid number of EEG channels: %d' % basic_info['eegChan']
        assert (basic_info['sampleRate'] >
                0), 'Invalid sampling rate: %d' % basic_info['sampleRate']
        assert (basic_info['dataSize'] == 2 or basic_info['dataSize'] == 4
                ), 'Invalid number of bytes per data point: %d' % basic_info[
                    'dataSize']

        # save into object
        self.basic_info = basic_info

    def process_channel_info(self, data):
        """ Request information about currently active EEG channels as part of the connection initialization.

        Parameters
        ----------
        None

        Returns
        -------
        info_list : array of dict
            One dict per channel containing channel information.

            id : int
            chanLabel : string
            chanType : int
            deviceType : int
            eegGroup : int
            posX : double
            posY : double
            posZ : double
            posStatus : int
            bipolarRef :int
            addScale :  int
            isDropDown : int
        """

        # response decoding data
        chanInfoLen = 136

        offset_chanLabel = 4
        offset_chanType = offset_chanLabel + 80
        offset_deviceType = offset_chanType + 4
        offset_eegGroup = offset_deviceType + 4
        offset_posX = offset_eegGroup + 4
        offset_posY = offset_posX + 8
        offset_posZ = offset_posY + 8
        offset_posStatus = offset_posZ + 8
        offset_bipolarRef = offset_posStatus + 4
        offset_addScale = offset_bipolarRef + 4
        offset_isDropDown = offset_addScale + 4

        # loop through data and decode
        info_list = []
        for kk in range(self.basic_info['eegChan']):
            block_offset = kk * chanInfoLen
            info_list.append({
                'id':
                int.from_bytes(data[block_offset:block_offset +
                                    offset_chanLabel],
                               byteorder='little'),
                'chanLabel':
                self.strclean.sub(
                    '',
                    str(
                        data[block_offset + offset_chanLabel:block_offset +
                             offset_chanType], 'utf-16')),
                'chanType':
                int.from_bytes(
                    data[block_offset + offset_chanType:block_offset +
                         offset_deviceType],
                    byteorder='little'),
                'deviceType':
                int.from_bytes(
                    data[block_offset + offset_deviceType:block_offset +
                         offset_eegGroup],
                    byteorder='little'),
                'eegGroup':
                int.from_bytes(
                    data[block_offset + offset_eegGroup:block_offset +
                         offset_posX],
                    byteorder='little'),
                'posX':
                struct.unpack(
                    '<d', data[block_offset + offset_posX:block_offset +
                               offset_posY]),
                'posY':
                struct.unpack(
                    '<d', data[block_offset + offset_posY:block_offset +
                               offset_posZ]),
                'posZ':
                struct.unpack(
                    '<d', data[block_offset + offset_posZ:block_offset +
                               offset_posStatus]),
                'posStatus':
                int.from_bytes(
                    data[block_offset + offset_posStatus:block_offset +
                         offset_bipolarRef],
                    byteorder='little'),
                'bipolarRef':
                int.from_bytes(
                    data[block_offset + offset_bipolarRef:block_offset +
                         offset_addScale],
                    byteorder='little'),
                'addScale':
                int.from_bytes(
                    data[block_offset + offset_addScale:block_offset +
                         offset_isDropDown],
                    byteorder='little'),
                'isDropDown':
                int.from_bytes(
                    data[block_offset + offset_isDropDown:block_offset +
                         chanInfoLen],
                    byteorder='little')
            })

        self.info_list = info_list

    def process_eeg(self, sample_start, data):
        numSamples = int(
            len(data) /
            (self.basic_info['dataSize'] * self.basic_info['eegChan']))

        if self.basic_info['dataSize'] == 2:
            # data are integers
            data = array.array('h', data)
            data = np.array(data)
            data = np.reshape(data, (self.basic_info['eegChan'], numSamples),
                              order='F')

        elif self.basic_info['dataSize'] == 4:
            # data are floats
            # data = struct.unpack('<f', data)
            data = array.array('f', data)
            data = np.array(data)
            data = np.reshape(data, (self.basic_info['eegChan'], numSamples),
                              order='F')

        # emit the received signal
        self.dataReceived.emit(sample_start, data)

    def process_event(self, data):
        ''' Parse event data packet.
        
        From CURRY's <Packets.h>:
            struct NetStreamingEvent
            {
                long	nEventType;
                long	nEventLatency;
                long	nEventStart;
                long	nEventEnd;
                wchar_t	wcEventAnnotation[260];
            };
        '''
        names = ['type', 'latency', 'start', 'end']
        output = dict(zip(names, struct.unpack('<llll', data[0:16])))
        try:
            output['annotation'] = self.strclean.sub('',
                                                     str(data[16:], 'utf-8'))
        except:
            output['annotation'] = ''

        # emit the event signal
        self.eventReceived.emit(output)

        # log event to console in debug mode
        if self.debug:
            print('>> Event Received: {:d} | {:d} | {:d} | {:d} | {:s}'.format(
                output['type'], output['latency'], output['start'],
                output['end'], output['annotation']))

    def send_ctrl_header(self, request_type: RequestType):
        try:
            header = self.init_header(chanID='CTRL',
                                      code=ControlCode.FROM_CLIENT,
                                      request=request_type,
                                      samples=0,
                                      sizeBody=0,
                                      sizeUn=0)
            self.con.write(header)
        except Exception as ex:
            print(traceback.format_exc())

    def init_header(self, chanID, code, request, samples, sizeBody, sizeUn):
        """Assemble a binary header.

        Parameters
        ----------
        chanID : int
        code : int
        request : int
        samples : int
        sizeBody : int
        sizeUn : int

        Returns
        -------
        array of bytes
        binary header
        """
        # header format is unsigned big-endian
        assert (len(chanID) == 4
                ), 'chanID length needs to be 4. It is %d.' % len(chanID)

        return bytes(chanID, 'ascii') + \
            int(code).to_bytes(2, byteorder='big', signed=False) + \
            int(request).to_bytes(2, byteorder='big', signed=False) + \
            int(samples).to_bytes(4, byteorder='big', signed=False) + \
            int(sizeBody).to_bytes(4, byteorder='big', signed=False) + \
            int(sizeUn).to_bytes(4, byteorder='big', signed=False)
