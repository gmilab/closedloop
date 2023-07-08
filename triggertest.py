import time
import datetime
from psychopy import parallel   # requires inpoutx64.dll in working dir + system32

port = parallel.ParallelPort(address=0x3ff8)

while True:
    port.setData(1)
    time.sleep(0.01)
    port.setData(0)
    print('Stim delivered at ' + datetime.datetime.now().isoformat())
    time.sleep(1)
