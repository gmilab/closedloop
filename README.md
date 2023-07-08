# closedloop
Scripts for closed-loop stimulation based on real-time prediction
of impending lapses in attention

Usage: Once you have installed required dependencies, the closed-loop
stimulation GUI is launched through the terminal with python GMICLES.py

This will launch the GUI. A participant-specific JSON file is then selected
from the dropbown box to select the relevant machine learning classifiers
used for real-time prediction.

The program runs in real-time while EEG data are streamed and is written to 
follow precise timing during an attentional flexiblity task (Oh et al., Brain
and Cognition, 2014) running in Presentation Software. However, the code 
can be modified to work with other tasks and software if the core concept 
for closed-loop brain stimulation remains the same.
