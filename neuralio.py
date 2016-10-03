import numpy as np
from scipy.io import wavfile
import os
from net import *
np.set_printoptions(threshold=np.inf)

SAMPLE_LENGTH = Net.layers[0]	# size of one sample as an input for the neural network
#SAMPLE_LENGTH = 10000
RATIO = 5							# scale maximum

##########################################################################################################

"""
Function eatWAV consumes an WAV file and produces pair of np arrays of np arrays

input_wav:: relative path to a WAV file
"""
def eatWAV(input_wav):
	rate, data = wavfile.read(os.path.abspath(input_wav))

	length = (len(data)/SAMPLE_LENGTH)*SAMPLE_LENGTH
	data = data[0:length]

	data = np.average(data, axis=1)
	data = np.tanh(data/(max(data)/RATIO))
	inputs, outputs = [], []

	data_prev = data[0:SAMPLE_LENGTH]
	for i in range(1, int(length/SAMPLE_LENGTH)):
		data_new = data[i*SAMPLE_LENGTH:(i + 1)*SAMPLE_LENGTH]
		inputs.append(np.array(data_new))
		outputs.append(np.array(data_prev))
		data_prev = data_new

	return rate, tuple([np.array(inputs), np.array(outputs)])

"""
Function vomitWAV creates a WAV file from given parameters.

name:: given string representing output file name (Example: out.wav)
rate:: the sample rate (in samples/sec).
data:: a 1-D or 2-D np array of either integer or float data-type
"""
def vomitWAV(name, rate, data):
	wavfile.write(name, rate, data)

# stuff for testing eatWAV
#rate, (inputs, outputs) = eatWAV('WAVs/1.wav')
#print inputs[100]

# stuff fore testint vomitWAV
#rate, data = wavfile.read(os.path.abspath('WAVs/1.wav'))
#data = np.average(data, axis=1)
#data = np.tanh(data/(max(data)/RATIO))
#vomitWAV('test.wav', rate, data)