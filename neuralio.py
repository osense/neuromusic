import sys
import numpy
from scipy.io import wavfile
import os
numpy.set_printoptions(threshold=numpy.inf)

def average(array):
	return sum(array)/float(len(array))

SAMPLE_LENGTH = 10000	# size of one sample as an input for the neural network
MAXIMUM = 1000.0		# number used to normalize data

##########################################################################################################

"""
Function eatWAV consumes an WAV file and produces pair of arrays of numpy arrays

input_wav:: relative path to a WAV file
"""
def eatWAV(input_wav):
	rate, data = wavfile.read(os.path.abspath(input_wav))
	length = len(data)

	data = numpy.divide(data[:,0], numpy.array([MAXIMUM]) )
	inputs, outputs = [], []

	data_prev = data[0:SAMPLE_LENGTH-1]
	for i in range(1, int(length/SAMPLE_LENGTH)):
		data_new = data[i*SAMPLE_LENGTH:(i+1)*SAMPLE_LENGTH - 1]
		inputs.append(numpy.array(data_new))
		outputs.append(numpy.array(data_prev))
		data_prev = data_new

	return rate, tuple([inputs, outputs])

"""
Function vomitWAV creates a WAV file from given parameters.

name:: given string representing output file name (Example: out.wav)
rate:: the sample rate (in samples/sec).
data:: a 1-D or 2-D numpy array of either integer or float data-type
"""
def vomitWAV(name, rate, data):
	wavfile.write(name, rate, data)
