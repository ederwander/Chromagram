#Author: ederwander
#http://ederwander.wordpress.com/2012/03/22/chromagram/
#Date: 11/02/2012

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal2
import math
import wave, pylab
import operator

#FFT Size
Nfft=2048

#Chroma centered on A5 = 880Hz 
A5=880

#Number of chroma bins
nbin=12

#Semitone
st=2**(1/float(nbin))

#Step
step=128

#Used for Downsample Signal
fr=11025

#DownSample Factor
df=4


tunechroma1=[np.log2(A5*st**i) for i in range(nbin)] 
tunechroma2=[int(np.log2(A5*st**i)) for i in range(nbin)]

chroma=np.asarray(tunechroma1)-np.asarray(tunechroma2);


spf = wave.open('Animal_cut.wav','r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
fs = spf.getframerate()

#Stereo to Mono
if spf.getnchannels() == 2:
	signal=signal[:,0]

plt.figure(1)
plt.title('Signal Wave...')
plt.plot(signal)

#Normalize
#signal = signal/signal.max();

#signal=signal2.resample(signal,int(round(len(signal)*fr)/float(fs)),window=None)

#downsample with low-pass filtering
#signal=signal2.resample(signal, len(signal)/df)

#Starting DownSample signal using Decimation
b = signal2.firwin(30, 1.0/df)
signal = signal2.lfilter(b, 1., signal)
signal = signal.swapaxes(0,-1)[30+1::4].swapaxes(0,-1)

plt.figure(2)

plt.title('Downsampled Signal...')
plt.plot(signal)

fs = fs / df

step=Nfft-step

Pxx, freqs, bins, im  = pylab.specgram(signal,Fs=fs,window=np.hamming(Nfft),NFFT=Nfft,noverlap=step,Fc=0)

freqs = freqs[1:,]

freqschroma=np.asarray(np.log2(freqs)) - np.asarray([int(np.log2(f)) for f in freqs])

nchroma=len(chroma)
nfreqschroma=len(freqschroma)

CD=np.zeros((nfreqschroma, nchroma))

for i in range(nchroma):
	CD[:,i] = np.abs(freqschroma - chroma[i])

FlipMatrix=np.flipud(CD)

min_index = []
min_value = []

for i in reversed(range(FlipMatrix.shape[0])):
	index, value = min(enumerate(FlipMatrix[i]), key=operator.itemgetter(1))
	min_index.append(index)
	min_value.append(value)


#Numpy Array for Chrome Scale population
CS = np.zeros((len(chroma),Pxx.shape[1]))

Magnitude= np.log(abs(Pxx[1:,]))


for i in range(CS.shape[0]):

	#Find index value in min_index list
	a = [index for index,x in enumerate(min_index) if x == i]
	
	#Numpy Array for values in each index
	AIndex = np.zeros((len(a),Pxx.shape[1]))

	t=0;
	for value in a:
		AIndex[t,:] = Magnitude[value,:]
		t=t+1

	MeanMag=[]
	for M in AIndex.T:
		MeanMag.append(np.mean(M))

	CS[i,:] = MeanMag

#normalize the chromagram array
CS= CS / CS.max()


plt.figure(3)
plt.title('Original Magnitude')
plt.imshow(Magnitude.astype('float64'),interpolation='nearest',origin='lower',aspect='auto')

plt.figure(4)
plt.title('Chromagram')
plt.imshow(CS.astype('float64'),interpolation='nearest',origin='lower',aspect='auto')

plt.show()


