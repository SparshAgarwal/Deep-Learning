#!/usr/bin/env python3
import librosa
import wave
import pyaudio
import csv
import argparse
import math
import numpy as np

# Stores an audio file for easy playback
class Audio():
	def __init__(self, filename):
		self.filename = filename
		self.chunk = 1024
		
		with wave.open(filename, 'rb') as wf:
			self.rate = wf.getframerate()
			self.total_sample_size = wf.getnframes()
	
	# Return a floating point time series of the audio file for the given duration, return the full time series if duration is not specified
	def time_series(self, duration=-1):
		points = librosa.load(self.filename, sr=self.rate)[0]
		
		if duration == -1:
			return points
		else:
			return points[:self.get_sample_size(duration)]

	# Return a floating point time series of the audio file 
	def time_series(self):
		return librosa.load(self.filename, sr=self.rate)[0]
	
	# Return the sample size for the given duration, return the maximum length of the sample size if duration exceed the sample's duration
	def get_sample_size(self, duration):
		return min(self.rate * duration, self.total_sample_size)
	
	# Play the audio file for the given duration, play the whole file if duration is not given
	def play(self, duration=-1):
		wf = wave.open(self.filename, 'rb')
		p = pyaudio.PyAudio()
		
		stream = p.open(format =
				p.get_format_from_width(wf.getsampwidth()),
				channels = wf.getnchannels(),
				rate = wf.getframerate(),
				output = True)
		
		if duration == -1:
			data = wf.readframes(self.chunk)
		else:
			d_chunk = self.get_sample_size(duration)
			data = wf.readframes(d_chunk)
			stream.write(data)

		while len(data) > 0:
			stream.write(data)
			data = wf.readframes(self.chunk)
		
		stream.close()
		p.terminate()
		wf.close()

def convert_audio_to_example(audio, label, n_fft=2048, hop_length=512):
	D = librosa.stft(audio.time_series(), n_fft=n_fft, hop_length=hop_length)
	D = np.transpose(D)
	
	D = np.insert(D, D.shape[1], label, 1)
	return D

def main():
	parser = argparse.ArgumentParser(description=("Convert wavfile to examples for machine learning.\n"
		"Currently splits sample into 1s chunk of magnitude of frequencies\nAn example is a list of "
		"magnitude of frequencies ending with a label."))

	parser.add_argument('filename', metavar='F', type=str, help='Filename of wav file')
	parser.add_argument('label', metavar='L', type=int, help='Label of this example, must be an integer, example: 0')
	parser.add_argument('-nfft', metavar="N", type=int, default=2048, help='n_fft window size for the fft, default to 2048')
	parser.add_argument('-hoplen', metavar="H", type=int, default=-1, help='hop length of fft algorithm, default to 1s of the wavfile')
	parser.add_argument('-outputfilename', metavar="O", type=str, default='out.csv', help='Name of the output file, default to out.csv')

	args = parser.parse_args()
	
	audio_file = Audio(args.filename)

	if math.log2(args.nfft) % 1 != 0:
		print("Error: nfft argument must be a power of 2")
		return

	if args.hoplen == -1:
		args.hoplen = audio_file.rate

	# Get FFT result (list of complex numbers)
	examples = convert_audio_to_example(audio_file, args.label, args.nfft, args.hoplen)

	# Convert to magnitude
	examples = np.abs(examples)

	# Write to csv
	with open(args.outputfilename, 'w') as csv_file:
		csv_writer = csv.writer(csv_file, delimiter=' ')

		for ex in examples:
			csv_writer.writerow(ex)

if __name__ == '__main__':
	main()

