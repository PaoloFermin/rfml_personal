#!/usr/bin/env python3

# Make sure I only use 1 GPU so someone else can use the other one
from __future__ import division
from tempfile import TemporaryFile
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

import math
import copy
from struct import pack, unpack

import glob
import numpy as np

from keras.models import load_model
from keras.utils import to_categorical
import random
import fixed_point as fp
#import parity_error_detection


#####################################################
# Collect and pre-process RF data (This is only for RF data. When we use other NNs (like VGG) we will use the ImageNet dataset.
#####################################################

def checkFileName(filename):
	if "bpsk" in filename:
		dictCheck = "bpsk"
	elif "qpsk" in filename:
		dictCheck = "qpsk"
	elif "qam16" in filename:
		dictCheck = "qam16"
	elif "qam64" in filename:
		dictCheck = "qam64"
	elif "vt" in filename:
		dictCheck = "vt"
	else:
		print("Error - There was no known mod scheme in the fileName")
	return dictCheck


def load_rf_data():
	# Get all files matching the description
	fileList = glob.glob("RF_data/*SNR10_0.dat")
	NUMOFFILES = len(fileList)

	if NUMOFFILES == 0:
		print("Error, no files found")

	# "Global" Vars
	NUMTRIALS = 10000
	SETSOFDATA = 0  # This var changes to keep track of label numbers too
	SAMPLESIZE = 1024
	# Dictionary will store the label (arbitrary) value for each mod scheme
	typesOfMods = {}

	# Check first file for mod scheme for labels
	dictName = checkFileName(fileList[0])
	if dictName not in typesOfMods:
		typesOfMods[dictName] = SETSOFDATA
		SETSOFDATA += 1

	# Intialize arrays
	rawData = np.fromfile(fileList[0], np.float32)

	rawData = rawData.astype(np.float16)

	rawDataLabs = typesOfMods[dictName] + np.zeros(NUMTRIALS)

	# Pack up all the data sets into one list
	for fileNum in range(1, len(fileList)):
		# Check for the mod scheme
		dictName = checkFileName(fileList[fileNum])
		if dictName not in typesOfMods:
			typesOfMods[dictName] = SETSOFDATA
			SETSOFDATA += 1

		newFile = np.fromfile(fileList[fileNum], np.float32)
		newFile = newFile.astype(np.float16)
		newFileLabs = typesOfMods[dictName] + np.zeros(NUMTRIALS)
		rawData = np.concatenate((rawData, newFile), axis=0)
		rawDataLabs = np.concatenate((rawDataLabs, newFileLabs), axis=0)

	# Reshape data into 2 channels, 1 row, 1024 samples in a trial, num Trials per set * number of file sets
	catsData = np.reshape(rawData, (2, 1, SAMPLESIZE, NUMTRIALS * NUMOFFILES), order='F')
	# Move data to be sorted as Samples, Row, Column, Channels
	overallData = np.transpose(catsData, (3, 1, 2, 0))
	overallDataLabs = to_categorical(rawDataLabs)

	# Shuffle Data
	currRandState = np.random.get_state()
	np.random.shuffle(overallData)
	np.random.set_state(currRandState)
	np.random.shuffle(overallDataLabs)

	return np.asarray(overallData[:10000]), np.asarray(overallDataLabs[:10000])


#######################################################################
# Flip bits and generate output files. This is where the experiment starts
#######################################################################

# Method that flips a bit in the weight
def flip_single_bit(targetWeight, targetBit):
	# Return a bytes object containing the value 'targetWeight' packed according the to 'f' packing format for 'floats'
	fs = pack('f', targetWeight)
	# unpack() the buffer 'fs' (presumably packed by pack()) according to the format 'BBBB' for 4 unsigned characters (integers in python) representing the four bytes in the byte and save them in the list byteList.
	byteList = list(unpack('BBBB', fs))
	# Choose the byte that contains the bit to be flipped and then find the
	# location of the bit in the byte.
	whichByte = int(math.floor(targetBit / 8))
	whichBit = targetBit % 8
	unluckyByte = byteList[whichByte]
	mask = 1 << whichBit
	unluckyByte = unluckyByte ^ mask
	byteList[whichByte] = unluckyByte
	fs = pack('BBBB', *byteList)
	# Unpack new weight as a tuple and save it in the weight matrix
	weightTuple = unpack('f', fs)
	pythonWeight = weightTuple[0]
	numpyWeight = np.array([pythonWeight]).astype(np.float32)
	return numpyWeight


# Method that determines which bit(s) to flip based user specifications. Called from choose_weights.
def choose_bits(targetWeight, startBit, endBit=None, random=False):
	changedWeight = None
	# random = true when we will choose a random bit from the range startBit - endBit
	if (random):
		if (endBit == None):
			print("Error: user specified to choose a random bit but did not give range of bits to randomly select from.")
		randomBit = np.random.randint(startBit, endBit + 1)
		changedWeight = flip_single_bit(targetWeight, randomBit)
	# random = false when we will choose a specific bit(s) to flip
	else:
		if (endBit == None):
			changedWeight = flip_single_bit(targetWeight, startBit)
		# flip all bits in the range
		else:
			for flipBit in range(startBit, endBit + 1):
				changedWeight = flip_single_bit(targetWeight, flipBit)
	return changedWeight


def choose_weights(weights, startBit, endBit, random, layer_num, bias_num, numberOfWeights, total_weights):
	print("bias_num: " + str(bias_num))
	print(type(bias_num))
	print("layer_num: " + str(layer_num))
	print(type(layer_num))
	if (layer_num == None and bias_num == None):
		which_layer = np.random.randint(0, len(weights) - 1)
		layer = weights[which_layer]
	if (layer_num == None and bias_num != None):
		which_layer = (bias_num * 2) - 1
		layer = weights[which_layer]
	if (layer_num != None and bias_num == None):
		which_layer = (layer_num * 2) - 2
		layer = weights[which_layer]
	if (layer_num != None and bias_num != None):
		rand = random.randit(0, 1)
		if (rand == 0):
			which_layer = (layer_num * 2) - 2
			layer = weights[which_layer]
		if (rand == 1):
			which_layer = (bias_num * 2) - 1
			layer = weights[which_layer]
	shape = np.asarray(np.shape(layer))

	if (layer_num != None or bias_num != None):
		new_num_broken_weights = int(standardize_broken_weights(numberOfWeights, total_weights, layer))
		if (new_num_broken_weights == 0 and numberOfWeights != 0):
			new_num_broken_weights = 1
	else:
		new_num_broken_weights = numberOfWeights

	for w in range(int(new_num_broken_weights)):
		if (len(shape) == 1):
			randi = np.random.randint(0, len(layer))
			target_weight = layer[randi]
			new_weight = choose_bits(target_weight, startBit, endBit, random)
			weights[which_layer][randi] = new_weight

		if (len(shape) > 1):
			randis = np.zeros(5)
			for i in range(len(shape)):
				randis[i] = np.random.randint(0, shape[i])
			# Could not find a layer in keras that returns a tuple greater than 5D
			a = int(randis[0])
			b = int(randis[1])
			c = int(randis[2])
			d = int(randis[3])
			e = int(randis[4])
			if (len(shape) == 2):
				target_weight = layer[a][b]
				new_weight = choose_bits(target_weight, startBit, endBit, random)
				weights[which_layer][a][b] = new_weight
			if (len(shape) == 3):
				target_weight = layer[a][b][c]
				new_weight = choose_bits(target_weight, startBit, endBit, random)
				weights[which_layer][a][b][c] = new_weight
			if (len(shape) == 4):
				target_weight = layer[a][b][c][d]
				new_weight = choose_bits(target_weight, startBit, endBit, random)
				weights[which_layer][a][b][c][d] = new_weight
			if (len(shape) == 5):
				target_weight = layer[a][b][c][d][e]
				new_weight = choose_bits(target_weight, startBit, endBit, random)
				weights[which_layer][a][b][c][d][e] = new_weight

	return weights


def standardize_broken_weights(num_broken_weights, num_total_weights, layer):
	num_layer_weights = layer.size
	ratio = num_broken_weights / num_total_weights
	return ratio * num_layer_weights


##############################
# Method to run experiments
##############################
# Currently set to run for 100 trials with set number of broken weights (SEUs).
# Must have cnn saved in directory. This cnn will be tested on RF data 100 FOR EACH SEU% (why this code takes so long).
# Parameters:
# startBit - the first possible bit that can be flipped.
# endBit - the last possible bit to be flipped (not inclusive, startBit = 0 endBit =12 will flip any bit from
# 0-11. endBit = None if user wants to flip one specific bit (indicated by the startBit)
# fileName - string of the name of the files the output will be saved to (ex. "case1_seu%_") the seu% for each
# output file will be added to the end of the name.
# random - boolean stating whether the user would like a randomly chosen, if the user specified a range.
# tmrStart and tmrEnd represent the range of bits the user wants to protect with tmr (must be 24-31). tmrStart = None
# if user does not want to implement tmr
# Output a separate .csv file for every seu% with the number of correct predictions each of the 100 CNNs predicted
# for that SEU
def rf_flip_bits(startBit, endBit, fileName, random, tmrStart, tmrEnd, layer=None, bias=None, RunParity=False, fixed_point = False):
	rf_inputs, rf_labels = load_rf_data()
	model = load_model('rf_cnn_float32.h5')
	old_weights = model.get_weights()
	total_weights = 0

	for i in range(0, len(old_weights)):
		current = old_weights[i]
		total_weights = total_weights + current.size
	print("Number of Weights in RF CNN: %d" % total_weights)
	num_broken_weights = []
	num_broken_weights.append(int(total_weights * 0))
	# These were removed because they did not break a single weight
	# num_broken_weights.append(int(total_weights * 0.000000001))
	# num_broken_weights.append(int(total_weights * 0.00000001))
	# num_broken_weights.append(int(total_weights * 0.0000001))
	# num_broken_weights.append(int(total_weights * 0.000001))
	num_broken_weights.append(int(total_weights * 0.00001))
	num_broken_weights.append(int(total_weights * 0.0001))
	# num_broken_weights.append(int(total_weights * 0.0002))
	# num_broken_weights.append(int(total_weights * 0.0003))
	# num_broken_weights.append(int(total_weights * 0.0004))
	# num_broken_weights.append(int(total_weights * 0.0005))
	# num_broken_weights.append(int(total_weights * 0.0006))
	# num_broken_weights.append(int(total_weights * 0.0007))
	# num_broken_weights.append(int(total_weights * 0.0008))
	# num_broken_weights.append(int(total_weights * 0.0009))
	num_broken_weights.append(int(total_weights * 0.001))
	num_broken_weights.append(int(total_weights * 0.01))
	num_broken_weights.append(int(total_weights * 0.1))
	num_broken_weights.append(int(total_weights * 0.5))
	num_broken_weights.append(int(total_weights * 1))

	print("Number of broken weights in each round: " + str(num_broken_weights))

	# Create confusion matrix and files to save at every number of broken
	# weights and over the entire experiment
	#confMatrix = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
	#globalConfMatrix = open(fileName + "confusion_matrix_individual", "w")
	#singleConfMatrix = open(fileName + "confusion_matrix", "w")
	if (RunParity == True):
		parity_file = open("parity_file.txt", "w")
	# Break the weights
	if (tmrStart != None):
		#print("In TMR")
		tmr_weights = copy.deepcopy(old_weights)
		for tmrBit in range(tmrStart, tmrEnd + 1):
			tmr_weights = tmr.set_tmr(tmr_weights, tmrBit)

	for numberOfWeights in num_broken_weights:
		results = []
		print("Number of broken weights this round: " + str(numberOfWeights))
		# This is where you can change the trial amount
		trials = 100
		for i in range(trials):

			new_weights = copy.deepcopy(old_weights)
			if (tmrStart != None):
				new_weights = tmr_weights

			if (RunParity):
				print("setting parity")
				new_weights = parity_error_detection.set_parity(new_weights)

			if (fixed_point):
				for l in new_weights:
					for weight in np.nditer(l, op_flags=['readwrite']):
						weight[...] = fp.to_floating_point(fp.to_fixed_point(weight))

			new_weights = choose_weights(new_weights, startBit, endBit, random, layer, bias, numberOfWeights, total_weights)

			if (tmrStart != None):
				for tmrBit in range(tmrStart, tmrEnd + 1):
					new_weights = tmr.check_tmr(new_weights, tmrBit)

			if (RunParity):
				print("checking parity")
				parity_result = parity_error_detection.run_parity_check(new_weights)
				parity_file.write("possible errors detected: " + str(parity_result))
				print("possible errors detected: " + str(parity_result))


			model.set_weights(new_weights)

			if (not RunParity):
				# Determine how many inputs are correctly identified and use to create confusion matrix
				scores = model.evaluate(rf_inputs, rf_labels, verbose=0)
				correctPredictions = scores[1] * 10000
				preds = model.predict_classes(rf_inputs)
				count = 0
				# check the predictions and calculate the total correct predictions
				"""while count < len(preds):
					# if actual was qam16
					if rf_labels[i][0] == 1:
						if preds[i] == 0:
							confMatrix[0][0] += 1
						if preds[i] == 1:
							confMatrix[0][1] += 1
						if preds[i] == 2:
							confMatrix[0][2] += 1
						if preds[i] == 3:
							confMatrix[0][3] += 1
					# if actual was qam64
					elif rf_labels[i][1] == 1:
						if preds[i] == 0:
							confMatrix[1][0] += 1
						if preds[i] == 1:
							confMatrix[1][1] += 1
						if preds[i] == 2:
							confMatrix[1][2] += 1
						if preds[i] == 3:
							confMatrix[1][3] += 1
					# if actual was qpsk
					elif rf_labels[i][2] == 1:
						if preds[i] == 0:
							confMatrix[2][0] += 1
						if preds[i] == 1:
							confMatrix[2][1] += 1
						if preds[i] == 2:
							confMatrix[2][2] += 1
						if preds[i] == 3:
							confMatrix[2][3] += 1
					# if actual was bpsk
					elif rf_labels[i][3] == 1:
						if preds[i] == 0:
							confMatrix[3][0] += 1
						if preds[i] == 1:
							confMatrix[3][1] += 1
						if preds[i] == 2:
							confMatrix[3][2] += 1
						if preds[i] == 3:
							confMatrix[3][3] += 1
					count += 1"""

				# Save to correctPredictions to final results
				results.append(correctPredictions)

				print("average accuracy for " + str(numberOfWeights) + " broken weights: " + str((sum(results) / float(len(results)) / 10000)))
				tempFileName = fileName + str(numberOfWeights) + '.csv'
				mathTime = np.asarray(results)
				np.savetxt(tempFileName, mathTime, delimiter=",")
				#singleConfMatrix.write(str(confMatrix))

	#globalConfMatrix.write(str(confMatrix))
