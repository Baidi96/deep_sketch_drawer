#modified based on https://github.com/coreyauger/quickdraw-gan
# and https://github.com/lc82111/Keras_HED
import os
import numpy as np
import cv2
import random

def read_file_list(filelist):

	pfile = open(filelist)
	filenames = pfile.readlines()
	pfile.close()

	filenames = [f.strip() for f in filenames]

	return filenames

def split_pair_names(filenames, base_dir):

	filenames = [c.split(' ') for c in filenames]
	filenames = [(os.path.join(base_dir, c[0]), os.path.join(base_dir, c[1])) for c in filenames]

	return filenames

class DataParser():
	def __init__(self, batch_size = 16, image_height = 256, image_width = 256, channels = 3, black_white = False, useTrainFile = True,  trainFileName = "newList.txt", validationFileName = 'newList.txt', producingEdgeMap = False):
		if useTrainFile:
			#uses a text file to record all the training/validation/test samples
			#allows the data to be loaded on the fly instead of storing it in memory

			#load training file
			#each line should be in the form of "input path" + space + "target1 path" + "\n"
			#example of line in file: "train/trainInput/image1.jpg train/trainTarget/target1.jpg\n"
			#path should be relative to the folder you run the engine file
			self.channels = channels
			self.train_val_dir = "./train_val/"
			#load paths of training examples (input and target)
			self.train_file = os.path.join(self.train_val_dir, trainFileName)
			self.training_pairs = read_file_list(self.train_file)
			#stores in format [(input1_path, targe1_path), (input2_path, targe2_path), .... ,(inputN_path, targetN_path)]
			#example: [("train/trainInput/image1.jpg", "train/trainTarget/target1.jpg")]
			#to access nth training input: self.train_samples[n - 1][0]
			#to access nth training target: self.train_samples[n - 1][1]
			self.train_samples = split_pair_names(self.training_pairs, self.train_val_dir)

			#load paths of validation examples, similar to training examples
			self.val_file = os.path.join(self.train_val_dir, validationFileName)
			self.val_pairs = read_file_list(self.train_file)
			self.val_samples = split_pair_names(self.val_pairs, self.train_val_dir)

			self.n_train_samples = len(self.training_pairs)
			self.training_ids = list(range(self.n_train_samples))
			self.n_val_samples = len(self.val_pairs)
			self.validation_ids =list(range(self.n_val_samples))
			np.random.shuffle(self.training_ids)
			np.random.shuffle(self.validation_ids)

			#cut off undersized batches
			self.batch_size_train = batch_size
			too_much = len(self.training_ids) % self.batch_size_train
			self.training_ids = self.training_ids[:-too_much]
			assert len(self.training_ids) % self.batch_size_train == 0
			self.steps_per_epoch = len(self.training_ids)/self.batch_size_train
			too_much = len(self.validation_ids) % (self.batch_size_train*2)
			self.validation_ids = self.validation_ids[:-too_much]
			assert len(self.validation_ids) % (self.batch_size_train*2) == 0
			self.validation_steps = len(self.validation_ids)//(self.batch_size_train*2)

		else:
			self.train_examples = read_all_examples()

		self.usingEdgeAsInput = black_white
		#set input image width and height
		#input images will be resized to the corresponding size
		self.image_width = image_width
		self.image_height = image_height
		self.channel = channels
		self.batch_size = batch_size
		self.target_regression = True


	def get_batch_with_ids(self, batch, training = True):
		"""
		batch: a list of ids representing the files
		"""
		filenames = []
		images = []
		edgemaps = []
		if training:
			self.samples = self.train_samples
		else:
			self.samples = self.val_samples
		for idx, b in enumerate(batch):
			if self.usingEdgeAsInput:
				#read as grey scale
				im = cv2.imread(self.samples[b][0], 0)
				#use cv2.INTER_AREA for resizing
				im = cv2.resize(im, (self.image_width, self.image_height),  interpolation = cv2.INTER_AREA )
			else:
				#read color input
				im = cv2.imread(self.samples[b][0])
				im = cv2.resize(im, (self.image_width, self.image_height))
			im = im / 255.0
			#target image should be binary, use area interpolation
			em = cv2.imread(self.samples[b][1])
			em = cv2.resize(em, (self.image_width, self.image_height), interpolation = cv2.INTER_AREA)
			#scale value to be either 0 or 1
			bin_em = em / 255.0
			images.append(im)
			edgemaps.append(bin_em)
			filenames.append(self.samples[b])
		images   = np.asarray(images)
		edgemaps = np.asarray(edgemaps)
		return images, edgemaps, filenames

	def get_data_for_GAN(self, sketchFolder, conditionFolder, conditionChannel = 1, augment = True):
		#reads and store all the data for GAN
		num_of_data = len(os.listdir(sketchFolder))
		if augment:
			num_of_data = num_of_data * 4
		data = np.zeros((num_of_data, 28, 28, 1))
		conditionImg = np.zeros((num_of_data, 28, 28, conditionChannel))
		count = 0
		allSketch = os.listdir("./" + sketchFolder)
		random.shuffle(allSketch)
		for i, file in enumerate(allSketch):
			sketch = cv2.imread("./" + sketchFolder +"/"+file, 0)
			prefix = file.split("-")[0]
			edge = prefix + ".jpg"
			if conditionChannel == 1:
				#read as binary
				conIm = cv2.imread("./" + conditionFolder +"/"+ edge, 0)
			else:
				conIm = cv2.imread("./" + conditionFolder +"/"+ edge)
			conIm = conIm/255.0
			temp = sketch/255.0
			data[count,:,:,:] = np.reshape(temp, (28, 28, 1))
			conditionImg[count, :, :, :] = np.reshape(conIm, (28, 28, conditionChannel))
			count += 1
			if augment:
				#rotate 90 degrees for three times to augment
				#1

				M = cv2.getRotationMatrix2D((14,14),90,1)
				temp = cv2.warpAffine(temp,M,(28,28))
				conIm =  cv2.warpAffine(conIm,M,(28,28))
				data[count,:,:, :] = np.reshape(temp, (28, 28, 1))
				conditionImg[count, :, :, :] = np.reshape(conIm, (28, 28, conditionChannel))
				count += 1
				#2
				temp = cv2.warpAffine(temp,M,(28,28))
				conIm =  cv2.warpAffine(conIm,M,(28,28))
				data[count,:,:, :] = np.reshape(temp, (28, 28, 1))
				conditionImg[count, :, :, :] = np.reshape(conIm, (28, 28, conditionChannel))
				count += 1
				#3
				temp = cv2.warpAffine(temp,M,(28,28))
				conIm =  cv2.warpAffine(conIm,M,(28,28))
				data[count,:,:, :] = np.reshape(temp, (28, 28, 1))
				conditionImg[count, :, :, :] = np.reshape(conIm, (28, 28, conditionChannel))
				count += 1
				#rotate back to original direction
				#temp = cv2.warpAffine(temp,M,(28,28))
				#conIm =  cv2.warpAffine(conIm,M,(28,28))
				#flip horizontally and vertically
				#flip
				#temp2 = cv2.flip(temp, 0)
				#conIm2 = cv2.flip(conIm, 0)
				#data[count,:,:, :] = np.reshape(temp2, (28, 28, 1))
				#conditionImg[count, :, :, :] = np.reshape(conIm2, (28, 28, conditionChannel))
				#count += 1
				#temp3 = cv2.flip(temp, 1)
				#conIm3 = cv2.flip(conIm, 1)
				#data[count,:,:, :] = np.reshape(temp3, (28, 28, 1))
				#conditionImg[count, :, :, :] = np.reshape(conIm3, (28, 28, conditionChannel))
				#count += 1


				#transpose and flip
				"""
				temp = cv2.transpose(temp)
				conIm =  cv2.transpose(conIm)
				data[count,:,:, :] = np.reshape(temp, (28, 28, 1))
				conditionImg[count, :, :, :] = np.reshape(conIm, (28, 28, conditionChannel))
				count += 1
				#2
				temp = cv2.transpose(temp)
				conIm =  cv2.transpose(conIm)
				#transpose back
				temp = cv2.flip(temp, 0)
				conIm = cv2.flip(conIm, 0)
				data[count,:,:, :] = np.reshape(temp, (28, 28, 1))
				conditionImg[count, :, :, :] = np.reshape(conIm, (28, 28, conditionChannel))
				count += 1
				temp = cv2.flip(temp, 1)
				conIm = cv2.flip(conIm, 0)
				data[count,:,:, :] = np.reshape(temp, (28, 28, 1))
				conditionImg[count, :, :, :] = np.reshape(conIm, (28, 28, conditionChannel))
				count += 1
				"""
		return data, conditionImg

		#def get_data_for_GAN_in_batch(self, sketchFolder, conditionFolder, conditionChannel = 1):




