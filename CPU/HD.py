import numpy as np
import sys
from copy import deepcopy

DEBUG = 1 
class MinMaxTracker:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def update_min_max(self, value):
        """
        Updates the minimum and maximum values based on the new value provided.

        Parameters:
        value: The new value to be added to the series for min/max evaluation.
        """
        if self.min_val is None or value < self.min_val:
            self.min_val = value
        if self.max_val is None or value > self.max_val:
            self.max_val = value

    def get_min_max(self):
        """
        Returns the minimum and maximum values of the series.

        Returns:
        tuple: A tuple containing the minimum and maximum values.
        """
        return self.min_val, self.max_val

tracker1 = MinMaxTracker()
tracker2 = MinMaxTracker()
tracker3 = MinMaxTracker()

def binarize(base_matrix):
	breakpoint()
	return np.where(base_matrix < 0, -1, 1)

"""
Random Projection Encoding: This function encodes input data (`X_data`) using random projections (RP) by multiplying the data with a base matrix. If `signed` is True, the resulting hyperdimensional vectors (HV) are binarized using the `binarize` function.
"""
def encoding_rp(X_data, base_matrix, signed=False):
	enc_hvs = []
	for i in range(len(X_data)):
		if i % int(len(X_data)/20) == 0:
			sys.stdout.write(str(int(i/len(X_data)*100)) + '% ')
			sys.stdout.flush()
		hv = np.matmul(base_matrix, X_data[i])
		if signed:
			hv = binarize(hv)
		enc_hvs.append(hv)
	return enc_hvs


"""
Level-ID Encoding: It encodes input data based on levels (quantization) and IDs, a more complex form of encoding that creates HVs by summing up level-specific HVs multiplied by ID-specific HVs, representing both the magnitude and identity of features.
"""
def encoding_idlv(X_data, lvl_hvs, id_hvs, D, bin_len, x_min, L=64):
	enc_hvs = []
	for i in range(len(X_data)):
		if i == int(len(X_data)/1):
			break
		if i % int(len(X_data)/20) == 0:
			sys.stdout.write(str(int(i/len(X_data)*100)) + '% ')
			sys.stdout.flush()
		sum_ = np.array([0] * D)
		for j in range(len(X_data[i])):
			bin_ = min( np.round((X_data[i][j] - x_min)/bin_len), L-1)
			bin_ = int(bin_)
			sum_ += lvl_hvs[bin_]*id_hvs[j]
		enc_hvs.append(sum_)
	return enc_hvs

"""
Permutation Encoding: Similar to level-ID encoding but uses permutation (rolling) of level HVs based on the feature index, which encodes both the feature's value and its position.
"""
def encoding_perm(X_data, lvl_hvs, D, bin_len, x_min, L=64):
	enc_hvs = []
	for i in range(len(X_data)):
		if i % int(len(X_data)/20) == 0:
			sys.stdout.write(str(int(i/len(X_data)*100)) + '% ')
			sys.stdout.flush()
		sum_ = np.array([0] * D)
		for j in range(len(X_data[i])):
			bin_ = min( np.round((X_data[i][j] - x_min)/bin_len), L-1)
			bin_ = int(bin_)
			sum_ += np.roll(lvl_hvs[bin_], j)
		enc_hvs.append(sum_)
	return enc_hvs

"""
Maximum Match Function: This function identifies the class of an encoded HV by computing the score (dot product normalized by class HV norms) with class-specific HVs and returning the class with the highest score.
"""
def max_match(class_hvs, enc_hv, class_norms):
		max_score = -np.inf
		max_index = -1
		for i in range(len(class_hvs)):
			score = np.matmul(class_hvs[i], enc_hv) / class_norms[i]
			
			if DEBUG: 
				tracker1.update_min_max(min(class_hvs[i]))
				tracker1.update_min_max(max(class_hvs[i]))
				tracker2.update_min_max(min(enc_hv))
				tracker2.update_min_max(max(enc_hv))
				tracker3.update_min_max(score)

			#score = np.matmul(class_hvs[i], enc_hv)
			if score > max_score:
				max_score = score
				max_index = i
		return max_index

"""
Training Function: It initializes the model by encoding training and validation data using one of the selected encoding schemes (`rp`, `rp-sign`, `idlv`, `perm`). It then trains a classifier by adjusting class-specific HVs based on training data and retraining epochs, evaluating performance on validation data to select the best model configuration. The function supports adjusting the dimensions of HVs (`D`), learning rate (`lr`), and other parameters.
"""
def train(X_train, y_train, X_test, y_test, D=500, alg='rp', epoch=20, lr=1.0, L=64):
	
	"""
	Data Preparation: Initially, the training data is shuffled to ensure randomness. A portion of the training data (20%) is separated out as validation data to evaluate the model's performance and avoid overfitting.
	"""
	np.random.seed(0)
	permvar = np.arange(0, len(X_train))
	np.random.shuffle(permvar)
	X_train = [X_train[i] for i in permvar]
	y_train = [y_train[i] for i in permvar]
	cnt_vld = int(0.2 * len(X_train))
	X_validation = X_train[0:cnt_vld]
	y_validation = y_train[0:cnt_vld]
	X_train = X_train[cnt_vld:]
	y_train = y_train[cnt_vld:]

	"""
	Encoding: The function supports three encoding schemes: random projection (rp, rp-sign), level-ID (idlv), and permutation (perm). Each encoding strategy represents input data as high-dimensional vectors (HDVs) in a unique way:

	Random Projection: Generates a base matrix with binary values to project the input data into a high-dimensional space, optionally applying binarization.

	Level-ID: Quantizes feature values into levels and combines them with identifier vectors to encode both the magnitude and identity of features.

	Permutation: Similar to level-ID but uses a permutation (rolling) of vectors based on the feature index to encode positional information.
	"""
	if alg in ['rp', 'rp-sign']:
		#create base matrix
		base_matrix = np.random.rand(D, len(X_train[0]))
		base_matrix = np.where(base_matrix > 0.5, 1, -1)
		base_matrix = np.array(base_matrix, np.int8)
		print('\nEncoding ' + str(len(X_train)) + ' train data')
		train_enc_hvs = encoding_rp(X_train, base_matrix, signed=(alg == 'rp-sign'))
		print('\n\nEncoding ' + str(len(X_validation)) + ' validation data')
		validation_enc_hvs = encoding_rp(X_validation, base_matrix, signed=(alg == 'rp-sign'))
	
	elif alg in ['idlv', 'perm']:
		#create level matrix
		lvl_hvs = []
		temp = [-1]*int(D/2) + [1]*int(D/2)
		np.random.shuffle(temp)
		lvl_hvs.append(temp)
		change_list = np.arange(0, D)
		np.random.shuffle(change_list)
		cnt_toChange = int(D/2 / (L-1))
		for i in range(1, L):
			temp = np.array(lvl_hvs[i-1])
			temp[change_list[(i-1)*cnt_toChange : i*cnt_toChange]] = -temp[change_list[(i-1)*cnt_toChange : i*cnt_toChange]]
			lvl_hvs.append(list(temp))
		lvl_hvs = np.array(lvl_hvs, dtype=np.int8)
		x_min = min( np.min(X_train), np.min(X_validation) )
		x_max = max( np.max(X_train), np.max(X_validation) )
		bin_len = (x_max - x_min)/float(L)
		
		#need to create id hypervectors if encoding is level-id
		if alg == 'idlv':
			cnt_id = len(X_train[0])
			id_hvs = []
			for i in range(cnt_id):
				temp = [-1]*int(D/2) + [1]*int(D/2)
				np.random.shuffle(temp)
				id_hvs.append(temp)
			id_hvs = np.array(id_hvs, dtype=np.int8)
			print('\nEncoding ' + str(len(X_train)) + ' train data')
			train_enc_hvs = encoding_idlv(X_train, lvl_hvs, id_hvs, D, bin_len, x_min, L)
			print('\n\nEncoding ' + str(len(X_validation)) + ' validation data')
			validation_enc_hvs = encoding_idlv(X_validation, lvl_hvs, id_hvs, D, bin_len, x_min, L)
		elif alg == 'perm':
			print('\nEncoding ' + str(len(X_train)) + ' train data')
			train_enc_hvs = encoding_perm(X_train, lvl_hvs, D, bin_len, x_min, L)
			print('\n\nEncoding ' + str(len(X_validation)) + ' validation data')
			validation_enc_hvs = encoding_perm(X_validation, lvl_hvs, D, bin_len, x_min, L)
	
		
	"""
	Training Process: The training process involves initializing class HVs, performing an initial training to create class-specific HVs by summing encoded HVs of samples belonging to each class, followed by retraining epochs where the model is fine-tuned by shuffling data and adjusting class HVs based on mispredictions.
	"""
	class_hvs = [[0.] * D] * (max(y_train) + 1)
	for i in range(len(train_enc_hvs)):
		class_hvs[y_train[i]] += train_enc_hvs[i]
	class_norms = [np.linalg.norm(hv) for hv in class_hvs]
	class_hvs_best = deepcopy(class_hvs)
	class_norms_best = deepcopy(class_norms)

	"""
	Retraining: Through specified epochs, the model is fine-tuned by adjusting the class-specific HDVs based on the learning rate and prediction accuracy. During retraining, the training data is reshuffled, and the model's performance is evaluated on the validation set. The best-performing model configuration on the validation set is retained for testing.
	"""	
	#retraining
	if epoch > 0:
		acc_max = -np.inf
		print('\n\n' + str(epoch) + ' retraining epochs')
		for i in range(epoch):
			sys.stdout.write('epoch ' + str(i) + ': ')
			sys.stdout.flush()
			#shuffle data during retraining
			pickList = np.arange(0, len(train_enc_hvs))
			np.random.shuffle(pickList)
			for j in pickList:
				predict = max_match(class_hvs, train_enc_hvs[j], class_norms)
				if predict != y_train[j]:
					class_hvs[predict] -= np.multiply(lr, train_enc_hvs[j])
					class_hvs[y_train[j]] += np.multiply(lr, train_enc_hvs[j])
			class_norms = [np.linalg.norm(hv) for hv in class_hvs]
			correct = 0
			for j in range(len(validation_enc_hvs)):
				predict = max_match(class_hvs, validation_enc_hvs[j], class_norms)
				if predict == y_validation[j]:
					correct += 1
			acc = float(correct)/len(validation_enc_hvs)
			sys.stdout.write("%.4f " %acc)
			sys.stdout.flush()
			if i > 0 and i%5 == 0:
				print('')
			if acc > acc_max:
				acc_max = acc
				class_hvs_best = deepcopy(class_hvs)
				class_norms_best = deepcopy(class_norms)
	
	del X_train
	del X_validation
	del train_enc_hvs
	del validation_enc_hvs

	print('\n\nEncoding ' + str(len(X_test)) + ' test data')
	if alg == 'rp' or alg == 'rp-sign':
		test_enc_hvs = encoding_rp(X_test, base_matrix, signed=(alg == 'rp-sign'))
	elif alg == 'idlv':
		test_enc_hvs = encoding_idlv(X_test, lvl_hvs, id_hvs, D, bin_len, x_min, L)
	elif alg == 'perm':
			test_enc_hvs = encoding_perm(X_test, lvl_hvs, D, bin_len, x_min, L)
	
	"""
	Testing Process: Encodes the test data using the chosen scheme and evaluates the performance of the best class HVs obtained during training, calculating accuracy as the proportion of correctly predicted samples.
    
	"""
	correct = 0
	for i in range(len(test_enc_hvs)):
		predict = max_match(class_hvs_best, test_enc_hvs[i], class_norms_best)
		if predict == y_test[i]:
			correct += 1
	acc = float(correct)/len(test_enc_hvs)

	print("tracker1:", tracker1.get_min_max())
	print("tracker2:", tracker2.get_min_max())
	print("tracker3:", tracker3.get_min_max())
	return acc
