import sys
import random as rand

def file_parser(file):
	for line in file:
		items = line.strip().split("//")

		if items[0]:
			yield items[0].strip()

def load_data(scanner):
	feature_names = []
	feature_values = []
	label_values = []

	num_features = int(next(scanner))

	for i in range(num_features):
		next_feature = next(scanner).split('-')
		feature_names += [next_feature[0].strip()]
		feature_values += [[x.strip() for x in next_feature[1].split()]]

	# Assume binary labels
	label_values += [next(scanner)]
	label_values += [next(scanner)]

	num_examples = int(next(scanner))

	return num_features, num_examples, feature_names, feature_values, label_values

# return a list of examples [(label, features1, ...), ...]
def load_example(scanner, num_examples):
	examples = []
	for i in range(num_examples):
		next_example = next(scanner).split()
		examples += [[x.strip() for x in next_example[1:]]]

	return examples

# python simple_perceptron.py trainset testset
def get_args():
	if not len(sys.argv) == 4:
		print("Usage: python simple_perceptron.py <trainset> <tuneset> <testset>")
		sys.exit()

	return sys.argv[1], sys.argv[2], sys.argv[3]

# Assumes that label is a boolean
class Perceptron:
	def __init__(self, num_features, feature_names, feature_values, label_values, learning_rate=0.1):
		self.num_features = num_features
		self.feature_names = {name: idx for idx, name in enumerate(feature_names)}
		self.feature_values = [{name: idx for idx, name in enumerate(values)} for values in feature_values]
		self.label_values = {name: idx for idx, name in enumerate(label_values)}
		
		self.weights = [rand.random() for i in range(num_features + 1)]
		self.bias, self.threshold = 1, 0
		self.learning_rate = learning_rate

	def labels_to_numbers(self, labels):
		return [self.label_values[label] for label in labels]

	def features_to_numbers(self, features):
		return [[self.feature_values[idx][feature] for idx, feature in enumerate(row)]for row in features]

	# train_set is a list of lists with labels and features [[label, feature1, ...], ...]
	def train(self, train_set):
		labels = self.labels_to_numbers([row[0] for row in train_set])
		features = self.features_to_numbers([row[1:] for row in train_set])

		for idx, row in enumerate(features):
			outcome = self.predict(row)
			diff = self.learning_rate * (labels[idx] - outcome)

			self.weights = [w + diff * x for x, w in zip([self.bias] + row, self.weights)]

	def predict(self, feature):
		inp = sum([x * w for x, w in zip([self.bias] + feature, self.weights)])
		
		if inp >= self.threshold:
			return 1
		else:
			return 0

	# test_set is a list of lists with labels and features [[label, feature1, ...], ...]
	def test(self, test_set):
		labels = self.labels_to_numbers([row[0] for row in test_set])
		features = self.features_to_numbers([row[1:] for row in test_set])

		num_cases = len(test_set)
		num_correct = sum([1 for idx, row in enumerate(features) if labels[idx] == self.predict(row)])

		return num_correct / num_cases

def main():
	trainset_name, tuneset_name, testset_name = get_args()

	with open(trainset_name, 'r') as file:
		scanner = file_parser(file)
		num_features, num_train_examples, feature_names, feature_values, label_values = load_data(scanner)
		train_set = load_example(scanner, num_train_examples)

	with open(tuneset_name, 'r') as file:
		scanner = file_parser(file)
		_, num_tune_examples, _, _, _ = load_data(scanner)
		tune_set = load_example(scanner, num_tune_examples)

	with open(testset_name, 'r') as file:
		scanner = file_parser(file)
		_, num_test_examples, _, _, _ = load_data(scanner)
		test_set = load_example(scanner, num_test_examples)

	perceptrons = [Perceptron(num_features, feature_names, feature_values, label_values, learning_rate=0.1) for i in range(10)]

	perceptron = perceptrons[0]

	for i in range(10):
		init_acc = perceptrons[i].test(tune_set)
		loop = 0
		patience = 5

		while loop < patience:
			loop += 1
			perceptrons[i].train(train_set)

			new_acc = perceptrons[i].test(tune_set)

			if new_acc > init_acc:
				init_acc = new_acc
				loop = 0

		if perceptrons[i].test(tune_set) > perceptron.test(tune_set):
			perceptron = perceptrons[i]

	print(perceptron.test(test_set))

main()