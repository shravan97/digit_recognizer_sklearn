
# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score
import numpy as np

# The digits dataset
def digit_rec(inp):
	digits = datasets.load_digits()
	# The data that we are interested in is made of 8x8 images of digits, let's
	# have a look at the first 3 images, stored in the `images` attribute of the
	# dataset.  If we were working from image files, we could load them using
	# pylab.imread.  Note that each image must have the same size. For these
	# images, we know which digit they represent: it is given in the 'target' of
	# the dataset.# images_and_labels = list(zip(digits.images, digits.target))
	# To apply a classifier on this data, we need to flatten the image, to# turn the data in a (samples, feature) matrix:
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	# Create a classifier: a support vector classifier
	classifier = svm.SVC(gamma=0.001)
	# We learn the digits on the first half of the digits
	classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])
	expected = digits.target[n_samples / 2:]
	predicted = classifier.predict(data[n_samples / 2:])
	# images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
	req_inp = np.array([inp])
	predicted_frm_inp = classifier.predict(req_inp)
	return predicted_frm_inp