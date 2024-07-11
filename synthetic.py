import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, LogisticRegression

class GaussianModel():
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def get_samples(self, num_samples, y_label):
        x = RNG.normal(self.mu, self.sigma, num_samples)
        y = [y_label]*num_samples

        return x, np.array(y)

class GaussianMixtureModel():
    def __init__(self, mu_arr, sigma_arr):
        self.mu_arr = mu_arr
        self.sigma_arr = sigma_arr

    def get_samples(self, num_samples, y_label):
        x_list = []
        for ite in range(len(self.mu_arr)):
            x_list.append(RNG.normal(self.mu_arr[ite], self.sigma_arr[ite], num_samples))

        x = []
        for ite in range(num_samples):
            elements_at_ite = [array[ite] for array in x_list]
            x.append(RNG.choice(elements_at_ite))

        x = np.array(x)
        y = [y_label]*num_samples

        return x, np.array(y)

RNG = np.random.default_rng(42)

NUM_SAMPLES_T = 1000
NUM_SAMPLES_F = 1000
NUM_SAMPLES_TEST = 5000

NUM_MODELS = 1000
RASHOMON_SET_TOLERANCE = 0.01

distrib1 = GaussianModel(mu=0., sigma=0.5)
distrib2 = GaussianModel(mu=2., sigma=0.5)
# distrib2 = GaussianMixtureModel(mu_arr=[1.5, 2.5], sigma_arr=[1., 1.])

## Training Data
tdata_x, tdata_y = distrib1.get_samples(NUM_SAMPLES_T, 1)
fdata_x, fdata_y = distrib2.get_samples(NUM_SAMPLES_F, 0)
train_x, train_y = np.concatenate((tdata_x, fdata_x), axis=0), np.concatenate((tdata_y, fdata_y), axis=0)
train_x = train_x.reshape(-1, 1)

## Testing Data
tdata_x, tdata_y = distrib1.get_samples(NUM_SAMPLES_TEST, 1)
fdata_x, fdata_y = distrib2.get_samples(NUM_SAMPLES_TEST, 0)
test_x, test_y = np.concatenate((tdata_x, fdata_x), axis=0), np.concatenate((tdata_y, fdata_y), axis=0)
test_x = test_x.reshape(-1, 1)

## Plot Testing Data
plt.hist(tdata_x, color='red', bins=100, alpha=0.3)
plt.hist(fdata_x, color='blue', bins=100, alpha=0.3)
plt.savefig('distrib.png')

## Train models and create Rashomon set
clf = LogisticRegression(random_state=0).fit(train_x, train_y)
base_acc = clf.score(test_x, test_y)

model_list = []
for ite in range(NUM_MODELS):
    clf = SGDClassifier(random_state=ite).fit(train_x, train_y)
    acc = clf.score(test_x, test_y)
    if acc > (base_acc - RASHOMON_SET_TOLERANCE):
        model_list.append(clf)

## Range of Decision Boundaries
cutoff_list = []
for clf in model_list:
    cutoff_list.append(-clf.intercept_[0]/clf.coef_[0,0]) ## Specifically for single feature setting
print("Max Cutoff: %.4f; Min Cutoff: %.4f" % (np.max(cutoff_list), np.min(cutoff_list)))
print("Overall range: %.4f" % (np.max(cutoff_list) - np.min(cutoff_list)))

## Calculate ambiguity
pred_list = []
for clf in model_list:
    pred_list.append(clf.predict(test_x))
pred_list = np.array(pred_list)

amb_bool = pred_list[0] == pred_list
amb = 1 - np.mean(np.all(amb_bool, axis=0))

print("Ambiguity on the Test Set: %.4f" % amb)