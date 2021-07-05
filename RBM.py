import numpy as np
from Para import *
import os

import concurrent.futures


class RBM:
    def __init__(self, num_visible, num_hidden, learning_rate):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.learning_rate = learning_rate
        np_rng = np.random.RandomState(1234)
        self.weights = np.asarray(np_rng.uniform(
            low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            size=(num_visible, num_hidden)))
        self.vbias = np.zeros(num_visible)
        self.hbias = np.zeros(num_hidden)

    def logistic(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sample_hidden(self, visible_prob):
        hidden_activation = np.dot(visible_prob, self.weights) + self.hbias
        hidden_prob = self.logistic(hidden_activation)
        return hidden_prob

    def sample_visible(self, hidden_prob):
        visible_activation = np.dot(hidden_prob, self.weights.T) + self.vbias
        visible_prob = self.logistic(visible_activation)
        return visible_prob

    def Contrastive_Divergence(self, dataset, k):
        num_examples = dataset.shape[0]
        positive_hidden_prob = self.sample_hidden(dataset)
        positive_hidden_activations = positive_hidden_prob > np.random.rand(num_examples, self.num_hidden)
        positive_association = np.dot(dataset.T, positive_hidden_prob)
        hidden_activations = positive_hidden_activations
        for steps in range(k):
            visible_prob = self.sample_visible(hidden_activations)
            visible_state = visible_prob > np.random.rand(num_examples, self.num_visible)
            hidden_prob = self.sample_hidden(visible_state)
            hidden_activations = hidden_prob > np.random.rand(num_examples, self.num_hidden)
        negative_visible_state = visible_state
        negative_hidden_prob = hidden_prob
        negative_association = np.dot(negative_visible_state.T, negative_hidden_prob)
        self.weights += (positive_association - negative_association) * self.learning_rate / num_examples
        self.vbias += (np.sum(dataset - negative_visible_state, axis=0)) * self.learning_rate / num_examples
        self.hbias += (np.sum(positive_hidden_prob - negative_hidden_prob, axis=0)) * self.learning_rate / num_examples
        error = np.sum((dataset - negative_visible_state) ** 2)
        return error

    def sampleset(self, num_sample):
        samples = np.ones((num_sample, self.num_visible))
        samples[0, :] = np.random.rand(self.num_visible)
        for i in range(1, num_sample):
            visible = samples[i - 1, :]
            hidden_prob = self.sample_hidden(visible)
            hidden_states = hidden_prob > np.random.rand(self.num_hidden)
            visible_prob = self.sample_visible(hidden_states)
            visible_states = visible_prob > np.random.rand(self.num_visible)
            samples[i, :] = visible_states
        return samples[1:, 0:]


# def train_and_sample(T, nH, nS, me, lr, output_data, output_parameters):
#     nH_name = 'nH = ' + str(nH)
#     load_path = os.path.join('Data', 'Training Data')
#     save_data_path = os.path.join('Data', output_data, nH_name)
#     save_weight_path = os.path.join('Data', output_parameters, 'Weights', nH_name)
#     save_error_path = os.path.join('Data', output_parameters, 'Errors', nH_name)
#     file_name = 'T = ' + format(T, '.2f') + '.npy'
#     completeLoad = os.path.join(load_path, file_name)
#     samples = (np.load(completeLoad) + 1) / 2  # convert to 0, 1
#     sz, N, N1 = samples.shape
#     samples_flat = np.reshape(samples, (sz, N * N1))
#     r = RBM(num_visible=64, num_hidden=nH)
#     G = r.train(samples_flat, max_epochs=me, learning_rate=lr)
#     print("Wights at T = " + format(T, '.2f') + ": ", r.weights)
#     RBM_data_flat = r.daydream(nS) * 2 - 1  # convert back to -1, 1
#     RBM_data = np.reshape(RBM_data_flat, (nS, N, N1))
#     completeSaveData = os.path.join(save_data_path, file_name)
#     np.save(completeSaveData, RBM_data)
#     completeSaveWeight = os.path.join(save_weight_path, file_name)
#     np.save(completeSaveWeight, r.weights)
#     completeSaveError = os.path.join(save_error_path, file_name)
#     np.save(completeSaveError, G)


def mini_and_sample(T, nH, nS, me, lr, output_data, output_parameters, k):
    nH_name = 'nH = ' + str(nH)
    load_path = os.path.join('Data', 'Training Data')
    save_data_path = os.path.join('Data', output_data, nH_name)
    save_weight_path = os.path.join('Data', output_parameters, 'Weights', nH_name)
    save_bias_path = os.path.join('Data', output_parameters, 'bias', nH_name)
    save_error_path = os.path.join('Data', output_parameters, 'Errors', nH_name)
    file_name = 'T = ' + format(T, '.2f') + '.npy'
    completeLoad = os.path.join(load_path, file_name)
    samples = (np.load(completeLoad) + 1) / 2  # convert to 0, 1
    sz, N, N1 = samples.shape
    samples_flat = np.reshape(samples, (sz, N * N1))
    dataset = np.reshape(samples_flat, (1000, 50, 64))
    r = RBM(num_visible=64, num_hidden=nH, learning_rate=lr)
    clo_er = []
    for epoch in range(me):
        error = 0
        for batch in range(1000):
            error += r.Contrastive_Divergence(dataset[batch], k=k)
        clo_er.append([epoch, error])
    print("Wights at T = " + format(T, '.2f') + ": ", r.weights)
    RBM_data_flat = r.sampleset(nS) * 2 - 1  # convert back to -1, 1
    RBM_data = np.reshape(RBM_data_flat, (nS - 1, N, N1))
    completeSaveData = os.path.join(save_data_path, file_name)
    np.save(completeSaveData, RBM_data)
    completeSaveWeight = os.path.join(save_weight_path, file_name)
    np.save(completeSaveWeight, r.weights)
    B = [r.vbias, r.hbias]
    completeSavebias = os.path.join(save_bias_path, file_name)
    np.save(completeSavebias, B)
    completeSaveError = os.path.join(save_error_path, file_name)
    np.save(completeSaveError, clo_er)


# def experi(T, lr):
#     load_path = os.path.join('Data', 'Training Data')
#     save_error_path = os.path.join('Data', 'Experiments')
#     file_name = 'T = ' + format(T, '.2f') + '.npy'
#     completeLoad = os.path.join(load_path, file_name)
#     samples = (np.load(completeLoad) + 1) / 2  # convert to 0, 1
#     sz, N, N1 = samples.shape
#     samples_flat = np.reshape(samples, (sz, N * N1))
#     r = RBM(num_visible=64, num_hidden=64)
#     G = r.train(samples_flat, max_epochs=100, learning_rate=lr)
#     out_name = 'T = ' + format(T, '.2f') + ', lr = ' + str(lr) + '.npy'
#     completeSaveError = os.path.join(save_error_path, out_name)
#     np.save(completeSaveError, G)


def sample_16(tem):
    output = 'RBM Generated Data'
    op = 'RBM Parameters'
    mini_and_sample(tem, 16, ns, me, lr, output, op, k)


def sample_4(tem):
    output = 'RBM Generated Data'
    op = 'RBM Parameters'
    mini_and_sample(tem, 4, ns, me, lr, output, op, k)


def sample_64(tem):
    output = 'RBM Generated Data'
    op = 'RBM Parameters'
    mini_and_sample(tem, 64, ns, me, lr, output, op, k)


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        s_64 = [executor.submit(sample_64, T) for T in T_range]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        s_4 = [executor.submit(sample_4, T) for T in T_range]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        s_16 = [executor.submit(sample_16, T) for T in T_range]
