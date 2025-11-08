import numpy as np
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class SemiSupervisedGMM(BaseEstimator):
    def __init__(self, n_components=3, max_iter=100, tol=1e-3, reg_covar=1e-6, lambda_supervised=1.0, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.lambda_supervised = lambda_supervised
        self.random_state = np.random.RandomState(random_state)

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        shuffled = self.random_state.permutation(n_samples)
        self.means_ = X[shuffled[:self.n_components]]
        self.weights_ = np.full(self.n_components, 1 / self.n_components)
        self.covariances_ = np.array([np.cov(X.T) + self.reg_covar * np.eye(n_features)
                                      for _ in range(self.n_components)])

    def _e_step(self, X, y=None):
        n_samples, _ = X.shape
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            rv = multivariate_normal(mean=self.means_[k], cov=self.covariances_[k])
            responsibilities[:, k] = self.weights_[k] * rv.pdf(X)

        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        idx = (y != -1).reshape(-1)
        
        if idx.any():
            #print(idx)
            #print(responsibilities.shape)
            #print(np.eye(self.n_components)[y[idx]].reshape(-1))
            responsibilities[idx,:] = np.eye(self.n_components)[y[idx]].reshape(-1, self.n_components)
        
        '''
        if y is not None:
            for idx, label in enumerate(y):
                #print(label)
                if label != -1:
                    hard_assign = np.zeros(self.n_components)
                    hard_assign[label] = 1
                    responsibilities[idx] = (self.lambda_supervised * hard_assign +
                                             (1 - self.lambda_supervised) * responsibilities[idx])
                    responsibilities[idx] /= responsibilities[idx].sum()
         '''           

        return responsibilities

    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        weights = responsibilities.sum(axis=0)
        self.weights_ = weights / n_samples
        self.means_ = (responsibilities.T @ X) / weights[:, np.newaxis]

        self.covariances_ = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = (responsibilities[:, k][:, np.newaxis] * diff).T @ diff / weights[k]
            self.covariances_[k].flat[::n_features + 1] += self.reg_covar

    def fit(self, X, y=None):
        y = np.array(y) if y is not None else np.full(X.shape[0], -1)
        self._initialize_parameters(X)

        log_likelihood = None
        for _ in range(self.max_iter):
            responsibilities = self._e_step(X, y)
            self._m_step(X, responsibilities)

            new_log_likelihood = np.sum(np.log(np.sum([
                self.weights_[k] * multivariate_normal(mean=self.means_[k], cov=self.covariances_[k]).pdf(X)
                for k in range(self.n_components)
            ], axis=0)))

            if log_likelihood is not None and np.abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        self.resp_ = responsibilities
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["means_", "covariances_", "weights_"])
        n_samples, _ = X.shape
        probs = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            rv = multivariate_normal(mean=self.means_[k], cov=self.covariances_[k])
            probs[:, k] = self.weights_[k] * rv.pdf(X)

        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
