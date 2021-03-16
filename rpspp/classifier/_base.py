
class ClassifierMixin:

    def fit(self, data, labels, test_data=None, test_label=None, log_dir=None):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError
