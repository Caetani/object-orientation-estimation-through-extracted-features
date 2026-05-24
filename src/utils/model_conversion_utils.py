import emlearn
import numpy as np

class SingleOutputTreeProxy:
    """
    Wraps a multi-output sklearn Tree object, exposing only one output's
    leaf values — without modifying the original tree or touching __setstate__.
    """
    def __init__(self, tree, output_idx):
        self._tree = tree
        self._output_idx = output_idx

    def __getattr__(self, name):
        if name == 'value':
            return self._tree.value[:, self._output_idx:self._output_idx+1, :]
        return getattr(self._tree, name)


class SingleOutputRegressorProxy:
    """
    Wraps a multi-output DecisionTreeRegressor, replacing tree_ with the proxy.
    """
    def __init__(self, model, output_idx):
        self._model = model
        self.tree_ = SingleOutputTreeProxy(model.tree_, output_idx)
        self.n_outputs_ = 1
        self.n_features_in_ = model.n_features_in_

    def __getattr__(self, name):
        return getattr(self._model, name)

def convert_model(model, save_dir, use_dtype='float'):
    for i, col in enumerate(['qw', 'qx', 'qy', 'qz']):
        proxy = SingleOutputRegressorProxy(model, i)
        cmodel = emlearn.convert(proxy, kind='DecisionTreeRegressor', method='inline', dtype=use_dtype)
        cmodel.save(name=f'model_{col}', file=f'{save_dir}/model_{col}.h')

def fix_thresholds(model):
    model.tree_.threshold[:] = np.ceil(model.tree_.threshold)
    return model