# Copyright © 2019 Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
#
import random as _random
import turicreate as _tc
from turicreate import extensions as _extensions
from turicreate.toolkits._model import CustomModel as _CustomModel

def create(dataset, target, feature=None, batch_size=0, max_iterations=0,
           seed=None, verbose=True):
    model = _extensions.one_shot_object_detector()
    if seed is None: seed = _random.randint(0, (1<<31)-1)
    # Option arguments to pass in to C++ Object Detector, if we use it:
    # {'mlmodel_path':'darknet.mlmodel', 'max_iterations' : 25}
    augmented_data = model.augment(dataset, target, _tc.SFrame(), {"seed":seed})
    model = _tc.object_detector.create(augmented_data)
    return OneShotObjectDetector(model)

class OneShotObjectDetector(_CustomModel):
    _PYTHON_ONE_SHOT_OBJECT_DETECTOR_VERSION = 1

    def __init__(self, model):
        self.__proxy__ = model

    @classmethod
    def _native_name(cls):
        return "one_shot_object_detector"

    def _get_version(self):
        return self._PYTHON_ONE_SHOT_OBJECT_DETECTOR_VERSION

    @classmethod
    def _load_version(cls, state, version):
        pass

    def predict(self, dataset):
        return self.__proxy__.predict(dataset)

    def evaluate(self, dataset, metric="auto"):
        return self.__proxy__.evaluate(dataset, metric)

    def export_coreml(self, filename, verbose=False):
        # TODO: The model exported out of here would look like an 
        # Object Detector model. We must change the metadata to make it a 
        # OneShotObjectDetector.
        self.__proxy__.export_to_coreml(filename)