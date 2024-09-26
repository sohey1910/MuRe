from abc import ABCMeta, abstractmethod
import numpy as np
import cv2
import os

class BaseModelMeta(metaclass=ABCMeta):
    """
    Abstract base class for all models.
    """
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def read_from_base64(self, data):
        pass

    def read_from_file(self, data):
        if not os.path.isfile(data):
            raise FileNotFoundError
        else:
            return cv2.imread(data)

    def read_from_list(self, data):
        if not isinstance(data, list):
            raise TypeError
        else:
            return np.array(data)