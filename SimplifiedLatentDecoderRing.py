import logging
import os
import json
from model import WAE
import torch
import sys
from TransformLatentMinMax import FloatConverter
import numpy as np



'''
This code will take two inputs:

1) Experiment Name (4 characters/numbers)
2) Coordinate in Question

The experiment name will be checked against valid experiments when instantiated.
The coordinate in question will be validated for correcteness during instatiation.

The two errors above are not recoverable if thrown.

There are three functions available:
A) Centroid_Only_Normalized: Returns three floats indicating the central values of the 375, namely the 63rd triplet.
Failure to do so will result in an error "NoCentroidFound"
A) Centroid_Only_DeNormalized: Returns three floats (which are returned to their original scale) indicating the central
values of the 375, namely the 63rd triplet. Failure to do so will result in an error "NoCentroidFound"
B) Triplet_Breakdown_In_Order: Returns a list of triplets, which are x,y,z coordinates (e.g. integers). Specifically,
125 such triplets should be returned lest there be an error "NoTripletsFound"
'''


class LatentDecoderRing:
    def __init__(self, model_path):
        # Initialize a model from file
        self.model = WAE()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        #Always good to hard-code the model to NOT LEARN
        self.model.eval()

    def Centroid_Only_DeNormalized(self, input_array):
        end_result_all_values = self.model.decode(input_array)
        end_result_all_values = end_result_all_values.detach().numpy()

        # Ensure the length is as expected
        assert len(end_result_all_values) == 375, "Expected length of array should be 375!"

        # Triplet indices start from 0, so the 63rd triplet would be at index 62
        triplet_idx = 62

        # Triplets mean sets of 3, so multiply index by 3 to get the actual starting index in the array
        start_idx = triplet_idx * 3

        # Get the 63rd triplet
        end_result = end_result_all_values[start_idx: start_idx + 3]
        end_result = tuple(end_result.tolist())

        #convert tuple to numpy array
        end_result = np.asarray(end_result)

        converter = FloatConverter()

        return_value = [converter.unconvert(val) for val in end_result]
        return tuple(return_value)



