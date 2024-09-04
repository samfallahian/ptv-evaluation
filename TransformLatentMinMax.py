'''
Remember, from the "33_Learn..." tranformations logic we found:
High of 2.6400
Low of -1.9800
We will use that here to transform on demand any nacent values
'''

import numpy as np
import pandas as pd
import torch


class FloatConverter:
    def __init__(self):
        self.min_value = -1.9800
        self.max_value = 2.6400
        self.scale = 1.0 / (self.max_value - self.min_value)
        self.shift = -self.min_value * self.scale

    def convert(self, value):
        if isinstance(value, (int, float, np.float16)):
            # Single float value
            converted_value = value * self.scale + self.shift
            if converted_value < self.min_value or converted_value > self.max_value:
                raise ValueError("Value out of range")
            return converted_value
        elif isinstance(value, np.ndarray):
            # Array of float values
            converted_values = value * self.scale + self.shift
            if np.any(converted_values < self.min_value) or np.any(converted_values > self.max_value):
                raise ValueError("Value out of range")
            return converted_values
        elif isinstance(value, pd.DataFrame):
            # DataFrame of float values
            converted_df = value * self.scale + self.shift
            if (converted_df < self.min_value).any().any() or (converted_df > self.max_value).any().any():
                raise ValueError("Value out of range")
            return converted_df
        else:
            raise TypeError(f"Unsupported input type: {type(value)}, value: {value}")

    def unconvert(self, value):
        original_value = (value - self.shift) / self.scale
        tolerance = 0.10  # tolerance of 10%
        min_tol = self.min_value * (1 - tolerance)
        max_tol = self.max_value * (1 + tolerance)

        # Create a warning if value is within tolerance
        # if min_tol <= original_value <= max_tol:
        #     print(f"Warning: Value {original_value} is within tolerance.")

        # Raise ValueError if value is out of range even with tolerance
        if original_value < min_tol or original_value > max_tol:
            print(f"Out of range value: {original_value}")
            raise ValueError("Value out of range")

        if isinstance(value, (int, float, np.float16)):
            # Single float value
            original_value = (value - self.shift) / self.scale
            if original_value < self.min_value or original_value > self.max_value:
                raise ValueError("Value out of range")
            return original_value
        elif isinstance(value, np.ndarray):
            # Array of float values
            original_values = (value - self.shift) / self.scale
            if np.any(original_values < self.min_value) or np.any(original_values > self.max_value):
                raise ValueError("Value out of range")
            return original_values
        elif isinstance(value, torch.Tensor):
            # New condition for Torch.Tensor type
            original_tensor = (value - self.shift) / self.scale
            return original_tensor.cpu().detach().numpy()
        elif isinstance(value, pd.DataFrame):
            # DataFrame of float values
            original_df = (value - self.shift) / self.scale
            if (original_df < self.min_value).any().any() or (original_df > self.max_value).any().any():
                raise ValueError("Value out of range")
            return original_df
        else:
            raise TypeError(f"Unsupported input type: {type(value)}, value: {value}")
