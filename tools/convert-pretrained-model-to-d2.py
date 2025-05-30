#!/usr/bin/env python

import pickle as pkl
import sys

import torch

"""
Usage:
  # download pretrained swin model:
  wget https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth
  # run the conversion
  ./convert-pretrained-model-to-d2.py swin_tiny_patch4_window7_224_22k.pth swin_tiny_patch4_window7_224_22k.pkl
  # Then, use swin_tiny_patch4_window7_224_22k.pkl with the following changes in config:
MODEL:
  WEIGHTS: "/path/to/swin_tiny_patch4_window7_224_22k.pkl"
INPUT:
  FORMAT: "RGB"
"""

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")["model"]

    res = {"model": obj, "__author__": "third_party", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
