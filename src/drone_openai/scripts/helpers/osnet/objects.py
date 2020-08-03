#! /usr/bin/env python

from torchreid.utils import FeatureExtractor

extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='osnet_x1_0.pth',
    device='cuda'
)