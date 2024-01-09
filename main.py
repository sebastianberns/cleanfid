#!/usr/bin/env python

from pathlib import Path
from typing import Tuple, Union, Optional
import warnings

import numpy as np
from scipy.linalg import sqrtm  # type: ignore[import]
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import Dataset

from cleanfeatures import CleanFeatures


class FID:
    """
    Compute data features given an input source
        model_path (str, Path):  path to the save directory of embedding model checkpoints. Default: './models'
        model (str, optional): name of embedding model. Default: 'InceptionV3'
        cf (CleanFeatures, optional): instance of CleanFeatures
        device (str, device, optional):  device (e.g. 'cpu' or 'cuda:0')
        kwargs (dict): additional model-specific arguments passed on to CleanFeatures.
    """
    def __init__(self, model_path: Union[str, Path] = './models', model: str = 'InceptionV3', 
                 cf: Optional[CleanFeatures] = None,
                 device: Optional[Union[str, torch.device]] = None, **kwargs) -> None:
        if cf is None:
            cf = CleanFeatures(model_path, model=model, device=device, log='warning', **kwargs)
        self.cf = cf


    """
    Compute features given a data source
        input (Tensor, nn.Module, Dataset):  data source to process
        kwargs (dict): additional data source-specific arguments passed on to the corresponding `cleanfeatures` method
    Return matrix of data features (ndarray) where rows are observations 
    and columns are variables
    """
    def compute_features(self, input: Union[Tensor, nn.Module, Dataset], **kwargs) -> np.ndarray:
        if isinstance(input, Tensor):  # Tensor ready for processing
            features = self.cf.compute_features_from_samples(input, **kwargs)
        elif isinstance(input, nn.Module):  # Generator model
            features = self.cf.compute_features_from_generator(input, **kwargs)
        elif isinstance(input, Dataset):  # Data set
            features, targets = self.cf.compute_features_from_dataset(input, **kwargs)
        else:
            raise ValueError(f"Input type {type(input)} is not supported")
        
        return features.cpu().numpy()


    """
    Calculate statistics of multi-variate normal distributions
        features (ndarray):  Matrix of data features where rows are observations and columns are variables
        weights (ndarray, optional):  1-D array of observation vector weights or probabilities
    Return tuple of statistics: mean (ndarray) and covariance matrix (ndarray)
    """
    def compute_statistics(self, features: np.ndarray, weights: Optional[
                           np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if weights is None:
            mean = np.mean(features, axis=0)
            cov = np.cov(features, rowvar=False)
        else:
            weights = np.atleast_1d(weights)
            mean = np.average(features, axis=0, weights=weights)
            cov = np.cov(features, rowvar=False, ddof=0, aweights=weights)
        return mean, cov
    

    """
    Compute data statistics given an input source

        input (Tensor, nn.Module, Dataset):  data source to process
        weights (ndarray, optional):  1-D array of observation vector weights or probabilities

    Return tuple of statistics: mean (ndarray) and covariance matrix (ndarray)
    """
    def compute_feature_statistics(self, input: Union[Tensor, nn.Module, Dataset], 
                                   weights: Optional[np.ndarray] = None, 
                                   **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        features = self.compute_features(input, **kwargs)
        stats = self.compute_statistics(features, weights=weights)
        return stats  # mean, cov


    """
    Calculate Fréchet distance between two multi-variate normal distributions
        mean1, mean2 (ndarray):  Vectors of distribution means [N]
        cov1, cov2 (ndarray):  Distribution covariance matrices [N x N]
    Return distance (float)
    """
    def frechet_distance(self, mean1: np.ndarray, cov1: np.ndarray, 
                               mean2: np.ndarray, cov2: np.ndarray, 
                               eps: float = 1e-6) -> float:
        mean1 = np.atleast_1d(mean1)
        mean2 = np.atleast_1d(mean2)

        cov1 = np.atleast_2d(cov1)
        cov2 = np.atleast_2d(cov2)

        assert mean1.shape == mean2.shape, f"Mean vectors have different lengths (mean1: {mean1.shape}, mean2: {mean2.shape})"
        assert cov1.shape == cov2.shape, f"Covariance matrices have different dimensions (mean1: {cov1.shape}, mean2: {cov2.shape})"

        mean_diff = mean1 - mean2
        mean_dist = np.dot(mean_diff, mean_diff)

        # Product might be almost singular
        covmean, _ = sqrtm(np.dot(cov1, cov2), disp=False)
        if not np.isfinite(covmean).all():
            warnings.warn(f"Mean of covariance matrices produces singular product. Adding {eps} to diagonals.")
            offset = np.eye(cov1.shape[0]) * eps
            covmean = sqrtm(np.dot(cov1 + offset, cov2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                raise ValueError(f"Mean of covariance matrices contains imaginary component: {np.max(np.abs(covmean.imag))}")
            covmean = covmean.real

        distance = mean_dist + np.trace(cov1) + np.trace(cov2) - 2 * np.trace(covmean)
        return distance


    """
    Calculate FID given two data sources
        input1, input2 (Tensor, nn.Module, Dataset):  data sources can be different types which envoke different processing functions
        weights1, weights2 (ndarray, optional):  1-D array of observation vector weights or probabilities

        kwargs are passed on to processing functions which take different additional arguments depending on the type of data source
            Tensor -> compute_features_from_samples:
                batch_size (int, optional):  Batch size for iterative processing. Default: 128
            nn.Module -> compute_features_from_generator:
                z_dim (int):  Number of generator input dimensions
                num_samples (int):  Number of samples to generate and process
                batch_size (int, optional):  Batch size for generator sampling. Default: 128
            Dataset -> compute_features_from_dataset:
                num_samples (int):  Number of samples to process
                batch_size (int, optional):  Batch size for sampling. Default: 128
                num_workers (int, optional):  Number of parallel threads. Best practice
                    is to set to the number of CPU threads available. Default: 0
                shuffle (bool, optional):  Indicates whether samples will be randomly
                    shuffled or not. Default: False
    Return distance between two distributions from sources (float)
    """
    def score(self, input1: Union[Tensor, nn.Module, Dataset], 
                    input2: Union[Tensor, nn.Module, Dataset],
                    weights1: Optional[np.ndarray] = None,
                    weights2: Optional[np.ndarray] = None, **kwargs) -> float:
        features1 = self.compute_features(input1, **kwargs)
        features2 = self.compute_features(input2, **kwargs)
        stats1 = self.compute_statistics(features1, weights=weights1)
        stats2 = self.compute_statistics(features2, weights=weights2)
        fid = self.frechet_distance(*stats1, *stats2)
        return fid
