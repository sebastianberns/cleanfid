# Fréchet Inception Distance of Clean Features

Compute the Fréchet Inception Distance (FID) between two distributions from different data sources (tensor, generator model, or dataset). Raw image data is passed through an embedding model to compute ‘clean’ features. Check the [cleanfeatures documentation](https://github.com/sebastianberns/cleanfeatures) for a list of available embedding models (default: InceptionV3). Partially builds on code from [bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR).

## Weighted FID

Apart from the conventional FID formulation, this repository implements *Weighted FID* (wFID) as proposed in [Towards Mode Balancing of Generative Models via Diversity Weights (Berns et al., 2023, ICCC)](https://arxiv.org/abs/2304.11961). 
Weight vectors can be passed alongside the data sources to quantify the weighting of individual data examples. Statistics are then computed on the weighted distribution.

## Setup

### Dependencies

- torch (Pytorch)
- numpy
- scipy
- cleanfeatures ([sebastianberns/cleanfeatures](https://github.com/sebastianberns/cleanfeatures))

## Usage

Assuming that the repository is available in the working directory or Python path.

```python
from cleanfid import FID  # 1.

measure = FID('path/to/model/checkpoint/')  # 2.
fid = measure.score(data_1, data_2)  # 3.
```

1. Import the main class.
2. Create a new instance, providing a directory path of an embedding model. This can be either the place the model checkpoint is already saved, or the place it should be downloaded and saved to.
3. Compute the FID, given two data sources (tensor, generator model, or dataset).

### Example use case

A typical use case is to evaluate model performance during training. For this, it is most efficient to first calculate the mean and covariance of the dataset before the training starts and save the statistics. Then, during training, compute the model statistics. Finally, calculate the FID as the Fréchet distance between the data and model distributions.

```python
from cleanfid import FID

measure = FID('path/to/model/checkpoint/')
num_samples = 50_000
batch_size = 128

# Before training, compute dataset mean and covariance
dataset_stats = measure.compute_feature_statistics(self.dataloader.dataset, 
  num_samples=num_samples, batch_size=batch_size)

# [...]

# During training, compute the model statistics
model_stats = measure.compute_feature_statistics(generator, z_dim=generator.z_dim, 
  num_samples=num_samples, batch_size=batch_size)
# Then evaluate the FID
fid = measure.frechet_distance(*dataset_stats, *model_stats)
```

### FID class

```python
measure = FID(model_path='./models', model='InceptionV3', device=None, **kwargs)
```

- `model_path` (str or Path object, optional): path to directory where model checkpoint is saved or should be saved to. Default: './models'.
- `model` (str, optional): choice of pre-trained feature extraction model. Options:
  - CLIP
  - DVAE (DALL-E)
  - InceptionV3 (default)
  - Resnet50
- `cf` (CleanFeatures, optional): an initialized instance of CleanFeatures. If set, all other arguments will be ignored.
- `device` (str or torch.device, optional): device which the loaded model will be allocated to. Default: 'cuda' if a GPU is available, otherwise 'cpu'.
- `kwargs` (dict): additional model-specific arguments passed on to `cleanfeatures`. See below.

#### CLIP model-specific arguments

- `clip_model` (str, optional): choice of pre-trained CLIP model. Options: RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14 (default), ViT-L/14@336px

### Methods

The class provides three methods to process different types of input: a data tensor, a generator network, or a dataloader.

All methods return a tensor of embedded features [B, F], where F is the number of features.

#### score

Calculate FID between two distributions from two data sources.

```python
fid = measure.score(input1, input2, weights1, weights2, **kwargs)
```

- `input1`, `input2` (Tensor or nn.Module or Dataset): data sources, can be different types (see above)
- `weights1`, `weights2` (ndarray, optional): 1-D array of observation vector weights or probabilities
- `kwargs` (dict): additional data source-specific arguments passed on to the corresponding `cleanfeatures` method. See below.

##### Data source-specific arguments

- Tensor of samples (`torch.Tensor`):
  - `batch_size` (int, optional): Batch size for sample processing. Default: 128
- Generator model (`torch.nn.Module`):
  - `z_dim` (int): Number of generator input dimensions
  - `num_samples` (int): Number of samples to generate and process
  - `batch_size` (int, optional): Batch size for sample processing. Default: 128
- Dataset (`torch.utils.data.Dataset`):
  - `num_samples` (int): Number of samples to generate and process
  - `batch_size` (int, optional): Batch size for sample processing. Default: 128
  - `num_workers` (int, optional): Number of parallel threads. Best practice is to set to the number of CPU threads available. Default: 0
  - `shuffle` (bool, optional): Indicates whether samples will be randomly shuffled or not. Default: False

#### frechet_distance

Calculate Fréchet distance between two multi-variate normal distributions.

```python
distance = measure.frechet_distance(mean1, cov1, mean2, cov2, eps=1e-6)
```

- `mean1`, `mean2` (ndarray): vectors of distribution means [N]
- `cov1`, `cov2` (ndarray): distribution covariance matrices [N x N]
- `eps` (float, optional): small number for numerical stability

#### compute_statistics

Calculate statistics of multi-variate normal distributions. Return tuple of statistics: mean and covariance matrix.

```python
mean, cov = measure.compute_statistics(features, weights)
```

- `features` (ndarray):  Matrix of data features where rows are observations and columns are variables
- `weights` (ndarray, optional):  1-D array of observation vector weights or probabilities

#### compute_features

Compute features given a data source (can be of different types), handled by `cleanfeatures`. Return matrix of data features where rows are observations and columns are variables.

```python
features = measure.compute_features(input, **kwargs)
```

- `input` accepts different data types:
  - (Tensor): data matrix with observations in rows and variables in columns. Processed by `cleanfeatures.compute_features_from_samples()`
  - (nn.Module): pre-trained generator model with tensor output [B, C, W, H]. Processed by `cleanfeatures.compute_features_from_generator()`
  - (Dataset): dataset with tensors in range [0, 1]. Processed by `cleanfeatures.compute_features_from_dataset()`
- `kwargs` (dict): additional data source-specific arguments passed on to the corresponding `cleanfeatures` method. See above.

#### compute_feature_statistics

Compute the statistics of features given a data source (can be of different types), handled by `cleanfeatures`. Return tuple of statistics: mean and covariance matrix.

Combines the previous two methods into one, first calling `compute_features`, followed by `compute_statistics`. 

```python
features = measure.compute_feature_statistics(input, **kwargs)
```

- `input` accepts different data types:
  - (Tensor): data matrix with observations in rows and variables in columns. Processed by `cleanfeatures.compute_features_from_samples()`
  - (nn.Module): pre-trained generator model with tensor output [B, C, W, H]. Processed by `cleanfeatures.compute_features_from_generator()`
  - (Dataset): dataset with tensors in range [0, 1]. Processed by `cleanfeatures.compute_features_from_dataset()`
- `kwargs` (dict): additional data source-specific arguments passed on to the corresponding `cleanfeatures` method. See above.

## References

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. [*Advances in Neural Information Processing Systems*, 30.](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html)
