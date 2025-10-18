# DGM-Project1
Project 1: Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN


## Pretrained models

You can use any of our models with `torch.hub.load`.
We provide 2 classes of models for each of our 3 submodels of the CVAE-WGAN architecture


cvae-wgan submodels:
    - CVAE: Conditional Variational Autoencoder
    - Generator
    - Critic
Classes:
    - Standalone: Trained standalone (CVAE) or while CVAE is frozen (Generator, Critic). Sequence lenght is 32.
    - Hybrid: Models trained jointly. We saved the best weights of each of the 4 (32, 64, 128) sequence lenghts used in the curriculum learning.


### CVAE:

**Standalone**

```python
cvae = torch.hub.load('StefannyAC/DGM-Project1', 'cvae_standalone')
```

**Hybrid**

```python
cvae = torch.hub.load('StefannyAC/DGM-Project1', 'cvae_hybrid', seq_len=seq_len)
```


### Generator:

**Standalone**

```python
generator = torch.hub.load('StefannyAC/DGM-Project1', 'generator_standalone')
```

**Hybrid**

```python
generator = torch.hub.load('StefannyAC/DGM-Project1', 'generator_hybrid', seq_len=seq_len)
```


### Critic:

**Standalone**

```python
critic = torch.hub.load('StefannyAC/DGM-Project1', 'critic_standalone')
```

**Hybrid**

```python
critic = torch.hub.load('StefannyAC/DGM-Project1', 'critic_hybrid', seq_len=seq_len)
```

