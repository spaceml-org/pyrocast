import torch
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable
print("hsic2.py")


#####################################################################################################
# HSIC in torch
#####################################################################################################

def centering(K, device):
    n_samples = K.shape[0]
    H = torch.eye(K.shape[0], ) - (1 / n_samples) * torch.ones((n_samples, n_samples))
    
    H = H.type(torch.DoubleTensor)
    H = H.to(device)
    #envs = envs.type(torch.DoubleTensor)
    return torch.mm(torch.mm(H, K), H)

def hsicRBF(x, z, device):
    
    #print("x shape",x.shape)
    #print(x[0:5,0:3])
    
    K_x = torch.cdist(x, x)
    K_x = K_x**2
    med = torch.median(K_x)
    mean = torch.mean(K_x)
    nug = torch.ones(1)*0.000000001
    nug = nug.to(device)
    med = torch.max(torch.cat((med[None], mean[None], nug)))
    #print("med x: ", med.cpu().detach().numpy())
    sigma = 1 / med 
    #print("sigma x: ", sigma.cpu().detach().numpy())
    K_x = torch.exp(-sigma*K_x)
    
    #print("K_x shape: ", K_x.shape)
    #print(K_x[0:4, 0:4])
    
    K_z = torch.cdist(z, z)
    K_z = K_z**2
    med = torch.median(K_z)
    mean = torch.mean(K_z)
    nug = torch.ones(1)*0.00001
    nug = nug.to(device)
    med = torch.max(torch.cat((med[None], mean[None], nug)))
    #print("med z: ", med.cpu().detach().numpy())
    sigma = 1 / med
    #print("sigma z: ", sigma.cpu().detach().numpy())
    K_z = torch.exp(-sigma*K_z)
    
    K_x = centering(K_x, device)
    K_z = centering(K_z, device)
    norm_K_x = torch.linalg.norm(K_x)
    norm_K_z = torch.linalg.norm(K_z)
    
    maxSens = torch.ones(1)*0.0001
    maxSens = maxSens.to(device)
    concatVal = torch.cat( (maxSens, norm_K_x[None]) )
    norm_K_x = torch.max( concatVal )
    
    maxSens = torch.ones(1)*0.0001
    maxSens = maxSens.to(device)
    concatVal = torch.cat( (maxSens, norm_K_z[None]) )
    norm_K_z = torch.max( concatVal )
    
    res = torch.sum(K_x*K_z)
    #print("res: ", res.cpu().detach().numpy())
    res = res / norm_K_x / norm_K_z
    return  res, norm_K_x,  norm_K_z


#####################################################################################################
# HSIC in jax
#####################################################################################################

# Squared Euclidean Distance Formula
@jax.jit
def sqeuclidean_distance(x: np.ndarray, y: np.ndarray) -> jnp.DeviceArray:
    """
        Calculates the euclidean distance between two vectors
    Args:
        x: vector 1: usually a feature map in Rp for an observation i
        y: vector 2: usually a feature map in Rp for an observation 2

    Returns:
        Euclidean distance value
    """
    return jnp.sum((x - y) ** 2)


# RBF Kernel
#@jax.jit
def rbf_kernel(params: dict, x: object, y: object) -> object:
    """
       Calculates the Radial basis function (rbf) similarity between two vectors
    Args:
        params:
        x: vector 1: usually a feature map in Rp for an observation i
        y: vector 2: usually a feature map in Rp for an observation 2

    Returns:
        RBF similarity value
    """
    return jnp.exp(- params['gamma'] * sqeuclidean_distance(x, y))


# Covariance Matrix
def covariance_matrix(kernel_func: Callable, x: np.ndarray, y: np.ndarray) -> jnp.ndarray:
    """
        Applies kernel function k k(x[i,:], x[j,.]) for i,j in 1,...,n. kernel must not take
        parameters
    Args:
        kernel_func: kernel function taking no parameters
        x: feature map in n x p
        y: feature map in m x p

    Returns:
        kernel_func similarity matrix
    """
    mapx1 = jax.vmap(
        lambda x, y: kernel_func(x, y), in_axes=(0, None), out_axes=0)
    mapx2 = jax.vmap(
        lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
    return mapx2(x, y)


# Covariance Matrix
def rbf_kernel_matrix(params: dict, x: np.ndarray, y: np.ndarray) -> jnp.ndarray:
    """
    Calculates the RBF kernel similarity matrix between feature maps x and y
    Args:
        params:
        x: feature map in n x p
        y: feature map in m x p

    Returns:
        rbf kernel similarity matrix n x m
    """
    mapx1 = jax.vmap(lambda x, y: rbf_kernel(params, x, y), in_axes=(0, None), out_axes=0)
    mapx2 = jax.vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
    return mapx2(x, y)

@jax.jit
def centering_jax(K: jnp.ndarray) -> jnp.ndarray:
    """
     Centers similarity matrices. Has the same result as if (infinite) features had been centered
    Args:
        K: Matrix to center

    Returns:
        centerd similiarity matrix:
    """
    n_samples = K.shape[0]
    #logging.debug(f"N: {n_samples}")
    #logging.debug(f"I: {np.ones((n_samples, n_samples)).shape}")
    H = jnp.eye(K.shape[0], ) - (1 / n_samples) * jnp.ones((n_samples, n_samples))
    return jnp.dot(jnp.dot(H, K), H)

@jax.jit
def hsicRBF_jax(x: np.ndarray, z: np.ndarray) -> jnp.DeviceArray:
    """
     Obtain Hilbert Schmidt Independence Criterion for features x and z using rbf
     kernel with lengthscale determined with median heuristic
    Args:
        x: matrix of n x p1 features
        z: matrix of n x p2 features

    Returns:
       hsic value
    """
    distsX = covariance_matrix(sqeuclidean_distance, x, x)
    #sigma = 1 / jnp.median(distsX)
    
    med = jnp.median(distsX)
    mean = jnp.mean(distsX)
    nug = jnp.ones(1)*0.000000001
    med = jnp.max(jnp.concatenate((med[None], mean[None], nug)))
    sigma = 1 / med 
    
    K_x = rbf_kernel_matrix({'gamma': sigma}, x, x)
    distsZ = covariance_matrix(sqeuclidean_distance, z, z)
    
    #sigma = 1 / jnp.median(distsZ)
    
    med = jnp.median(distsZ)
    mean = jnp.mean(distsZ)
    nug = jnp.ones(1)*0.000000001
    med = jnp.max(jnp.concatenate((med[None], mean[None], nug)))
    sigma = 1 / med 
    
    
    K_z = rbf_kernel_matrix({'gamma': sigma}, z, z)
    K_x = centering_jax(K_x)
    K_z = centering_jax(K_z)
    return jnp.sum(K_x * K_z) / jnp.linalg.norm(K_x) / jnp.linalg.norm(K_z)
