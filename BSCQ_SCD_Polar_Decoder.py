
import numpy as np
from numpy import linalg as LA
# import jax
# import jax.numpy as jnp
# from jax import grad, jit, vmap
from scipy.linalg import sqrtm
from numpy.linalg import matrix_rank
# from jax.numpy import linalg as jla
from scipy.linalg import logm, expm

from scipy.linalg import block_diag
import copy
import numba
from numba import int64, float64, jit, njit, vectorize
import matplotlib.pyplot as plt

import argparse as ap
import json
import sys
import time


def qubit(x):
  '''
    Compute orthonormal basis states

          Arguments:
                  x : 0 or 1

          Returns:
                  state |0> or |1>
  '''
  if x==0:
    return np.array([1,0])
  elif x==1:
    return np.array([0,1])


def pauli(x):
    '''
    Compute pauli matrices

          Arguments:
                  x : {0,1,2,3}

          Returns:
                 Pauli Matrices: {I,X,Y,Z}

    '''
    if x==0:
        return np.array([[1,0],[0,1]])
    if x==1:
        return np.array([[0,1],[1,0]])
    if x==2:
        return np.array([[0,-1j],[+1j,0]])
    if x==3:
        return np.array([[1,0],[0,-1]])
    else:
        print('invalid')

# modified rho based on delta, gamma, and input x rather than theta and p
def rhom(delta,g,x):
  '''
   Code to compute density matrix for given parameters

       Arguments:
                d(float): delta parameter of the BSCQ channel
                g(float): gamma parameter of the BSCQ channel
                x(0 or 1): 1 indicates applying Pauli X on the density matrix

       Returns:
                (float[:,:]): density matrix

  '''
  return pauli(x)@np.array([[delta, g],[g, 1-delta]])@pauli(x)

# "bitnode" take input states parameterized by (d1, g1) and (d2, g2) and returns the post-measurement state parameters and probabilities at a bitnode
def bitnode(d1,d2,g1,g2):

  '''
    Code to compute parameters of post measurement states after applying bitnode operation
         Arguments:
                  d1(float): parameter delta for the first BSCQ channel
                  d2(float): parameter delta for the second BSCQ channel
                  g1(float) : parameter gamma for the first BSCQ channel
                  g2(float): parameter gamma for the second  BSCQ channel

          Returns:
                  (list[float[:],float[:]]): delta and gamma parameters for the post measurement states
                  (float[:]): probability of the post measurement states

  '''
  # the state given root value z=0 is rho0 = np.kron(rhom(d1,g1,0),rhom(d2,g2,0))
  # the state given root value z=1 is rho1 = np.kron(rhom(d1,g1,1),rhom(d2,g2,1))

  # we find the paired measurements by taking the eigenvectors of the difference matrix rhofull
  rhofull = np.kron(rhom(d1,g1,0),rhom(d2,g2,0))-np.kron(rhom(d1,g1,1),rhom(d2,g2,1))

  evals, evecs = np.linalg.eigh(rhofull)
  # fix eigenvector v0
  v0 = evecs[:,0]

  # symmetry operator Un = np.kron(pauli(1),pauli(1))
  Un = np.kron(pauli(1),pauli(1))
  # check if the second eigenvector evecs[:,1] is orthogonal to Un@evecs[:,1]
  x = evecs[:,1]@(Un@evecs[:,1])

  #print("evals")
  #print(evals)
  if np.abs(x)<10e-10:
    v1 = evecs[:,1]
  # if not orthogonal, combine evecs[:,1], evecs[:,2] to create v1 s.t. v1@(Un@v1)= 0
  if np.abs(x)>=10e-10:
    vec1, vec2 = evecs[:,1], evecs[:,2]
    b11, b12, b21, b22 = np.inner(vec1, (Un@vec1).conj()), np.inner(vec2, (Un@vec1).conj()), np.inner(vec1, (Un@vec2).conj()), np.inner(vec2, (Un@vec2).conj())

    alpha = (-b12-b21-np.sqrt((b12+b21)**2-4*b11*b22))/(2*b22)
    v1 = vec1+alpha*vec2
    v1 = v1/np.sqrt(v1@v1)
    ##
  #print("v0")
  #print(v0)
  #print("v1")
  #print(v1)
  ##
  # the paired measurement is then given by {|v0><v0| + Un|v0><v0|Un, |v1><v1| + Un|v1><v1|Un}
  ## find new state parameters (d1a, g1a) for measurement outcome 0
  # find probability p0 of observing measurement  outcome 0
  p0 = v0@np.kron(rhom(d1,g1,0),rhom(d2,g2,0))@v0+v0@Un@np.kron(rhom(d1,g1,0),rhom(d2,g2,0))@Un@v0
  d1a, g1a = v0@np.kron(rhom(d1,g1,0),rhom(d2,g2,0))@v0/(p0+10e-21), v0@np.kron(rhom(d1,g1,0),rhom(d2,g2,0))@(Un@v0)/(p0+10e-21)
  ## find new state parameters (d1b, g1b) for measurement outcome 1
  # find probability p1 of observing measurement  outcome 1
  p1 = v1@np.kron(rhom(d1,g1,0),rhom(d2,g2,0))@v1+v1@Un@np.kron(rhom(d1,g1,0),rhom(d2,g2,0))@Un@v1
  d2a, g2a = v1@np.kron(rhom(d1,g1,0),rhom(d2,g2,0))@v1/(p1+10e-21), v1@np.kron(rhom(d1,g1,0),rhom(d2,g2,0))@(Un@v1)/(p1+10e-21)
  #print("pvec")
  #print([p0, p1])
  return np.array([[1-d1a, g1a], [1-d2a, g2a]]), np.array([p0, p1])

# "checknode" take input states parameterized by (d1, g1) and (d2, g2) and returns the post-measurement state parameters and probabilities at a checknode
def checknode(d1,d2,g1,g2):
  '''
    Code to compute parameters of post measurement states after applying checknode operation
         Arguments:
                  d1(float): parameter delta for the first BSCQ channel
                  d2(float): parameter delta for the second BSCQ channel
                  g1(float) : parameter gamma for the first BSCQ channel
                  g2(float): parameter gamma for the second  BSCQ channel

          Returns:
                  (list[float[:],float[:]]): delta and gamma parameters for the post measurement states
                  (float[:]): probability of the post measurement states

  '''

  # rho0, rho1 correspond to the states at a check node when z=0 (z=1) respectively
  rho0, rho1 = 1/2*(np.kron(rhom(d1,g1,0),rhom(d2,g2,0)) + np.kron(rhom(d1,g1,1),rhom(d2,g2,1))), 1/2*(np.kron(rhom(d1,g1,0),rhom(d2,g2,1)) + np.kron(rhom(d1,g1,1),rhom(d2,g2,0)))
  # for check node combining, the optimal choice of eigenvectors appears to always be generated by v0 and v1
  v0 = 1/np.sqrt(2)*np.array([1,0,0,1])
  v1 = 1/np.sqrt(2)*np.array([-1, 0, 0, 1])
  # symmetry operator for a check node
  Un = np.kron(pauli(1),pauli(0))
  ## find new state parameters (d1a, g1a) for measurement outcome 0
  # find probability p0 of observing measurement  outcome 0
  p0 = v0@rho0@v0+v0@Un@rho0@Un@v0
  d1a, g1a = v0@rho0@v0/(p0+10e-21), v0@rho0@(Un@v0)/(p0+10e-21)
  ## find new gamma, delta for second outcome
  p1 = v1@rho0@v1+v1@Un@rho0@Un@v1
  d2a, g2a = v1@rho0@v1/(p1+10e-21), v1@rho0@(Un@v1)/(p1+10e-21)
  # return new gamma, delta pairs as well as respective probabilities tra and tr2a
  # d1a=min([d1a,1-d1a])
  # d2a=min([d2a,1-d2a])
  return np.array([[d1a, g1a], [d2a, g2a]]), np.array([p0, p1])

H = 1/np.sqrt(2)*np.array([[1,-1],[1,1]])
sgz = np.array([[1,0],[0,-1]])
sgx = np.array([[0,1],[1,0]])

def helstrom(density_mat1,unitary):
  '''
   Code to compute Helstrom measurement

     Arguments:
             density_mat1(float[:,:]): density matrix
             unitary(float[:,:]): unitary
     Returns:
             l(float[:]): eigenvalues
             v(float[:,:]): eigenvectors
  '''
  r1=density_mat1
  u=unitary
  r2=u@r1@np.conjugate(np.transpose(u))
  l,v=LA.eig(r1-r2)
  return l,v


def helstrom_success(rho1,unitary):
  '''
   Code to compute Helstrom success probability

     Arguments:
             density_mat1(float[:,:]): density matrix
             unitary(float[:,:]): unitary
     Returns:
             (float): Helstrom success probability
  '''
  l,vec=helstrom(rho1,unitary)
  u=unitary
  rho2=u@rho1@np.conjugate(np.transpose(u))
  v_pos_eig = np.array(vec[:,l>0])
  v_pos_eigh= np.conjugate(np.transpose(v_pos_eig))
  p1=np.trace(v_pos_eigh @ rho1 @ v_pos_eig)
  p2=1-np.trace(v_pos_eigh@ rho2 @ v_pos_eig)
  return 0.5*(p1+p2)



def helstrom_projection_operators(rho0,rho1):
    '''
         Code to obtain projection operators corresponding to the Helstrom measurement between tow density matrices

         Arguments:
             rho0(float[:,:]): first density matrix
             rh01(float[:,:]): second density matrix


         Returns:
            pi0(float[:,:]): first projection operator
            pi1(float[:,:]): second projection operator
    '''
    l,vec=LA.eigh(rho0-rho1)
    pi0=np.zeros(np.shape(rho0))
    pi1=np.zeros(np.shape(rho1))
    for i in range(len(l)):
        if l[i]>0:
            pi0=pi0+np.outer(vec[:,i],vec[:,i])
        else:
            pi1=pi1+np.outer(vec[:,i],vec[:,i])

    return pi0,pi1

@njit
def rhom_jit(delta:float64,g:float64) -> (float64[:,:]):
    '''
       Code to compute density matrix for given parameters using numba

          Arguments:
                d(float64): delta parameter of the BSCQ channel
                g(float)64: gamma parameter of the BSCQ channel


          Returns:
                (float64[:,:]): density matrix

    '''
    return np.array([[delta, g],[g, 1-delta]],dtype=float64)

# "bitnode" take input states parameterized by (d1, g1) and (d2, g2) and returns the post-measurement state parameters and probabilities at a bitnode
@njit
def bitnode_jit(d1:float64,d2:float64,g1:float64,g2:float64) ->(float64[:,:],float64[:]):
  '''
    Code to compute parameters of post measurement states after applying bitnode operation using numba
         Arguments:
                  d1(float64): parameter delta for the first BSCQ channel
                  d2(float64): parameter delta for the second BSCQ channel
                  g1(float64) : parameter gamma for the first BSCQ channel
                  g2(float64): parameter gamma for the second  BSCQ channel

          Returns:
                  (list[float64[:],float64[:]]): delta and gamma parameters for the post measurement states
                  (float64[:]): probability of the post measurement states

  '''
  # the state given root value z=0 is rho0 = np.kron(rhom(d1,g1,0),rhom(d2,g2,0))
  # the state given root value z=1 is rho1 = np.kron(rhom(d1,g1,1),rhom(d2,g2,1))

  # we find the paired measurements by taking the eigenvectors of the difference matrix rhofull
  x=np.array([[0,1],[1,0]],dtype=float64)

  rhofull = np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))-np.kron(x@rhom_jit(d1,g1)@x,x@rhom_jit(d2,g2)@x)

  evals, evecs = np.linalg.eigh(rhofull)
  # fix eigenvector v0
  v0 = evecs[:,0]

  # symmetry operator Un = np.kron(pauli(1),pauli(1))
  Un = np.kron(x,x)
  # check if the second eigenvector evecs[:,1] is orthogonal to Un@evecs[:,1]
  x1 = evecs[:,1]@(Un@evecs[:,1])

  #print("evals")
  #print(evals)
  if np.abs(x1)<10e-10:
    v1 = evecs[:,1]
  # if not orthogonal, combine evecs[:,1], evecs[:,2] to create v1 s.t. v1@(Un@v1)= 0
  if np.abs(x1)>=10e-10:
    vec1, vec2 = evecs[:,1], evecs[:,2]
    b11, b12, b21, b22 = np.dot(vec1, (Un@vec1).conj()), np.dot(vec2, (Un@vec1).conj()), np.dot(vec1, (Un@vec2).conj()), np.dot(vec2, (Un@vec2).conj())

    alpha = (-b12-b21-np.sqrt((b12+b21)**2-4*b11*b22))/(2*b22)
    v1 = vec1+alpha*vec2
    v1 = v1/np.sqrt(v1@v1)
    ##
  #print("v0")
  #print(v0)
  #print("v1")
  #print(v1)
  ##
  # the paired measurement is then given by {|v0><v0| + Un|v0><v0|Un, |v1><v1| + Un|v1><v1|Un}
  ## find new state parameters (d1a, g1a) for measurement outcome 0
  # find probability p0 of observing measurement  outcome 0
  p0 = v0@np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))@v0+v0@Un@np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))@Un@v0
  d1a, g1a = v0@np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))@v0/(p0+10e-21), v0@np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))@(Un@v0)/(p0+10e-21)
  ## find new state parameters (d1b, g1b) for measurement outcome 1
  # find probability p1 of observing measurement  outcome 1
  p1 = v1@np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))@v1+v1@Un@np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))@Un@v1
  d2a, g2a = v1@np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))@v1/(p1+10e-21), v1@np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2))@(Un@v1)/(p1+10e-21)
  #print("pvec")
  #print([p0, p1])
  d1a=min([d1a,1-d1a])
  d2a=min([d2a,1-d2a])
  return np.array([[d1a, g1a], [d2a, g2a]]), np.array([p0, p1])

# "checknode" take input states parameterized by (d1, g1) and (d2, g2) and returns the post-measurement state parameters and probabilities at a checknode
@njit
def checknode_jit(d1:float64,d2:float64,g1:float64,g2:float64) ->(float64[:,:],float64[:]):
  '''
    Code to compute parameters of post measurement states after applying checknode operation using numba
         Arguments:
                  d1(float64): parameter delta for the first BSCQ channel
                  d2(float64): parameter delta for the second BSCQ channel
                  g1(float64) : parameter gamma for the first BSCQ channel
                  g2(float64): parameter gamma for the second  BSCQ channel

          Returns:
                  (list[float64[:],float64[:]]): delta and gamma parameters for the post measurement states
                  (float64[:]): probability of the post measurement states

  '''
  x=np.array([[0,1],[1,0]],dtype=float64)
  I=np.array([[1,0],[0,1]],dtype=float64)
  # rho0, rho1 correspond to the states at a check node when z=0 (z=1) respectively
  rho0, rho1 = 1/2*(np.kron(rhom_jit(d1,g1),rhom_jit(d2,g2)) + np.kron(x@rhom_jit(d1,g1)@x,x@rhom_jit(d2,g2)@x)), 1/2*(np.kron(rhom_jit(d1,g1),x@rhom_jit(d2,g2)@x) + np.kron(x@rhom_jit(d1,g1)@x,rhom_jit(d2,g2)))
  # for check node combining, the optimal choice of eigenvectors appears to always be generated by v0 and v1
  v0 = 1/np.sqrt(2)*np.array([1,0,0,1])
  v1 = 1/np.sqrt(2)*np.array([-1, 0, 0, 1])
  # symmetry operator for a check node
  Un = np.kron(x,I)
  ## find new state parameters (d1a, g1a) for measurement outcome 0
  # find probability p0 of observing measurement  outcome 0
  p0 = v0@rho0@v0+v0@Un@rho0@Un@v0
  d1a, g1a = v0@rho0@v0/(p0+10e-21), v0@rho0@(Un@v0)/(p0+10e-21)
  ## find new gamma, delta for second outcome
  p1 = v1@rho0@v1+v1@Un@rho0@Un@v1
  d2a, g2a = v1@rho0@v1/(p1+10e-21), v1@rho0@(Un@v1)/(p1+10e-21)
  # return new gamma, delta pairs as well as respective probabilities tra and tr2a
  d1a=min([d1a,1-d1a])
  d2a=min([d2a,1-d2a])
  return np.array([[d1a, g1a], [d2a, g2a]]), np.array([p0, p1])

@njit
def bitnode_jit2(d1:float64,d2:float64,g1:float64,g2:float64,pr=None)->(float64[:,:]):
  '''
    Code to sample post measurement state after applying bitnode operation using numba
         Arguments:
                  d1(float64): parameter delta for the first BSCQ channel
                  d2(float64): parameter delta for the second BSCQ channel
                  g1(float64) : parameter gamma for the first BSCQ channel
                  g2(float64): parameter gamma for the second  BSCQ channel

          Returns:
                  (float[:]): delta and gamma of the sampled post measurement state

  '''
  rho,pb=bitnode_jit(d1,d2,g1,g2)
  if pr==None:
    s=pb[0]
  else:
    s=pr
  o=int(np.random.random()>s)
  #choice([0,1],p=[s,1-s])
  if o==0:
    return rho[0]
  else:
    return rho[1]

@njit
def checknode_jit1(d1:float64,d2:float64,g1:float64,g2:float64,pr=None)->(float64[:,:]):
  '''
    Code to sample post measurement state after applying checknode operation using numba
         Arguments:
                  d1(float64): parameter delta for the first BSCQ channel
                  d2(float64): parameter delta for the second BSCQ channel
                  g1(float64) : parameter gamma for the first BSCQ channel
                  g2(float64): parameter gamma for the second  BSCQ channel

          Returns:
                  (float[:]): delta and gamma of the sampled post measurement state

  '''
  rho,pc=checknode_jit(d1,d2,g1,g2)
  if pr==None:
    s=pc[0]
  else:
    s=pr
  o=int(np.random.random()>s)
  if o==0:
    return rho[0]
  else:
    return rho[1]

@njit
def bitnode_vec_jit(d1:float64[:],d2:float64[:],g1:float64[:],g2:float64[:],pr_vec=None,perm=None)->(float64[:],float64[:]):
    '''
     Code to apply bitnode operation on two bags with delta gamma parameters and sample post measurement delta gamma parameters,
          chooses delta, gamma parameters randomly from the bag.

         Arguments:
                  d1(float64[:]):  list of delta parameters corresponding to the first bag
                  d2(float64[:]): list of gamma parameters corresponding to the first bag
                  g1(float64[:]) : list of delta parameters corresponding to the second bag
                  g2(float64[:]): list of gamma parameters corresponding to the second bag
        Returns:
                 d(float64[:]): delta parameters for the sampled post measurement states
                 g(float64[:]):  gamma parameters for the sampled post measurement states

    '''
    l=np.shape(d1)[0]
    if perm==None:
      p=np.random.permutation(l)
    else:
      p=perm
    d=np.zeros(l)
    g=np.zeros(l)
    if pr_vec==None:
      for i in range(l):
        d[i],g[i]=bitnode_jit2(d1[i],d2[p[i]],g1[i],g2[p[i]])

    else:
      for i in range(l):
        d[i],g[i]=bitnode_jit2(d1[i],d2[p[i]],g1[i],g2[p[i]],pr_vec[i])

    return d,g
@njit
def checknode_vec_jit(d1:float64[:],d2:float64[:],g1:float64[:],g2:float64[:],pr_vec=None,perm=None)->(float64[:],float64[:]):
    '''
     Code to apply checknode operation on two bags with delta gamma parameters and sample post measurement delta gamma parameters,
          chooses delta, gamma parameters randomly from the bag.

         Arguments:
                  d1(float64[:]):  list of delta parameters corresponding to the first bag
                  d2(float64[:]): list of gamma parameters corresponding to the first bag
                  g1(float64[:]) : list of delta parameters corresponding to the second bag
                  g2(float64[:]): list of gamma parameters corresponding to the second bag
        Returns:
                 d(float64[:]): delta parameters for the sampled post measurement states
                 g(float64[:]):  gamma parameters for the sampled post measurement states

    '''
    l=np.shape(d1)[0]
    if perm==None:
      p=np.random.permutation(l)
    else:
      p=perm
    d=np.zeros(l)
    g=np.zeros(l)
    if pr_vec==None:
      for i in range(l):
        d[i],g[i]=checknode_jit1(d1[i],d2[p[i]],g1[i],g2[p[i]])

    else:
      for i in range(l):
        d[i],g[i]=checknode_jit1(d1[i],d2[p[i]],g1[i],g2[p[i]],pr_vec[i])

    return d,g


@njit
def bitnode_power_jit(d:float64[:],g:float64[:],k:int64,pr_vec=None,perm=None)->(float64[:],float64[:]):
  '''
    code to apply bitnode operation with given bitnode degree on two bags with delta gamma parameters
    and sample post measurement delta gamma parameters, chooses delta, gamma parameters randomly from the bag.
    Arguments:
                  d1(float64[:]):  list of delta parameters corresponding to the first bag
                  d2(float64[:]): list of gamma parameters corresponding to the first bag
                  g1(float64[:]) : list of delta parameters corresponding to the second bag
                  g2(float64[:]): list of gamma parameters corresponding to the second bag
                  k(int64): degree
        Returns:
                 d1(float64[:]): delta parameters for the sampled post measurement states
                 g1(float64[:]):  gamma parameters for the sampled post measurement states

  '''
  if k==1:
    return d,g
  else:
    d1,g1=bitnode_vec_jit(d,d,g,g,pr_vec,perm)
    if k>2:
      for i in range(k-2):
        d1,g1=bitnode_vec_jit(d,d1,g,g1,pr_vec,perm)
    return d1,g1


@njit
def checknode_power_jit(d:float64[:],g:float64[:],k:int64,pr_vec=None,perm=None)->(float64[:],float64[:]):
  '''
    code to apply checknode operation with given checknode degree on two bags with delta gamma parameters
    and sample post measurement delta gamma parameters, chooses delta, gamma parameters randomly from the bag.
    Arguments:
                  d1(float64[:]):  list of delta parameters corresponding to the first bag
                  d2(float64[:]): list of gamma parameters corresponding to the first bag
                  g1(float64[:]) : list of delta parameters corresponding to the second bag
                  g2(float64[:]): list of gamma parameters corresponding to the second bag
                  k(int64): degree
        Returns:
                 d1(float64[:]): delta parameters for the sampled post measurement states
                 g1(float64[:]):  gamma parameters for the sampled post measurement states

  '''
  if k==1:
    return d,g
  else:
    d1,g1=checknode_vec_jit(d,d,g,g,pr_vec,perm)
    if k>2:
      for i in range(k-2):
        d1,g1=checknode_vec_jit(d,d1,g,g1,pr_vec,perm)
    return d1,g1

@njit
def polar(n:int64,d:float64[:],g=float64[:])->(float64[:,:,:]):
  '''
   code to compute density evolution PMBPQM based SCD polar decoder over BSCQ channels

         Arguments:
               n(int64): number of polar stages
               d(float64[:]): array with delta parameter of the channel, length corresponds to number of samples
               g(float64[:]): array with gamma parameter of the channel, length corresponds to number of samples

        Returns:
               old_stage(float[:,:,:]): list that contains delta and gamma parameters of channels synthesized after n stages of polar recursion,
                                        i th element in the list contains delta, gamma parameters of the i th channel at the n th stage
  '''
 # s=0
  #for i in range(n+1):
  #  s=s+2**i

  old_stage=np.zeros(shape=(1,2,len(d)),dtype=float64)
  old_stage[0][0],old_stage[0][1]=d,g
 # o=np.zeros(shape=(s,2,len(x)))
  for i in range(n):
    new_stage=np.zeros(shape=(2**(i+1),2,len(d)))
    for j in range(2**(i)):
      new_stage[2*j][0],new_stage[2*j][1]=checknode_vec_jit(old_stage[j][0],old_stage[j][0],old_stage[j][1],old_stage[j][1])
      new_stage[2*j+1][0],new_stage[2*j+1][1]=bitnode_vec_jit(old_stage[j][0],old_stage[j][0],old_stage[j][1],old_stage[j][1])
    old_stage=new_stage
  return old_stage


def polar_design(biterrd,d):
    '''
    Design polar code from additive channel noise scores

          Arguments:
                  biterrd (float[:]): Numpy array of channel noise scores (e.g., error rates)
                  d (float): Sum constraint of total noise for unfrozen channels

          Returns:
                  f (float[:]): Numpy array indicating frozen bits by 0 and info bits by 0.5
    '''
    # Sort into increasing order and compute cumulative sum
    order = np.argsort(biterrd)
    SE = biterrd[order]
    CSE = np.cumsum(SE)

    # Find best frozen bits
    k = np.sum(CSE<d)
    f = np.zeros(len(biterrd))
    f[order[:k]] = 0.5
    return f

def bitnode_gen_unitary(d1,d2,g1,g2):
  '''
    Generate bitnode unitary for two BSCQ channels

    Arguments:
                  d1(float): parameter delta for the first BSCQ channel
                  d2(float): parameter delta for the second BSCQ channel
                  g1(float) : parameter gamma for the first BSCQ channel
                  g2(float): parameter gamma for the second  BSCQ channel


          Returns:
                  vb(float[:,:]): Numpy 2D array
  '''

  rhofull = np.kron(rhom(d1,g1,0),rhom(d2,g2,0))-np.kron(rhom(d1,g1,1),rhom(d2,g2,1))

  evals, evecs = LA.eigh(rhofull)
  # fix eigenvector v0
  v0 = evecs[:,0]

  # symmetry operator Un = np.kron(pauli(1),pauli(1))
  Un = np.kron(pauli(1),pauli(1))
  # check if the second eigenvector evecs[:,1] is orthogonal to Un@evecs[:,1]
  x = evecs[:,1]@(Un@evecs[:,1])

  #print("evals")
  #print(evals)
  if np.abs(x)<10e-10:
    v1 = evecs[:,1]
  # if not orthogonal, combine evecs[:,1], evecs[:,2] to create v1 s.t. v1@(Un@v1)= 0
  if np.abs(x)>=10e-10:
    vec1, vec2 = evecs[:,1], evecs[:,2]
    b11, b12, b21, b22 = np.inner(vec1, (Un@vec1).conj()), np.inner(vec2, (Un@vec1).conj()), np.inner(vec1, (Un@vec2).conj()), np.inner(vec2, (Un@vec2).conj())

    alpha = (-b12-b21-np.sqrt((b12+b21)**2-4*b11*b22))/(2*b22)
    v1 = vec1+alpha*vec2
    v1 = v1/np.sqrt(v1@v1)
  vb=np.zeros((4,4))
  vb[:,0]=Un@v0
  vb[:,1]=Un@v1
  vb[:,2]=v0
  vb[:,3]=v1
  return np.transpose(vb)

def checknode_gen_unitary(d1,d2,g1,g2):
  '''
   Generate checknode unitary for two BSCQ channels
        Returns:
                  vcf(float[:,:]): Numpy 2D array
  '''
  v0 = 1/np.sqrt(2)*np.array([1,0,0,1])
  v1 = 1/np.sqrt(2)*np.array([-1, 0, 0, 1])
  ux=np.kron(pauli(1),np.eye(2))
  vcf=np.zeros((4,4))
  vcf[:,0]=v0
  vcf[:,1]=v1
  vcf[:,2]=ux@v0
  vcf[:,3]=ux@v1
  return np.transpose(vcf)

CU=checknode_gen_unitary(0.1,0.1,0,0)

def apply_permutation(qubit_list,number_qubits,conditional_list1=[],conditional_list2=[]):
  '''
     Code to generate permuatation and inverse permutation to swap quantum states over
     different quantum systems to apply unitary

         Arguments:
               qubit_list(int[:]): list of indices of systems where unitary to be applied
               number_qubits(int): total number of quantum systems
               conditional_list1(int[:], default=empty array): conditional list of indices of systems for the first system in the qubit_list
               conditional_list2(int[:], default=empty array): conditional list of indices of systems for the second system in the qubit_list

         Returns:
               p(int[:]): list containing destination indices for the desired permutation
               pi(int[:]): list containing destination indices for the inverse of corresponding permutation

  '''

  '''
   convention: the permuatiton qubits listed in conditional_list1 in the front, followed by qubits listed in conditional_list2,
               qubits listed in the qubit_list are pushed to the end
  '''
  n=number_qubits
  l=len(qubit_list)
  l1=len(conditional_list1)
  l2=len(conditional_list2)
  p=np.zeros(n,dtype=int)
  for j in range(l):
    p[qubit_list[j]]=l1+l2++j
  for j in range(l1):
    p[conditional_list1[j]]=j
  for j in range(l2):
    p[conditional_list2[j]]=l1+j
  c=0
  for j in range(n):
    if j not in qubit_list and j not in conditional_list1 and j not in conditional_list2:
      p[j]=l+l1+l2+c
      c=c+1
  p1=np.zeros((len(p),len(p)))
  for i in range(len(p)):
    p1[:,i][p[i]]=1
  p2=p1.T
  pi=np.zeros(n,dtype=int)
  for i in range(n):
    pi[i]=int(np.where(p1.T[:,i]==1)[0][0])
  return p,pi
def apply_unitary(qubit_list,unitary,q_state,number_qubits):
  '''
      Code to apply unitary on given quantum systems
          Arguments:
                 qubit_list(int[:]):  list of indices of systems where unitary to be applied
                 unitary(float[:,:]): unitary to be applied
                 q_state(float[:]): joint qubit state
                 number_qubits(int): total number of quantum systems
         Returns:
                 q_out(float[:]): final qubit state after applying the unitary

  '''
  n=number_qubits
  l=len(qubit_list)
  u=unitary
  q=np.reshape(q_state,2*np.ones(n,dtype=int))
  p,pi=apply_permutation(qubit_list,n)
  q1=np.moveaxis(q,np.array(range(n)),p)
  q2=np.reshape(q1,(2**l,2**(n-l)))
  '''
    this function utilizes the broadcasting feature of numpy array
  '''
  q_out=u@q2
  q_out=np.reshape(q_out,2*np.ones(n,dtype=int))
  q_out=np.moveaxis(q_out,np.array(range(n)),pi)
  q_out=np.reshape(q_out,np.shape(q_state))
  return q_out

def apply_unitary_inverse(qubit_list,unitary,q_state,number_qubits):
  '''
      Code to apply inverse of a unitary on given quantum systems
          Arguments:
                 qubit_list(int[:]):  list of indices of systems where unitary to be applied
                 unitary(float[:,:]): unitary whose inverse to be applied
                 q_state(float[:]): joint qubit state
                 number_qubits(int): total number of quantum systems
         Returns:
                 q_out(float[:]): final qubit state after applying the inverse of the unitary

  '''
  n=number_qubits
  l=len(qubit_list)
  u=unitary
  q=np.reshape(q_state,2*np.ones(n,dtype=int))
  p,pi=apply_permutation(qubit_list,n)
  q1=np.moveaxis(q,np.array(range(n)),p)
  q2=np.reshape(q1,(2**l,2**(n-l)))
  q_out=np.transpose(u)@q2
  q_out=np.reshape(q_out,2*np.ones(n,dtype=int))
  q_out=np.moveaxis(q_out,np.array(range(n)),pi)
  q_out=np.reshape(q_out,np.shape(q_state))
  return q_out

def apply_conditional_unitary(qubit_list,conditional_list1,conditional_list2,dg_list1,dg_list2,q_state,number_qubits):
  '''
      Code to apply conditional bitnode unitary on given quantum systems conditioned on quantum states of other quantum systems
          Arguments:
                 qubit_list(int[:]):  list of indices of systems where unitary to be applied
                 conditional_list1(int[:]):conditional list of indices of systems for the first system in the qubit_list
                 conditional_list2(int[:]): conditional list of indices of systems for the second system in the qubit_list
                 dg_list1(float[:,:]): list of delta and gamma parameters for the quantum states corresponding to quantum systems in conditional_list1,
                                       i th element corresponds to  delta and gamma parameter for quantum state of i th quantum system
                                       in  conditional_list1
                 dg_list2(float[:,:]): list of delta and gamma parameters for the quantum states corresponding to quantum systems in conditional_list2,
                                       i th element corresponds to  delta and gamma parameter for quantum state of i th quantum system in
                                       conditional_list2
                 q_state(float[:]): joint qubit state
                 number_qubits(int): total number of quantum systems
         Returns:
                 q_out(float[:]): final qubit state after applying conditional bitnode unitary

  '''
  n=number_qubits
  l=len(qubit_list)
  l1=len(conditional_list1)
  l2=len(conditional_list2)
  q=np.reshape(q_state,2*np.ones(n,dtype=int))
  p=np.zeros(n,dtype=int)
  p,pi=apply_permutation(qubit_list,n,conditional_list1,conditional_list2)
  q1=np.moveaxis(q,np.array(range(n)),p)
  q2=np.reshape(q1,(2**l1,2**l2,2**l,2**(n-l1-l2-l)))
  q_out=np.zeros((2**l1,2**l2,2**l,2**(n-l1-l2-l)))
  for i in range(2**l1):
    for j in range(2**l2):
      q_out[i,j,:]=bitnode_gen_unitary(dg_list1[i][0],dg_list2[j][0],dg_list1[i][1],dg_list2[j][1])@q2[i,j,:]
  q_out=np.reshape(q_out,2*np.ones(n,dtype=int))
  q_out=np.moveaxis(q_out,np.array(range(n)),pi)
  q_out=np.reshape(q_out,np.shape(q_state))
  return q_out
def apply_conditional_unitary_inverse(qubit_list,conditional_list1,conditional_list2,dg_list1,dg_list2,q_state,number_qubits):
  '''
      Code to apply inverse of conditional bitnode unitary on given quantum systems conditioned on quantum states of other quantum systems
          Arguments:
                 qubit_list(int[:]):  list of indices of systems where unitary to be applied
                 conditional_list1(int[:]):conditional list of indices of systems for the first system in the qubit_list
                 conditional_list2(int[:]): conditional list of indices of systems for the second system in the qubit_list
                 dg_list1(float[:,:]): list of delta and gamma parameters for the quantum states corresponding to quantum systems in conditional_list1,
                                       i th element corresponds to  delta and gamma parameter for quantum state of i th quantum system
                                       in  conditional_list1
                 dg_list2(float[:,:]): list of delta and gamma parameters for the quantum states corresponding to quantum systems in conditional_list2,
                                       i th element corresponds to  delta and gamma parameter for quantum state of i th quantum system
                                       in  conditional_list2
                 q_state(float[:]): joint qubit state
                 number_qubits(int): total number of quantum systems
         Returns:
                 q_out(float[:]): final qubit state after applying conditional bitnode unitary

  '''
  n=number_qubits
  l=len(qubit_list)
  l1=len(conditional_list1)
  l2=len(conditional_list2)
  q=np.reshape(q_state,2*np.ones(n,dtype=int))
  p=np.zeros(n,dtype=int)
  p,pi=apply_permutation(qubit_list,n,conditional_list1,conditional_list2)
  q1=np.moveaxis(q,np.array(range(n)),p)
  q2=np.reshape(q1,(2**l1,2**l2,2**l,2**(n-l1-l2-l)))
  q_out=np.zeros((2**l1,2**l2,2**l,2**(n-l1-l2-l)))
  for i in range(2**l1):
    for j in range(2**l2):
      q_out[i,j,:]=np.transpose(bitnode_gen_unitary(dg_list1[i][0],dg_list2[j][0],dg_list1[i][1],dg_list2[j][1]))@q2[i,j,:]
  q_out=np.reshape(q_out,2*np.ones(n,dtype=int))
  q_out=np.moveaxis(q_out,np.array(range(n)),pi)
  q_out=np.reshape(q_out,np.shape(q_state))
  return q_out

def sample_state(l,v):
  '''
     Code to sample quantum state for a given ensemble with two quantum states

        Arguments:
                 l(float[:]): i th element corresponds to probability of sampling i th quantum state from the ensemble
                 v(float[:,:]): i th column represents the i th quantum state
        Returns:
                 sampled state
  '''
  o=int(np.random.random()>l[0])
  if o==0:
    return v[:,0]
  else:
    return v[:,1]

def sample_joint_state(l,v,number_qubits):
  '''
     Code to sample joint quantum state of multiple qubits for a given ensemble with two quantum states

        Arguments:
                 l(float[:]): i th element corresponds to probability of sampling i th quantum state from the ensemble
                 v(float[:,:]): i th column represents the i th quantum state
                 number_qubits(int): total number of qubits in the joint state
        Returns:
                (float[:]): sampled joint state
  '''
  if number_qubits==1:
    return sample_state(l,v)
  else:
    return np.kron(sample_joint_state(l,v,number_qubits-1),sample_state(l,v))


def sample_joint_random_state(l,v,v1,number_qubits,bit_sequence):
  '''
     Code to sample joint quantum state of multiple qubits for given ensembles with two quantum states

        Arguments:
                 l(float[:]): i th element corresponds to probability of sampling i th quantum state from the ensembles
                 v(float[:,:]): i th column represents the i th quantum state of the first ensemble
                 v1(float[:,:]): i th column represents the i th quantum state of the second ensemble
                 number_qubits(int): total number of qubits in the joint state
                 bit_sequence(int[:]): binary sequence, 0 indicates sampling state from the first ensemble,
                                       1 indicates sampling state from the second ensemble
        Returns:
                 (float[:]): sampled joint state
  '''
  b=bit_sequence
  if b[0]==0:
    s=sample_state(l,v)
  else:
    s=sample_state(l,v1)
  for i in range(1,number_qubits):
    if b[i]==0:
      s=np.kron(s,sample_state(l,v))
    else:
      s=np.kron(s,sample_state(l,v1))
  return s

def cnop(quantum_state,informtion_qubit1,information_qubit2,conditional_list1,conditional_list2,number_qubits):
  '''
     Code to apply check-node operation on two qubits of a joint quantum state

        Arguments:
                 quantum_state(float[:]): joint quantum state
                 informtion_qubit1(int): index of the first qubit where check-node operation to be applied
                 informtion_qubit2(int): index of the second qubit where check-node operation to be applied
                 conditional_list1([int[:],float[:,:]]):first element contains conditional list of indices of systems
                                                        for the informtion_qubit1,
                                                        second element contains corresponding delta and gamma parameters for corresponding
                                                        qubits
                 conditional_list2([int[:],float[:,:]]):first element contains conditional list of indices of systems
                                                        for the informtion_qubit2,
                                                        second element contains corresponding delta and gamma parameters for corresponding
                                                        qubits
                 number_qubits(int): total number of qubits in the joint state

        Returns:
                 q_out(float[:]): output state after check-node operation where all information is in information_qubit1,
                                  reliability in information_qubit2
                 conditional_list([int[:],float[:,:]): updated conditional list for informtion_qubit1

  '''
  qubit_list=np.array([informtion_qubit1,information_qubit2])
  q_out=apply_unitary(qubit_list,CU,quantum_state,number_qubits)
  l1=len(conditional_list1[0])
  l2=len(conditional_list2[0])
  conditional_list=[[information_qubit2],[]]
  c1=[]
  c2=[]
  for i in range(2**l1):
    for j in range(2**l2):
      rc=checknode(conditional_list1[1][i][0],conditional_list2[1][j][0],conditional_list1[1][i][1],conditional_list2[1][j][1])[0]
      c1.append(rc[0])
      c2.append(rc[1])
  conditional_list[1]=c1+c2

  for i in range(l1):
    conditional_list[0].append(conditional_list1[0][i])
  for j in range(l2):
    conditional_list[0].append(conditional_list2[0][j])
  return q_out, conditional_list

def cnop_inverse(quantum_state,informtion_qubit1,information_qubit2,conditional_list1,conditional_list2,number_qubits):
  '''
     Code to apply inverse of the check-node operation on two qubits of a joint quantum state

        Arguments:
                 quantum_state(float[:]): joint quantum state
                 informtion_qubit1(int): index of the first qubit where check-node operation to be applied
                 informtion_qubit2(int): index of the second qubit where check-node operation to be applied
                 conditional_list1([int[:],float[:,:]]):first element contains conditional list of indices of systems
                                                        for the informtion_qubit1,
                                                        second element contains corresponding delta and gamma parameters for corresponding
                                                        qubits
                 conditional_list2([int[:],float[:,:]]):first element contains conditional list of indices of systems
                                                        for the informtion_qubit2,
                                                        second element contains corresponding delta and gamma parameters for corresponding
                                                        qubits
                 number_qubits(int): total number of qubits in the joint state

        Returns:
                 q_out(float[:]): output state after check-node operation where all information is in information_qubit1,
                                  reliability in information_qubit2

  '''
  qubit_list=np.array([informtion_qubit1,information_qubit2])
  q_out=apply_unitary_inverse(qubit_list,CU,quantum_state,number_qubits)
  return q_out

def vnop(quantum_state,informtion_qubit1,information_qubit2,conditional_list1,conditional_list2,number_qubits):
  '''
     Code to apply bit-node operation on two qubits of a joint quantum state

        Arguments:
                 quantum_state(float[:]): joint quantum state
                 informtion_qubit1(int): index of the first qubit where check-node operation to be applied
                 informtion_qubit2(int): index of the second qubit where check-node operation to be applied
                 conditional_list1(list[int[:],float[:,:]]):first element contains conditional list of indices of systems
                                                        for the informtion_qubit1,
                                                        second element contains corresponding delta and gamma parameters for corresponding
                                                        qubits
                 conditional_list2(list[int[:],float[:,:]]):first element contains conditional list of indices of systems
                                                        for the informtion_qubit2,
                                                        second element contains corresponding delta and gamma parameters for corresponding
                                                        qubits
                 number_qubits(int): total number of qubits in the joint state

        Returns:
                 q_out(float[:]): output state after bit-node operation where all information is in information_qubit1,
                                  reliability in information_qubit2
                 conditional_list([int[:],float[:,:]): updated conditional list for informtion_qubit1

  '''
  qubit_list=np.array([informtion_qubit1,information_qubit2])
  l1=len(conditional_list1[0])
  l2=len(conditional_list2[0])
  if l1==0 and l2==0:
    q_out=apply_unitary(qubit_list,bitnode_gen_unitary(conditional_list1[1][0][0],conditional_list2[1][0][0],conditional_list1[1][0][1],conditional_list2[1][0][1]),quantum_state,number_qubits)
  else:
    q_out=apply_conditional_unitary(qubit_list,conditional_list1[0],conditional_list2[0],conditional_list1[1],conditional_list2[1],quantum_state,number_qubits)
  conditional_list=[[information_qubit2],[]]
  c1=[]
  c2=[]
  for i in range(2**l1):
    for j in range(2**l2):
      rc=bitnode(conditional_list1[1][i][0],conditional_list2[1][j][0],conditional_list1[1][i][1],conditional_list2[1][j][1])[0]
      c1.append(rc[0])
      c2.append(rc[1])
  conditional_list[1]=c1+c2

  for i in range(l1):
    conditional_list[0].append(conditional_list1[0][i])
  for j in range(l2):
    conditional_list[0].append(conditional_list2[0][j])
  return q_out, conditional_list

def vnop_inverse(quantum_state,informtion_qubit1,information_qubit2,conditional_list1,conditional_list2,number_qubits):
  '''
     Code to apply inverse of the bit-node operation on two qubits of a joint quantum state

        Arguments:
                 quantum_state(float[:]): joint quantum state
                 informtion_qubit1(int): index of the first qubit where check-node operation to be applied
                 informtion_qubit2(int): index of the second qubit where check-node operation to be applied
                 conditional_list1(list[int[:],float[:,:]]):first element contains conditional list of indices of systems
                                                        for the informtion_qubit1,
                                                        second element contains corresponding delta and gamma parameters for corresponding
                                                        qubits
                 conditional_list2(list[int[:],float[:,:]]):first element contains conditional list of indices of systems
                                                        for the informtion_qubit2,
                                                        second element contains corresponding delta and gamma parameters for corresponding
                                                        qubits
                 number_qubits(int): total number of qubits in the joint state

        Returns:
                 q_out(float[:]): output state after bit-node operation where all information is in information_qubit1,
                                  reliability in information_qubit2

  '''
  qubit_list=np.array([informtion_qubit1,information_qubit2])
  l1=len(conditional_list1[0])
  l2=len(conditional_list2[0])
  if l1==0 and l2==0:
    q_out=apply_unitary_inverse(qubit_list,bitnode_gen_unitary(conditional_list1[1][0][0],conditional_list2[1][0][0],conditional_list1[1][0][1],conditional_list2[1][0][1]),quantum_state,number_qubits)
  else:
    q_out=apply_conditional_unitary_inverse(qubit_list,conditional_list1[0],conditional_list2[0],conditional_list1[1],conditional_list2[1],quantum_state,number_qubits)
  return q_out

def measure_first_qubit(quantum_state,number_qubits,projection_operator):
  '''
    Code to compute error probability after measuring first qubit of joint qubit state in some arbitrary basis
     Arguments:
                 quantum_state(float[:]): joint quantum state
                 number_qubits(int): total number of qubits in the joint state
                 projection_operator(float[:,:]): projection operator to be applied on the first qubit


     Returns:
              (float): measurement success probability


  '''
  q=quantum_state
  n=number_qubits
  pi=projection_operator
  q1=np.reshape(q,(2,2**(n-1)))
  q2=pi@q1
  q3=np.reshape(q2,2**n)
  return q3@q


def apply_projection_first_qubit(quantum_state,number_qubits,projection_operator):
  '''
    Code to apply projection operator on the first qubit of joint qubit state
     Arguments:
                 quantum_state(float[:]): joint quantum state
                 number_qubits(int): total number of qubits in the joint state
                 projection_operator(float[:,:]): projection operator to be applied on the first qubit


     Returns:
               q3(float[:]): final state


  '''
  q=quantum_state
  n=number_qubits
  pi=projection_operator
  q1=np.reshape(q,(2,2**(n-1)))
  q2=pi@q1
  q3=np.reshape(q2,2**n)
  return q3
def apply_flip(q_state,number_qubits,index):

  '''
    Code to apply Pauli X on a qubit of joint quantum state
    Arguments:
                 q_state(float[:]): joint quantum state
                 number_qubits(int): total number of qubits in the joint state
                 index(int): index of the qubit in the joint state
    Returns:
               q_out(float[:]): final state


  '''
  n=number_qubits
  x=np.array([[0,1],[1,0]])
  p=np.array(range(n))
  p[0]=index
  p[index]=0
  q=np.reshape(q_state,2*np.ones(n,dtype=int))
  q1=np.moveaxis(q,np.array(range(n)),p)
  q2=np.reshape(q1,(2,2**(n-1)))
  q_out=x@q2
  q_out=np.reshape(q_out,2*np.ones(n,dtype=int))
  q_out=np.moveaxis(q_out,np.array(range(n)),p)
  q_out=np.reshape(q_out,np.shape(q_state))
  return q_out

def hard_decision_flip(q_state,number_qubits,hard_decision,index):
  '''
    Code to apply Pauli X depending on the hard decision outcome on a qubit of joint quantum state
    Arguments:
                 q_state(float[:]): joint quantum state
                 number_qubits(int): total number of qubits in the joint state
                 hard_decision(int): binary, 1 indicates apply Pauli X
                 index(int): index of the qubit in the joint state
    Returns:
                 (float[:]): final state


  '''
  if hard_decision==1:
    return apply_flip(q_state,number_qubits,index)
  else:
    return q_state



@njit('(int64[:])(int64[:])') # Input/output specifications to make Numba work
def polar_transform(u):
    '''
    Encode polar information vector u

          Arguments:
                  u (int64[:]): Numpy array of input bits

          Returns:
                  x (int64[:]): Numpy array of encoded bits
    '''
    # Recurse down to length 1
    if (len(u)==1):
        return u;
    else:
        # R_N maps odd/even indices (i.e., u1u2/u2) to first/second half
        # Compute odd/even outputs of (I_{N/2} \otimes G_2) transform
        x = np.zeros(len(u), dtype=np.int64)
        x[:len(u)//2] = polar_transform((u[::2]+u[1::2])%2)
        x[len(u)//2:] = polar_transform(u[1::2])
        return x

def polar_decoder_cq_output(yi,yd,quantum_state,number_qubits,info_set,codeword):
  '''
       Successive cancellation decoder for binary polar codes on symmetric CQ channels. This can be seen as a recursive classical algorithm
       that controls a quantum computer to implement the paired-measurement BPQM polar decoder.
       Arguments:
               yi(int[:]):qubit indices for the code bits
               yd (List[(int[:],float[:])]): list of bit conditional channel parameter
               quantum_state(float[:]): joint quantum state
               number_qubits(int):total number of qubits in the joint state
               info_set(int[:]):Information set, 1 indicates information bit, 0 indicates frozen bit
               codeword(int[:]): the value of frozen bits

        Returns:
               out(int[:]): input hard dec in input order
               q:quantum state after applying passing through the decoder and applying measurements

  '''
  q=quantum_state
  N=len(yi)

  pi=np.outer(qubit(0),qubit(0))
  '''
   recurse down to length 1,

   if the bit is in the information set apply Helstrom measurement,
   depending on hard decision outcome apply Pauli X on the qubit,

   if the bit is frozen, depending the value apply Pauli X on the qubit

  '''
  f=info_set
  if N==1:
    if f[0]==1:
        p1=measure_first_qubit(q,number_qubits,pi)
        o=int(np.random.random()>p1)
        if o==0:
            q=apply_projection_first_qubit(q,number_qubits,pi)
            return o,q
        else:
            q=apply_projection_first_qubit(q,number_qubits,np.eye(2)-pi)
            q=apply_flip(q,number_qubits,0)
            return o,q
    else:
        q=hard_decision_flip(q,number_qubits,codeword[0],0)
        return codeword[0],q

  yd1=[]
  '''
    Apply check node operation to go back one stage and estimate u1
  '''
  for i in range(0,N,2):
    q,list1=cnop(q,yi[i],yi[i+1],yd[i],yd[i+1],number_qubits)
    yd1.append(list1)

  '''
   R_NˆT maps u1est to top polar code
  '''
  out1,q=polar_decoder_cq_output(yi[::2],yd1,q,number_qubits,f[:(N//2)],codeword[:(N//2)])


  for i in range(0,N,2):
    q=cnop_inverse(q,yi[i],yi[i+1],yd[i],yd[i+1],number_qubits)
  '''
      Invert check node unitary operation
  '''
  yd2=[]
  '''
    Apply bit node operation to go back one stage and estimate u2
  '''
  for i in range(0,N,2):
    q,list2=vnop(q,yi[i],yi[i+1],yd[i],yd[i+1],number_qubits)
    yd2.append(list2)
  '''
   R_NˆT maps u2est to bottom polar code
  '''
  out2,q=polar_decoder_cq_output(yi[::2],yd2,q,number_qubits,f[(N//2):],codeword[(N//2):])
  '''
   Invert bitnode unitary operation
  '''
  for i in range(0,N,2):
    q=vnop_inverse(q,yi[i],yi[i+1],yd[i],yd[i+1],number_qubits)

  out=np.hstack([out1,out2])
  return out,q
def sample_codeword(info_set,frozen_bits):
    '''
      Code to sample input bits for polar encoder for given information set and values of frozen bits

        Arguments:
                info_set(int[:]):information set, 1 indicates information bit, 0 indicates frozen bit
                frozen_bits(int[:]): values of frozen bits
        Returns:
                c(int[:]): sampled sequence
    '''
    l=len(info_set)
    c=np.random.randint(high=2,low=0,size=l)
    o=0
    for i in range(l):
        if info_set[i]==0:
            c[i]=frozen_bits[o]
            o=o+1
    return c

def polar_decoder_cq_sample_output_avg_error(dp,gp,number_samples,number_qubits,info_set,frozen_bits):
  '''
   Code to compute channel error rate and block error rate
    Arguments:
                dp(float): delta parameter of the BSCQ channel
                gp(float): gamma parameter of the BSCQ channel
                number_samples(int): number of samples used to compute error rate
                number_qubits(int):total number of qubits in the joint state
                info_set(int[:]):information set, 1 indicates information bit, 0 indicates frozen bit
                frozen_bits(int[:]): values of frozen bits
        Returns:
                h(float[:]): channel error rates in input order
                block(float): block error rate

  '''
  r=rhom(dp,gp,0)

  x=np.array([[0,1],[1,0]])
  l,v=LA.eigh(r)
  l1,v1=LA.eigh(x@r@x)
  list1=[[],[[dp,gp]]]

  yi=np.array(range(number_qubits),dtype=int)
  yd=[]
  for i in range(number_qubits):
    yd.append(list1)

  o=0
  dec_codeword=np.zeros(len(info_set))
  for i in range(len(info_set)):
        if info_set[i]==0:
            dec_codeword[i]=frozen_bits[o]
            o=o+1
  h=np.zeros(number_qubits)
  block=0

  for i in range(number_samples):
    codeword=sample_codeword(info_set,frozen_bits)
    encoded_bits=polar_transform(np.array(codeword))
    q=sample_joint_random_state(l,v,v1,number_qubits,encoded_bits)
    h1,q1=polar_decoder_cq_output(yi,yd,q,number_qubits,info_set,dec_codeword)
    h=h+np.mod(h1+codeword,2)
    if np.sum(np.mod(h1+codeword,2))>=1:
        block=block+1

  return h/number_samples,block/number_samples
def codeword_check(output,codeword):
    '''
      code to bit index where error happened for the first time in the codeword (from MSB to LSB)

        Arguments:
                codeword(int[:]): input bits
                output(int[:]): output of the decoder
        Returns:
                (int): index of the error
    '''
    e=np.mod(output+codeword,2)
    if np.sum(e)==0:
        return len(e)
    for j in range(0,len(e)):
        if e[j]==1:
            return j
def polar_decoder_cq_sample_output_random_frozen_avg_error(dp,gp,number_samples,number_qubits,info_set,num_frozen):
  '''
   Code to compute channel error rate and block error rate for input with randomly chosen frozen bits
    Arguments:
                dp(float): delta parameter of the BSCQ channel
                gp(float): gamma parameter of the BSCQ channel
                number_samples(int): number of samples used to compute error rate
                number_qubits(int):total number of qubits in the joint state
                info_set(int[:]):information set, 1 indicates information bit, 0 indicates frozen bit
                num_frozen: number of frozen bits
        Returns:
                h(float[:]): channel error rates in input order
                block(float): block error rate
                ber_num(int[:]): number of times error occured for the first times for the qubit indices
                ber_den(int[:]):number of times no error occured in the previous qubits

  '''

  '''
     ber_num[i]/ber_den[i] gives error rate for i th qubit which is not caused by wrong measurement outcome of previous qubits,
     it gives lower bound on the individual channel error rate,
     it also gives insight about error rate due to the disturbance caused by measurements on the previous qubits
  '''
  r=rhom(dp,gp,0)

  x=np.array([[0,1],[1,0]])
  l,v=LA.eigh(r)
  l1,v1=LA.eigh(x@r@x)
  list1=[[],[[dp,gp]]]

  yi=np.array(range(number_qubits),dtype=int)
  yd=[]
  for i in range(number_qubits):
    yd.append(list1)


  h=np.zeros(number_qubits)
  block=0
  ber_num=np.zeros(number_qubits)
  ber_den=np.zeros(number_qubits)

  for i in range(number_samples):
    frozen_bits=np.random.randint(high=2,low=0,size=num_frozen)
    o=0
    dec_codeword=np.zeros(len(info_set))
    for j in range(len(info_set)):
        if info_set[j]==0:
            dec_codeword[j]=frozen_bits[o]
            o=o+1

    codeword=sample_codeword(info_set,frozen_bits)

    encoded_bits=polar_transform(np.array(codeword,dtype=int64))
    q=sample_joint_random_state(l,v,v1,number_qubits,encoded_bits)
    h1,q1=polar_decoder_cq_output(yi,yd,q,number_qubits,info_set,dec_codeword)
    h=h+np.mod(h1+codeword,2)
    if np.sum(np.mod(h1+codeword,2))>=1:
        block=block+1

    e=codeword_check(h1,codeword)

    if e==number_qubits:
        ber_den=ber_den+1
    else:
        ber_num[e]=ber_num[e]+1
        for k in range(0,e+1):
            ber_den[k]=ber_den[k]+1
  return h/number_samples,block/number_samples,ber_num,ber_den


def main():
  #dp=0.05
  #gp=0.15
  #n=3
  #num_de_samples=20000
  #num_samples=50
  #num_qubits=2**n
  #info_set=[0,0,0,1,0,1,1,1]
  #num_frozen=4
  d=np.ones(num_de_samples)*delta
  g=np.ones(num_de_samples)*gamma
  p=polar(n,d,g)
  D=[]
  print(f'Results for lenth-{2**n} polar code over BSCQ with d={delta},g={gamma} ')
  #f=polar_design(ww,0.001)
  #print('rate for code with given design constraint:',np.mean(f==0.5))
  for i in range(2**n):
    d=np.mean(p[i][0])
    g=np.mean(p[i][1])
    r1=rhom(d,g,0)
    hel=1-helstrom_success(r1,pauli(1))
    D.append(hel)
  print('Density Evolution Error Rate :\n',D)
  h,block,ber_num,ber_den=polar_decoder_cq_sample_output_random_frozen_avg_error(delta,gamma,M,num_qubits,info_set,num_frozen)
  print('Simulated Channel Error Rate:',h)
  print('Block Error Rate:',block)
  print('BER_NUM:',ber_num)
  print('BER_DEN:',ber_den)

if __name__ =="__main__":
  # Setup parser
  parser = ap.ArgumentParser('Prints Block Error Rate for Length-16 CQ Polar')
  parser.add_argument('--verbose', '-v', help='Display text output', action="store_true")
  parser.add_argument('-d', dest='delta', type=float, default=0.05, help='Delta')
  parser.add_argument('-g', dest='gamma', type=float, default=0.15, help='gamma')
  parser.add_argument('-N', dest='num_qubits', type=int, default=8, help='Block Length')
  parser.add_argument('-n', dest='n', type=int, default=3, help='polar stages')
  parser.add_argument('-M', dest='M', type=int, default=100, help='Number of Blocks')
  parser.add_argument('-nde', dest='num_de_samples', type=int, default=100, help='Number of DE samples')
  parser.add_argument('-s','--seed', dest='seed', type=int, default=None, help='Seed for RNG')
  # parse arguments
  args = parser.parse_args()
  # Random seed to generate samples
  if (args.seed is None):
    vars(args).update({"seed":int(time.time())%65536})
  # Instantiate arguments as local variables for simplicity
  locals().update(vars(args))
  # Simulation loop
  rng = np.random.RandomState(seed)
  if num_qubits==4:
    info_set=[0,0,1,1]
    #frozen_val=[0,1]
    num_frozen=2
  if num_qubits==8:
    info_set=[0,0,0,1,0,1,1,1]
    #frozen_val=[1,0,1,0]
    num_frozen=4
  if num_qubits==16:
    info_set=[0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1]
   # frozen_val=[1,0,1,0,1,0,1,0]
    num_frozen=8
  main()
