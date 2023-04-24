#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 22:10:29 2022

@author: sen
"""

import torch 
import numpy as np

#---------- Albert Gus' values
Gu_A = np.array([[ -1.,          0.,          0.,          0.,          0.,          0.,      0.,          0.       ],
                 [ -1.7320508,  -1.9999999,   0.,          0.,          0.,          0.,      0.,          0.       ],
                 [ -2.236068,   -3.8729835,  -3.,          0.,          0.,          0.,      0.,          0.       ],
                 [ -2.6457512,  -4.582576,   -5.9160795,  -4.,          0.,          0.,      0.,          0.       ],
                 [ -3.,         -5.196152,   -6.708204,   -7.9372544,  -5.,          0.,      0.,          0.       ],
                 [ -3.3166249,  -5.7445626,  -7.4161983,  -8.774965,   -9.949875,   -6.0000005,            0.,          0.],
                 [ -3.6055512,  -6.244998,   -8.062258,   -9.539392,  -10.816654,  -11.958261,    -7.,          0.       ],
                 [ -3.8729835,  -6.708204,   -8.6602545, -10.246951,  -11.61895,   -12.845233,  -13.964241,   -8.       ]])

Gu_B = np.array([[1.       ],
                 [1.7320508],
                 [2.236068 ],
                 [2.6457512],
                 [3.       ],
                 [3.3166249],
                 [3.6055512],
                 [3.8729835]])

Gu_A_stacked_GBT = np.array([[[ 3.3333e-01, -2.2247e-16, -1.2406e-15, -1.0861e-15, -9.6030e-16,
          -8.9707e-16, -1.0764e-15, -6.4121e-17],
         [-5.7735e-01,  2.9802e-08, -4.9960e-16, -1.1102e-16,  6.6613e-16,
           6.6613e-16, -5.5511e-17, -1.3878e-16],
         [-1.4907e-01, -7.7460e-01, -2.0000e-01,  8.3267e-17,  1.6653e-16,
           7.9103e-16,  4.5797e-16, -2.4286e-16],
         [ 1.5848e-08, -2.9410e-08, -7.8881e-01, -3.3333e-01, -2.2204e-16,
          -4.7184e-16, -2.6368e-16,  2.0123e-16],
         [-4.2195e-08,  6.9194e-08,  1.2778e-01, -7.5593e-01, -4.2857e-01,
          -1.6653e-16,  4.0246e-16,  2.3245e-16],
         [ 1.0742e-08, -5.3531e-08, -3.5315e-02,  2.0893e-01, -7.1071e-01,
          -5.0000e-01,  0.0000e+00,  3.1225e-17],
         [ 2.0008e-08,  4.2576e-08,  1.2797e-02, -7.5709e-02,  2.5754e-01,
          -6.6435e-01, -5.5556e-01, -2.3592e-16],
         [-2.8418e-08,  2.6127e-08, -5.4986e-03,  3.2530e-02, -1.1066e-01,
           2.8545e-01, -6.2063e-01, -6.0000e-01]],

        [[ 6.0000e-01, -6.5007e-16,  2.9577e-16, -6.2489e-16, -3.6457e-16,
          -5.3025e-16, -2.9663e-16, -3.3172e-17],
         [-4.6188e-01,  3.3333e-01,  2.7756e-16,  5.5511e-16, -3.3307e-16,
           0.0000e+00,  1.9429e-16,  2.1511e-16],
         [-2.5555e-01, -7.3771e-01,  1.4286e-01,  2.4980e-16,  4.7184e-16,
           1.9429e-16,  5.5511e-17,  6.9389e-18],
         [-7.5593e-02, -2.1822e-01, -8.4515e-01, -4.1633e-16,  2.4980e-16,
          -4.1633e-17, -1.5266e-16,  2.7756e-17],
         [-9.5238e-03, -2.7493e-02, -1.0648e-01, -8.8192e-01, -1.1111e-01,
           5.5511e-17,  4.1633e-17, -2.0817e-17],
         [-3.2994e-09, -2.5326e-08,  4.4054e-08,  9.6132e-09, -8.8443e-01,
          -2.0000e-01,  5.5511e-17,  6.9389e-18],
         [ 2.1919e-08,  9.7390e-09, -7.4257e-08,  3.2528e-08,  8.7407e-02,
          -8.6969e-01, -2.7273e-01, -1.7868e-16],
         [-2.0191e-08,  3.3195e-08, -4.2545e-08, -3.3834e-08, -1.5648e-02,
           1.5570e-01, -8.4632e-01, -3.3333e-01]],

        [[ 7.1429e-01, -2.9468e-16, -4.4561e-16,  4.2637e-16,  2.6322e-16,
           1.1188e-16,  3.0451e-16,  4.8584e-17],
         [-3.7115e-01,  5.0000e-01, -1.1102e-16, -1.1102e-16,  2.2204e-16,
          -6.1062e-16, -4.7184e-16, -3.4694e-18],
         [-2.6620e-01, -6.4550e-01,  3.3333e-01,  1.3878e-17,  9.7145e-17,
           0.0000e+00, -9.0206e-16, -6.2450e-17],
         [-1.2599e-01, -3.0551e-01, -7.8881e-01,  2.0000e-01, -3.3307e-16,
           1.9429e-16,  3.4694e-17,  4.5103e-17],
         [-3.8961e-02, -9.4475e-02, -2.4393e-01, -8.6588e-01,  9.0909e-02,
           9.7145e-17,  3.8858e-16,  7.4593e-17],
         [-7.1788e-03, -1.7408e-02, -4.4947e-02, -1.5954e-01, -9.0453e-01,
          -5.9605e-08,  8.7430e-16,  4.2501e-17],
         [-6.0033e-04, -1.4558e-03, -3.7587e-03, -1.3342e-02, -7.5641e-02,
          -9.1987e-01, -7.6923e-02, -6.4185e-17],
         [-2.2128e-08,  4.0451e-09, -5.0275e-08, -5.9445e-08,  1.6271e-07,
           5.4538e-08, -9.2072e-01, -1.4286e-01]],

        [[ 7.7778e-01, -1.9083e-17, -4.6461e-16, -6.1988e-16, -5.8724e-16,
          -5.9954e-16, -2.6400e-16,  0.0000e+00],
         [-3.0792e-01,  6.0000e-01,  2.2204e-16,  2.2204e-16,  6.1062e-16,
           1.6653e-16,  2.2204e-16,  0.0000e+00],
         [-2.5297e-01, -5.6334e-01,  4.5455e-01,  1.6653e-16, -8.3267e-17,
          -4.1633e-17, -1.4572e-16,  0.0000e+00],
         [-1.4966e-01, -3.3328e-01, -7.1710e-01,  3.3333e-01, -2.0817e-16,
           2.2204e-16, -2.4286e-16,  0.0000e+00],
         [-6.5268e-02, -1.4535e-01, -3.1274e-01, -8.1408e-01,  2.3077e-01,
           8.3267e-17,  1.9429e-16,  0.0000e+00],
         [-2.0616e-02, -4.5911e-02, -9.8784e-02, -2.5714e-01, -8.7471e-01,
           1.4286e-01,  1.3878e-16,  0.0000e+00],
         [-4.4824e-03, -9.9820e-03, -2.1478e-02, -5.5908e-02, -1.9018e-01,
          -9.1111e-01,  6.6667e-02,  0.0000e+00],
         [-6.0187e-04, -1.3403e-03, -2.8839e-03, -7.5069e-03, -2.5536e-02,
          -1.2234e-01, -9.3095e-01,  0.0000e+00]],

        [[ 8.1818e-01, -1.8125e-16, -1.6823e-16,  1.3858e-16,  2.7918e-17,
           2.1474e-17,  1.6176e-16, -2.5374e-17],
         [-2.6243e-01,  6.6667e-01, -2.7756e-16, -1.6653e-16, -5.5511e-17,
           1.9429e-16,  1.9429e-16, -4.5103e-17],
         [-2.3455e-01, -4.9654e-01,  5.3846e-01, -1.3878e-16, -2.7756e-17,
           6.9389e-17, -4.5797e-16,  3.2092e-17],
         [-1.5859e-01, -3.3572e-01, -6.5012e-01,  4.2857e-01,  8.3267e-17,
          -3.8858e-16, -1.1796e-16,  8.6736e-18],
         [-8.3916e-02, -1.7765e-01, -3.4401e-01, -7.5593e-01,  3.3333e-01,
          -1.2490e-16, -1.8041e-16, -1.7347e-18],
         [-3.4790e-02, -7.3648e-02, -1.4262e-01, -3.1339e-01, -8.2916e-01,
           2.5000e-01,  3.0531e-16, -3.8164e-17],
         [-1.1124e-02, -2.3548e-02, -4.5601e-02, -1.0020e-01, -2.6511e-01,
          -8.7928e-01,  1.7647e-01,  4.2501e-17],
         [-2.6553e-03, -5.6210e-03, -1.0885e-02, -2.3919e-02, -6.3284e-02,
          -2.0989e-01, -9.1270e-01,  1.1111e-01]],

        [[ 8.4615e-01, -2.9996e-16, -5.4926e-17,  1.6813e-17, -2.5195e-16,
           1.1752e-16,  2.7742e-16, -2.8853e-17],
         [-2.2840e-01,  7.1429e-01, -1.6653e-16, -1.6653e-16,  1.6653e-16,
          -2.2204e-16,  3.6082e-16, -3.4694e-17],
         [-2.1624e-01, -4.4263e-01,  6.0000e-01, -2.9143e-16, -1.5266e-16,
          -7.6328e-17, -5.2042e-17, -8.6736e-18],
         [-1.5991e-01, -3.2733e-01, -5.9161e-01,  5.0000e-01, -1.3878e-17,
           5.5511e-17,  6.2450e-17,  0.0000e+00],
         [-9.5992e-02, -1.9649e-01, -3.5514e-01, -7.0035e-01,  4.1176e-01,
           0.0000e+00, -9.7145e-17,  1.7347e-17],
         [-4.7166e-02, -9.6547e-02, -1.7450e-01, -3.4412e-01, -7.8038e-01,
           3.3333e-01, -3.1225e-17, -1.7347e-17],
         [-1.8891e-02, -3.8669e-02, -6.9890e-02, -1.3782e-01, -3.1256e-01,
          -8.3918e-01,  2.6316e-01, -1.7347e-18],
         [-6.0876e-03, -1.2461e-02, -2.2522e-02, -4.4414e-02, -1.0072e-01,
          -2.7043e-01, -8.8195e-01,  2.0000e-01]],

        [[ 8.6667e-01,  4.2344e-16,  1.2209e-16,  2.3204e-16,  2.3908e-16,
          -1.5840e-16,  2.9779e-16, -2.7686e-17],
         [-2.0207e-01,  7.5000e-01, -7.2164e-16, -2.7756e-16, -4.1633e-16,
          -4.9960e-16,  8.3267e-17,  2.1511e-16],
         [-1.9949e-01, -3.9869e-01,  6.4706e-01, -2.4980e-16, -2.7756e-17,
           1.8735e-16, -2.0817e-16,  5.2042e-17],
         [-1.5736e-01, -3.1449e-01, -5.4134e-01,  5.5556e-01, -4.8572e-16,
          -8.3267e-17, -5.7593e-16,  8.3267e-17],
         [-1.0330e-01, -2.0645e-01, -3.5537e-01, -6.4983e-01,  4.7368e-01,
          -3.5388e-16,  3.5388e-16, -3.1225e-17],
         [-5.7103e-02, -1.1412e-01, -1.9644e-01, -3.5921e-01, -7.3315e-01,
           4.0000e-01, -1.7694e-16,  9.0206e-17],
         [-2.6604e-02, -5.3170e-02, -9.1522e-02, -1.6736e-01, -3.4158e-01,
          -7.9722e-01,  3.3333e-01, -5.7246e-17],
         [-1.0392e-02, -2.0768e-02, -3.5749e-02, -6.5371e-02, -1.3342e-01,
          -3.1140e-01, -8.4632e-01,  2.7273e-01]],

        [[ 8.8235e-01, -1.4908e-17,  9.6610e-17,  2.5264e-16,  1.6303e-16,
          -4.1469e-16,  4.0790e-16, -4.4266e-17],
         [-1.8113e-01,  7.7778e-01,  2.7756e-16,  3.6082e-16,  8.3267e-17,
           2.7756e-17,  1.9429e-16,  0.0000e+00],
         [-1.8461e-01, -3.6238e-01,  6.8421e-01,  4.1633e-17, -1.6653e-16,
           9.0206e-17, -1.9082e-16,  5.5511e-17],
         [-1.5290e-01, -3.0015e-01, -4.9820e-01,  6.0000e-01,  2.0817e-16,
          -3.9552e-16, -9.3675e-17, -7.9797e-17],
         [-1.0733e-01, -2.1068e-01, -3.4970e-01, -6.0474e-01,  5.2381e-01,
          -2.9143e-16, -1.3531e-16,  5.5511e-17],
         [-6.4721e-02, -1.2705e-01, -2.1088e-01, -3.6467e-01, -6.8917e-01,
           4.5455e-01,  6.5919e-17, -3.1225e-17],
         [-3.3650e-02, -6.6054e-02, -1.0964e-01, -1.8960e-01, -3.5832e-01,
          -7.5625e-01,  3.9130e-01,  5.5511e-17],
         [-1.5061e-02, -2.9564e-02, -4.9072e-02, -8.4861e-02, -1.6037e-01,
          -3.3848e-01, -8.0952e-01,  3.3333e-01]]])


Gu_B_stacked_GBT = np.array([[6.6667e-01,  5.7735e-01,  1.4907e-01, -1.5848e-08,
                              4.2195e-08, -1.0742e-08,   -2.0008e-08,  2.8418e-08],
                            [4.0000e-01,  4.6188e-01,  2.5555e-01,  7.5593e-02,  
                            9.5238e-03,  3.2994e-09, -2.1919e-08,  2.0191e-08],
                            [2.8571e-01,  3.7115e-01,  2.6620e-01,  1.2599e-01,  
                            3.8961e-02,  7.1788e-03, 6.0033e-04,  2.2128e-08],
                            [2.2222e-01,  3.0792e-01,  2.5297e-01,  1.4966e-01,
                              6.5268e-02,  2.0616e-02,  4.4824e-03,  6.0187e-04],
                            [1.8182e-01,  2.6243e-01,  2.3455e-01,  1.5859e-01,
                              8.3916e-02,  3.4790e-02,  1.1124e-02,  2.6553e-03],
                            [1.5385e-01,  2.2840e-01,  2.1624e-01,  1.5991e-01,  
                              9.5992e-02,  4.7166e-02,  1.8891e-02,  6.0876e-03],
                            [1.3333e-01,  2.0207e-01,  1.9949e-01,  1.5736e-01,  
                              1.0330e-01,  5.7103e-02, 2.6604e-02,  1.0392e-02],
                            [1.1765e-01, 1.8113e-01,  1.8461e-01,  1.5290e-01,  
                            1.0733e-01,  6.4721e-02,  3.3650e-02,  1.5061e-02]])


Gu_coefs = np.array([[[ 2.3513e-01,  2.0363e-01,  5.2577e-02, -5.5895e-09,  1.4882e-08,
          -3.7889e-09, -7.0568e-09,  1.0023e-08]],

        [[ 4.0576e-01,  2.6490e-01, -3.3701e-02, -5.6627e-02, -7.1343e-03,
          -1.3838e-08, -4.7506e-09,  1.4944e-08]],

        [[ 3.5937e-01,  7.2189e-02, -2.2545e-01, -8.6126e-02,  2.5252e-02,
           1.1226e-02,  9.3872e-04,  3.6177e-09]],

        [[ 4.2782e-01,  1.3816e-01, -6.5221e-02,  1.5500e-01,  1.5606e-01,
           2.6968e-02, -4.6502e-03, -1.5067e-03]],

        [[ 5.7355e-01,  3.0244e-01,  8.4267e-02,  1.8955e-01,  7.4506e-07,
          -1.4422e-01, -7.2802e-02, -1.3105e-02]],

        [[ 5.0014e-01,  1.0705e-01, -1.8648e-01, -1.3038e-01, -2.6791e-01,
          -1.7971e-01,  4.9144e-02,  8.3597e-02]],

        [[ 1.3004e-01, -4.8061e-01, -7.1708e-01, -4.4194e-01, -2.8475e-01,
           3.7278e-02,  2.1051e-01,  5.7035e-02]],

        [[ 1.8084e-01, -2.9561e-01, -2.3676e-01,  3.0236e-01,  5.1647e-01,
           6.1457e-01,  3.6490e-01, -2.4948e-02]]])

def build_LegS(N):
    """
    The, non-vectorized implementation of the, measure derived from the Scaled Legendre basis.

    Args:
        N (int): Order of coefficients to describe the orthogonal polynomial that is the HiPPO projection.

    Returns:
        A (jnp.ndarray): The A HiPPO matrix.
        B (jnp.ndarray): The B HiPPO matrix.

    """
    q = np.arange(
        N, dtype=np.float64
    )  # q represents the values 1, 2, ..., N each column has
    k, n = np.meshgrid(q, q)
    r = 2 * q + 1
    M = -(np.where(n >= k, r, 0) - np.diag(q))  # represents the state matrix M
    D = np.sqrt(
        np.diag(2 * q + 1)
    )  # represents the diagonal matrix D $D := \text{diag}[(2n+1)^{\frac{1}{2}}]^{N-1}_{n=0}$
    A = D @ M @ np.linalg.inv(D)
    B = np.diag(D)[:, None]
    B = (
        B.copy()
    )  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    
    return A, B


def compute_A(n, k):
    '''
    Computes the hippo A matrix using the piecewise equation on page 31 eq. 29.
    Input: n columns and k columns of a square matrix of size N.
    Output: Individual values for the elements in the A matrix. 
    '''
    if n > k:
        val = np.sqrt(2 * n + 1) * np.sqrt(2 * k + 1)
    if n == k:
        val =n + 1 
    if n < k:
        val = 0
    return val

def compute_B(n):
    '''
    Computes the hippo B matrix using the piecewise equation on page 31 eq. 29.
    Input: n columns a square matrix of size N.
    Output: Individual values for the elements in the B matrix. 
    '''
    val = np.sqrt(2 * n + 1)
    return val

def get_A_and_B(N = 8):
    '''
    Creates the A and B matrix given the size N along a single axis of 
    a square matrix.
    Input: Size N of the square matrix along a single axis.
    Output: A and B matrix.
    '''
    A = np.zeros((N, N))
    B = np.zeros((N, 1))
    
    for n in range(A.shape[0]):
        B[n][0] = compute_B(n)
        for k in range(A.shape[1]):
            A[n, k] = compute_A(n, k)
    
    return (A  * -1).astype(np.float32), B.astype(np.float32)

testA, testB = get_A_and_B()

guA, guB = build_LegS(N=8)


def continious_hippo_operator(A, B,  f, c_t =  None):
    '''
    Performs the continious hippo operation as specified on page 31 eq. 29.
    Input: n columns a square matrix of size N.
    Output: Individual values for the elements in the B matrix. 
    '''
    lst = []
    if c_t is None:
        c_t = np.zeros((A.shape[0],))
    for t in range(1, len(f)):
        c_t = - 1/t * A * c_t + 1/t * B * f[t]  
        lst.append(c_t)
    return c_t.astype(np.float32), lst

np.random.seed(1701)
print("Random Seed: ", np.random.seed(1701))
#test_input = np.random.random(size = 8)  
#test_input= np.array([0.3527, 0.6617, 0.2434, 0.6674 ,1.2293,0.0964,-2.2756, 0.5618])
test_input = np.array([[0.3527], 
                  [0.6617], 
                  [0.2434],
                  [0.6674],
                  [1.2293],
                  [0.0964],
                  [-2.2756],
                  [0.5618]], dtype=np.float32) 

hippo_test_out, hippo_test_out_lst = continious_hippo_operator(testA, testB, test_input)

def generalized_bilinear_transform(A, B, alpha, delta_t):
    '''
    Performs the generalised bilinaer transform from EQ.13 in the paper.
    Input: A an np array of size N by N, B np array of size 1 by 8,
            Step size delta_t = 1/input length and data f_k.
    Output: transformed A and B matrix, and result for EQ13.
    '''
    I = np.eye(A.shape[0]) 
    EQ13_p1 = I - (delta_t * alpha * A)
    EQ13_p2 = I + (delta_t * (1 - alpha) * A)

    EQA = np.linalg.lstsq(EQ13_p1, EQ13_p2, rcond=None)[0]
    EQB =  np.linalg.lstsq(EQ13_p1, B, rcond=None)[0] 
    EQB = delta_t * EQB
  
    EQ13 = EQA + EQB
    return EQA.astype(np.float32), EQB.astype(np.float32), EQ13.astype(np.float32)


def discrete_hippo_recurrance(A, B, c_k, f_k):
    if c_k is None:
        c_k =  np.zeros((A.shape[0],1))
    c_k = A @ c_k + B * f_k
    return c_k.astype(np.float32)

def train(A, B, c_k, f_k):
    GBT_A_lst = []
    GBT_B_lst = []
    coefs_lst = []
    
    for t in range(1, f_k.shape[0]+1):
        delta_t = 1/ t
        GBT_A, GBT_B, EQ13 = generalized_bilinear_transform(A, B, alpha = 0.5, delta_t = delta_t)
        GBT_A_lst.append(GBT_A)
        GBT_B_lst.append(GBT_B.T)
        c_k = discrete_hippo_recurrance(GBT_A, GBT_B, c_k, f_k[t-1])
        coefs_lst.append(c_k.T)
    return c_k.astype(np.float32), GBT_A.astype(np.float32), GBT_B.astype(np.float32), EQ13, GBT_A_lst, GBT_B_lst, coefs_lst


hippo_out, GBT_A_out, GBT_B_out, EQ13_out, GBT_A_out_lst, GBT_B_out_lst, Coefs_out_lst = train(A = testA, B = testB, c_k = None, f_k = test_input)

print("------------Debug Prints-------------")
print(f"Albert Gus' A matrix:\n{Gu_A}")
print(f"Our A matrix:\n{testA}")
print(f"Albert Gus' B matrix:\n{Gu_B}")
print(f"Our B matrix:\n{testB}")

print(f"Albert Gus' GBT A matrix:\n{Gu_A_stacked_GBT}")
print(f"Our GBT A matrix:\n{GBT_A_out_lst}")
print(f"Albert Gus' GBT B matrix:\n{Gu_B_stacked_GBT}")
print(f"Our GBT B matrix:\n{GBT_B_out_lst}")

print(f"Albert Gus' coefs :\n{Gu_coefs}")
print(f"Our coefs:\n{Coefs_out_lst}")

print("------------Comparison Prints-------------")
print(f"Albert Gus' vs. Our A is close:\n{np.allclose(Gu_A, testA)}")
print(f"Albert Gus' vs. Our B is close:\n{np.allclose(Gu_B, testB)}")
print(f"Albert Gus' vs. Our A is equal:\n{Gu_A == testA}")
print(f"Albert Gus' vs. Our B is equal:\n{Gu_B == testB}")

for t in range(8):
    print(f"Albert Gus' vs. Our A is close at t_{t+1} :\n{np.allclose(Gu_A_stacked_GBT[t], GBT_A_out_lst[t], rtol=1e-06, atol=1e-04)}")
    print(f"Albert Gus' vs. Our B is close at t_{t+1} :\n{np.allclose(Gu_B_stacked_GBT[t], GBT_B_out_lst[t], rtol=1e-06, atol=1e-04)}")

    print(f"Albert Gus' vs. Our A is equal at t_{t+1} :\n{np.round(Gu_A_stacked_GBT[t].astype(np.float32),5) == np.round(GBT_A_out_lst[t].astype(np.float32),5)}")
    print(f"Albert Gus' vs. Our B is equal at t_{t+1} :\n{np.round(Gu_B_stacked_GBT[t].astype(np.float32),5) == np.round(GBT_B_out_lst[t].astype(np.float32).T,5)}")

    print(f"Albert Gus' vs. Our Coeff is close at t_{t+1} :\n{np.allclose(Gu_coefs[t], Coefs_out_lst[t], rtol=1e-06, atol=1e-04)}")
    print(f"Albert Gus' vs. Our Coeff is equal at t_{t+1} :\n{np.round(Gu_coefs[t].astype(np.float32),5) == np.round(Coefs_out_lst[t].astype(np.float32),5)}")

print("------------Shapes Prints-------------")
print(f"Albert Gus' A GBT stacked shape:\n{Gu_A_stacked_GBT.shape}")
print(f"Albert Gus' A GBT single entry shape:\n{Gu_A_stacked_GBT[0].shape}")
print(f"Albert Gus' B GBT stacked shape:\n{Gu_B_stacked_GBT.shape}")
print(f"Albert Gus' B GBT single entry shape:\n{Gu_B_stacked_GBT[0].shape}")

print(f"Albert Gus' Coeff single entry shape:\n{Gu_coefs[-1].shape}")
print(f"Our Coeff vector entry shape:\n{hippo_out.shape}")

print(f"Our A GBT shape:\n{GBT_A_out.shape}")
print(f"Our A GBT shape:\n{GBT_B_out.shape}")


