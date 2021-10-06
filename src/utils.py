import sys
import itertools

sys.path.append("./src")
import torch
import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import linear_sum_assignment as lap

##################################
###  General Helper Functions  ###
##################################

def outer_product(factors):
    
    """
    Calculates the outer product ABC = sum_{i=1}^r a_i (outer) b_i (outer) c_i
    """
    shape = [factors[i].shape[0] for i in range(len(factors))]
    
    r = factors[0].shape[1]
    k = len(factors)
    X_approx = torch.zeros(tuple(shape))
    
    for j in range(r):
        temp = torch.einsum(','.join([chr(97+i) for i in range(k)]), *[factors[i][:,j] for i in range(k)])
        X_approx += temp
        
    return X_approx

def outer_product_np(factors):
    
    """
    Calculates the outer product ABC = sum_{i=1}^r a_i (outer) b_i (outer) c_i
    """
    shape = [factors[i].shape[0] for i in range(len(factors))]
    
    r = factors[0].shape[1]
    k = len(factors)
    X_approx = np.zeros(tuple(shape))
    
    for j in range(r):
        temp = np.einsum(','.join([chr(97+i) for i in range(k)]), *[factors[i][:,j] for i in range(k)])
        X_approx += temp
        
    return X_approx

def display_tensor(X, figsize=(12,5)):
    
    vmin=0
    vmax=3.5
    
    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=figsize)
    axs[0].axes.get_xaxis().set_ticks([])
    axs[0].axes.get_yaxis().set_ticks([])
    axs[1].axes.get_xaxis().set_ticks([])
    axs[1].axes.get_yaxis().set_ticks([])
    axs[2].axes.get_xaxis().set_ticks([])
    axs[2].axes.get_yaxis().set_ticks([])
    X_max = np.max(X,axis=0)
    axs[0].imshow(X_max, vmin=vmin, vmax=vmax)
    X_max = np.max(X,axis=1)
    axs[1].imshow(X_max, vmin=vmin, vmax=vmax)
    X_max = np.max(X,axis=2)
    axs[2].imshow(X_max, vmin=vmin, vmax=vmax)
    plt.show()
    
    
def display_topic_matrices(Ss, S_trues):
    
    fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(8,5))


    for i, S in enumerate(Ss):

        axs[0,i].axes.get_xaxis().set_ticks([])
        axs[0,i].axes.get_yaxis().set_ticks([])
        axs[1,i].axes.get_xaxis().set_ticks([])
        axs[1,i].axes.get_yaxis().set_ticks([])

        best_value, best_i, best_j = find_fit((S.T / np.sum(S, axis=1)).T, S_trues[i])

        axs[0,i].imshow((S.T / np.sum(S, axis=1)).T[best_i][:,best_j])
        axs[1,i].imshow(S[best_i][:,best_j])


    plt.show()
##############################################
###  Helper Functions For Synthetic Tensor ###
##############################################

def get_synthetic_tensor(noise=0.1, seed=1):
    
    a = 0.75
    b = 2
    c = 3.5

    X = np.zeros((40,40,40))

    X[0:20,0:20,0:20] = a
    X[20:40,20:40,20:40] = a

    X[0:12,0:12,0:8] = b
    X[12:20,12:20,8:20] = b

    X[20:27,20:34,20:27] = b
    X[27:40,34:40,27:40] = b

    X[0:12,0:6,0:4] = c
    X[0:12,6:12,4:8] = c

    X[12:17,12:20,8:13] = c
    X[17:20,12:20,13:20] = c

    X[20:23,20:27,20:27] = c
    X[23:27,27:34,20:27] = c
    
    #add Gaussian Noise
    np.random.seed(seed)
    X = X + noise * np.abs(np.random.randn(40, 40, 40))
    
    return X

def get_synthetic_gt_topics():

    S_true_7_2 = np.zeros((7,2))
    S_true_7_2[0:4,0] = 1
    S_true_7_2[4:7,1] = 1
    
    S_true_7_4 = np.zeros((7,4))
    S_true_7_4[0:2,0] = 1
    S_true_7_4[2:4,1] = 1
    S_true_7_4[4:6,2] = 1
    S_true_7_4[6,3] = 1

    S_true_4_2 = np.zeros((4,2))
    S_true_4_2[0:2,0] = 1
    S_true_4_2[2:4,1] = 1
    
    return S_true_7_2, S_true_7_4, S_true_4_2


#############################################################
###  Helper Functions for Topic Modelling and Recon Loss  ###
#############################################################

def cost(x, x_true):
    """
    Given a row of S and S_true, evaluate the costs

    Parameters
    ----------
    x: 1darray(n)
        Predicted
    x_true:
        1darray(n)
        Ground truth
    """

    return 1 - x[x_true==1].item()

    # Can you do x_true==1 faster with x_true.nonzero()?

def get_cost_matrix(S, S_true):
    """Get row to row assignment costs matrix.

    Parameters
    ----------
    S : 2darray(m, n)
        Predicted
    S_true : 2darray(m, n)
        Ground truth

    Returns
    -------
    costs : 2darray(m, m)
        The `i,j` element is the cost between row `i` of `S` and row `j` of `S_true`.
    """

    m, n = S.shape

    C = np.empty((m,m))

    for i in range(m):
        for j in range(m):
            C[i, j] = cost(S[i], S_true[j])

    return C

def find_fit(S, S_true):

    m, n = S.shape
    if m < n:
        # Note: the loss function is not actually symmetric, we just pretend.
        return find_fit(S.T, S_true.T)

    # Consider using np.fromiter combined with itertools.chain
    # https://stackoverflow.com/questions/34018470/reconcile-np-fromiter-and-multidimensional-arrays-in-python
    best_j, best_x, best_value = None, None, float("inf")
    for j, x in enumerate(itertools.permutations(range(n))):
        costs = get_cost_matrix(S[:, list(x)], S_true).T
        row_ind, col_ind = lap(costs)
        val = costs[row_ind, col_ind].sum() / m
        if(val < best_value):
            best_i = col_ind
            best_j = list(x)
            best_value = val

    return best_value, best_i, best_j


def loss(S, S_true):
    # Here we assume that the rows of S are normalized, so that each row sums to 1
    return (S.shape[0] - np.sum(S[S_true==1])) / S.shape[0]
   
def categorical_cross_entropy(x, c):
    
    n = len(x)
    
    loss = 0
    for i in range(n):
        if(c == i):
            loss -= np.log(x[i])
        else:
            loss -= np.log(1 - x[i])
            
    return loss

def topic_modeling_loss(S, S_true):
    
    N = S.shape[0]
    loss = 0
    
    for i in range(N):
        loss += categorical_cross_entropy(S[i], np.argmax(S_true[i]))
        
    return loss

def display_modeling(FROM, TO, PRED, best_i, best_j):

    fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(5,7))
    axs[0][0].imshow(FROM) 
    axs[0][1].imshow(TO)
    axs[0][2].imshow(PRED)

    axs[1][0].imshow(FROM[:,best_i])
    axs[1][1].imshow(TO[:,best_j])
    axs[1][2].imshow(PRED[best_i,:][:,best_j])

    
def recon_loss(X, approx):
    
    return  np.linalg.norm(np.ndarray.flatten(X-approx), 2)  / np.linalg.norm(np.ndarray.flatten(X), 2)

def measure_modeling(S, S_true):

    TRUE = S_true
    PRED = (S.T / np.sum(S, axis=1)).T

    best_value, best_i, best_j = find_fit(PRED, TRUE)
  
    return best_value

def handle_losses(recon, topic):
    
    # depending on the number of modes in `recon' we either average over one or two dimensions
    if len(recon.shape) == 2:
        s = (1)
    else:
        s = (1,2)
        
    if len(topic.shape) == 2:
        t = (1)
    else:
        t = (1,2)
    
    average_recons = np.mean(recon, axis=s)
    average_topics = np.mean(topic, axis=t)
    
    best_recons = np.min(recon, axis=s)
    best_topics = np.min(topic, axis=t)
    
    worst_recons = np.max(recon, axis=s)
    worst_topics = np.max(topic, axis=t)
    
    print("Topic Modeling Loss 7 to 2 (average, best, worst):", average_topics[0], best_topics[0], worst_topics[0])
    print("Topic Modeling Loss 7 to 4 (average, best, worst):", average_topics[1], best_topics[1], worst_topics[1])
    print("Topic Modeling Loss 4 to 2 (average, best, worst):", average_topics[2], best_topics[2], worst_topics[2])
    print("")
    print("Relative Recon Loss Rank 7 (average, best, worst):", average_recons[0], best_recons[0], worst_recons[0])
    print("Relative Recon Loss Rank 4 (average, best, worst):", average_recons[1], best_recons[1], worst_recons[1])
    print("Relative Recon Loss Rank 2 (average, best, worst):", average_recons[2], best_recons[2], worst_recons[2])

    
def measure_single(history, X, S_trues, kind="Neural"):
    
    results_topic = np.empty((3,3))
    
    if kind == "Neural":
        
        X1 = np.asarray(history.get('A_X1')[-1])
        X2 = np.asarray(history.get('B_X1')[-1])
        X3 = np.asarray(history.get('C_X1')[-1])

        A_A1 = np.asarray(history.get('A_A1')[-1])
        A_S1 = np.asarray(history.get('A_S1')[-1])
        B_A1 = np.asarray(history.get('B_A1')[-1])
        B_S1 = np.asarray(history.get('B_S1')[-1])
        C_A1 = np.asarray(history.get('C_A1')[-1])
        C_S1 = np.asarray(history.get('C_S1')[-1])

        A_A2 = np.asarray(history.get('A_A2')[-1])
        A_S2 = np.asarray(history.get('A_S2')[-1])
        B_A2 = np.asarray(history.get('B_A2')[-1])
        B_S2 = np.asarray(history.get('B_S2')[-1])
        C_A2 = np.asarray(history.get('C_A2')[-1])
        C_S2 = np.asarray(history.get('C_S2')[-1])
        
    else:
        
        X1, A_A1, A_S1, A_A2, A_S2 = history[0]
        X2, B_A1, B_S1, B_A2, B_S2 = history[1]
        X3, C_A1, C_S1, C_A2, C_S2 = history[2]
    
    results_topic[0,0] = measure_modeling(A_S2.T, S_trues[0])
    results_topic[1,0] = measure_modeling(B_S2.T, S_trues[0])
    results_topic[2,0] = measure_modeling(C_S2.T, S_trues[0])

    results_topic[0,1] = measure_modeling(A_S1.T, S_trues[1])
    results_topic[1,1] = measure_modeling(B_S1.T, S_trues[1])
    results_topic[2,1] = measure_modeling(C_S1.T, S_trues[1])

    results_topic[0,2] = measure_modeling(A_A2, S_trues[2])
    results_topic[1,2] = measure_modeling(B_A2, S_trues[2])
    results_topic[2,2] = measure_modeling(C_A2, S_trues[2])
    
    approx_1 = outer_product_np(X1, X2, X3)
    approx_2 = outer_product_np(A_A1 @ A_S1, B_A1 @ B_S1, C_A1 @ C_S1)
    approx_3 = outer_product_np(A_A1 @ A_A2 @ A_S2, B_A1 @ B_A2 @ B_S2, C_A1 @ C_A2 @ C_S2)
    
    results_recon = np.asarray([recon_loss(X, approx_1), recon_loss(X, approx_2), recon_loss(X, approx_3)])
    
    return results_topic, results_recon

def measure_all(histories, X, S_trues, kind="Neural"):
    
    all_topic_losses = np.empty((3, 3, len(histories)))
    all_recon_losses = np.empty((3, len(histories)))
    
    for i in range(len(histories)):
        topic_losses, recon_losses = measure_single(histories[i], X, S_trues, kind=kind)
        all_topic_losses[:,:,i] = topic_losses
        all_recon_losses[:,i] = recon_losses
        
    return all_topic_losses, all_recon_losses
        

             