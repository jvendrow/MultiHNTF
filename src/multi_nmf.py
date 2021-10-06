import numpy as np


def update(X, A, S, w, w2, val=0, eps=1e-10):

    if val == 0:
        wS = w2.T @ S
        return np.multiply(w, np.divide(A.T @ X @ wS.T, eps + A.T @ A @ w @ wS @ wS.T))
    else:
        Aw = A @ w
        return np.multiply(w2, (np.divide(Aw.T @ X @ S.T, eps + Aw.T @ Aw @ w2.T @ S @ S.T)).T)
    
def update_supervised(X, A, S, Y, B, w, w2, lam, eps=1e-10):
    
    Aw = A @ w
    return np.multiply(w2, (np.divide(Aw.T @ X @ S.T + lam * B.T @ Y @ S.T, eps + Aw.T @ Aw @ w2.T @ S @ S.T + lam * B.T @ B @ w2.T @ S @ S.T)).T)

def update_B(X, A, S, Y, B, w, w2, lam, eps=1e-10):
    
    wS = w2.T @ S
    return np.multiply(B, np.divide(Y @ wS.T, eps + B @ wS @ wS.T))   

def run_HNMF_unsupervised(X, A, S, r, N=400):
           
    W = np.random.rand(A.shape[1],r)
    V = np.random.rand(A.shape[1],r)

    for _ in range(N):
    
        W = update(X, A, S, W, V, val=0)
        V = update(X, A, S, W, V, val=1)
        
    return W, V
  
def run_HNMF_supervised(X, A, S, Y, r, N=400, lam=1):
           
    W = np.random.rand(A.shape[1],r)
    V = np.random.rand(A.shape[1],r)
    B = np.random.rand(r, r)

    for _ in range(N):

        W = update(X, A, S, W, V, val=0)
        V = update_supervised(X, A, S, Y, B, W, V, lam)
        B = update_B(X, A, S, Y, B, W, V, lam)
        
    return W, V, B



def update_single(X, A, S, w, val=0, eps=1e-10):

    if val == 0:
        wS = w.T @ S
        return np.multiply(w, np.divide(A.T @ X @ wS.T, eps + A.T @ A @ w @ wS @ wS.T))
    else:
        Aw = A @ w
        return np.multiply(w, (np.divide(Aw.T @ X @ S.T, eps + Aw.T @ Aw @ w.T @ S @ S.T)).T)


def update_supervised_single(X, A, S, Y, B, w, lam, eps=1e-10):
    
    Aw = A @ w
    return np.multiply(w, (np.divide(Aw.T @ X @ S.T + lam * B.T @ Y @ S.T, eps + Aw.T @ Aw @ w.T @ S @ S.T + lam * B.T @ B @ w.T @ S @ S.T)).T)

def update_B_single(X, A, S, Y, B, w, lam, eps=1e-10):
    
    wS = w.T @ S
    return np.multiply(B, np.divide(Y @ wS.T, eps + B @ wS @ wS.T))   

def run_HNMF_unsupervised_single(X, A, S, r, N=400):
           
    W = np.random.rand(A.shape[1],r)

    
    for _ in range(N):
        
        #if _ % 20 == 0:
        #print(np.linalg.norm(X - A @ W @ W.T @ S))
    
        W1 = update_single(X, A, S, W, val=0)
        W2 = update_single(X, A, S, W, val=1)
        
        W = (2*W + W1 + W2) / 4
        
    return W

def run_HNMF_supervised_single(X, A, S, Y, r, N=400, lam=1):
           
    W = np.random.rand(A.shape[1],r)
    B = np.random.rand(r, r)

    for _ in range(N):

        W1 = update_single(X, A, S, W, val=0)
        W2 = update_supervised_single(X, A, S, Y, B, W, lam)
        
        W = (2*W + W1 + W2) / 4
        
        B = update_B_single(X, A, S, Y, B, W, lam)
        
    return W, B