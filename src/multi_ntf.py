import numpy as np
import tensorly as tl

def flatten(X, mode=0):

        l = np.arange(len(X.shape))
        l = np.concatenate(([mode], np.delete(l,mode)))
        X_flat = X.transpose(*l).reshape((X.shape[mode], -1))

        return X_flat
    
def calculate_loss(X, Factors, mode=0):

        X_flat = flatten(X, mode)
        H = tl.tenalg.khatri_rao([Factors[i] for i in range(len(Factors)) if i != mode]).T
        return np.linalg.norm(X_flat - Factors[mode] @ H)
    
def op(X, A, S, w, eps=1e-10):
    return np.multiply(w, np.divide(A.T @ X @ S.T, eps + A.T @ A @ w @ S @ S.T))

def update(X, Factors, Ws, mode=0):
    
    X_flat = flatten(X, mode)
    H = tl.tenalg.khatri_rao([Factors[i] @ Ws[i] for i in range(len(Factors)) if i != mode]).T
    return op(X_flat, Factors[mode], H, Ws[mode])

def run(X, Factors, r, N=100):
    
    Ws = [np.random.rand(Factors[0].shape[1],r) for _ in range(len(Factors))]
    
    losses = []
    for K in range(N):
        for i in range(len(Factors)):
            Ws[i] = update(X, Factors, Ws, mode=i)
        losses.append(calculate_loss(X, [Factors[i] @ Ws[i] for i in range(len(Factors))]))
        
    return Ws


def update_single(X, Factors, W, mode=0):
    
    X_flat = flatten(X, mode)
    H = tl.tenalg.khatri_rao([Factors[i] @ W for i in range(len(Factors)) if i != mode]).T
    return op(X_flat, Factors[mode], H, W)

def run_single(X, Factors, r, N=100):
    
    W = np.random.rand(Factors[0].shape[1],r)
    
    losses = []
    for K in range(N):
        Ws = []
        for i in range(len(Factors)):
            Ws.append(update_single(X, Factors, W, mode=i))
           
        Ws.append(W)
        Ws.append(W)
        Ws.append(W)
        W = np.stack(Ws).mean(axis=0)
                       
        losses.append(calculate_loss(X, [Factors[i] @ W for i in range(len(Factors))]))
    
    return W