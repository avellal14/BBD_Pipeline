import numpy as np
from scipy.sparse import coo_matrix

def ESLclique(A):
    A = np.array(A)

    if not A.shape[0] == A.shape[1]:
        raise ValueError('Adjacency matrix is not square.')

    if not np.all(np.logical_or(A == 1, A == 0)):
        raise ValueError('Adjacency matrix is not boolean (zero-one valued).')

    if not np.all(np.equal(A, A.transpose())):
        raise ValueError('Adjacency matrix is not undirected (symmetric).')

    if np.trace(np.abs(A)) != 0:
        raise ValueError('Adjacency matrix contains self-edges (check your diagonal).')

    n = A.shape[1]

    global SC
    global nc
    global ns
    global nSC

    SC = np.empty((np.int(1e4), 2))
    SC[:] = np.NAN
    nc = 0
    ns = 0
    nSC = 1e4

    ########################################################
    def BKpivot(R, P, X):
        global SC
        global nc
        global ns
        global nSC

        if (not np.any(P)) and (not np.any(X)):
            nc = nc + 1
            nr = np.sum(R)
            if ns + nr > nSC:
                tmp = np.empty((np.int(1e4), 2))
                tmp[:] = np.NAN
                SC = np.cstack([SC,tmp])
                nSC = nSC + 1e4

            SC[ns:(ns + nr), 0] = np.where(R)[0]
            SC[ns:(ns + nr), 1] = nc - 1
            ns = ns + nr

        else:
            ppivots = np.logical_or(P, X)
            pcounts = np.matmul(A[ppivots,:],np.expand_dims(P.astype(np.float),axis = 1))
            ind = np.argmax(pcounts)
            u_p = np.where(ppivots)[0][ind]

            for u in np.ravel(np.where(np.squeeze(np.logical_and(A[u_p,:] == 0, P)))):
                if u.size:
                    Rnew = np.copy(R)
                    Rnew[u] = 1
                    Nu = np.squeeze(A[u,:] == 1)
                    Pnew = np.logical_and(P, Nu)
                    Xnew = np.logical_and(X, Nu)
                    BKpivot(Rnew, Pnew, Xnew)
                    P[u] = 0
                    X[u] = 1

    #########################################################################

    O = np.empty((n))
    O[:] = np.NAN

    A0 = A.astype(np.float)

    for i in range(n):
        Nb = np.sum(A0, axis = 1)
        j = np.argmin(Nb)
        A0[j,:] = 0
        A0[:,j] = 0
        A0[j, j] = np.Inf
        O[i] = j

    for i in range(n):
        v = O[i]
        R = np.zeros((n), dtype=bool)
        R[np.int(v)] = 1
        Nv = A[np.int(v),:] == 1
        P = np.copy(Nv)
        P[O[:np.int(i + 1)].astype(np.int)] = 0
        X = np.copy(Nv)
        X[O[i:].astype(np.int)] = 0
        BKpivot(R, P, X)

    SC = SC[0:ns,:]
    data = np.array(np.ones(SC.shape[0]))
    sp = coo_matrix((data, (SC[:,0], SC[:,1])), shape = (n, nc))

    return sp