import os
import numpy as np
from scipy.stats import norm, t

# Generate Simulation Datasets
for M in range(1):
    path = './Simu'  # set your own folder path
    name1 = '/SimuData_p50'  # Case Pc=50 
    name2 = '/SimuData_p100'  # Case Pc=100
    os.mkdir(path)
    os.mkdir(os.path.join(path, name1))
    os.mkdir(os.path.join(path, name2))
    
    # Case Pc=100
    N = 200
    m = 100
    T = 180
    stdv = 0.05
    theta_w = 0.02
    stde = 0.05
    
    rho = np.random.uniform(0.9, 1, size=(m, 1))
    c = np.zeros((N*T, m))
    for i in range(m):
        x = np.zeros((N, T))
        x[:, 0] = norm.rvs(size=N)
        for t in range(1, T):
            x[:, t] = rho[i]*x[:, t-1] + norm.rvs(size=N)*np.sqrt(1-rho[i]**2)
        r = np.argsort(x, axis=0)
        szx = x.shape
        x1 = np.zeros(szx)
        ridx = np.arange(1, N+1)
        for k in range(szx[1]):
            x1[r[:, k], k] = ridx*2/(N+1) - 1
        c[:, i] = x1.reshape(-1)
    
    per = np.repeat(np.arange(1, N+1), T)
    time = np.tile(np.arange(1, T+1), N)
    vt = norm.rvs(size=(3, T))*stdv
    beta = c[:, [0, 1, 2]]
    betav = np.zeros(N*T)
    for t in range(T):
        ind = (time == t+1)
        betav[ind] = beta[ind, :].dot(vt[:, t])
        
    y = np.zeros(T)
    y[0] = norm.rvs(size=1)
    q = 0.95
    for t in range(1, T):
        y[t] = q*y[t-1] + norm.rvs(size=1)*np.sqrt(1-q**2)
        
    cy = c.copy()
    for t in range(T):
        ind = (time == t+1)
        cy[ind, :] = c[ind, :]*y[t]
    
    ep = t.rvs(df=5, size=N*T)*stde
    
    # Model 1
    theta = np.concatenate(([1, 1], np.zeros(m-2), [0, 0, 1], np.zeros(m-3))) * theta_w
    r1 = np.hstack((c, cy)).dot(theta) + betav + ep
    rt = np.hstack((c, cy)).dot(theta)
    
    pathc = os.path.join(path, name2, 'c' + str(M) + '.csv')
    np.savetxt(pathc, np.hstack((c, cy)), delimiter=',')
    
    pathr = os.path.join(path, name2, 'r1_' + str(M) + '.csv')
    np.savetxt(pathr, r1, delimiter=',')
    
    # Model 2
    theta = np.concatenate(([1, 1], np.zeros(m-2),
                            
                            
    #%%% Model 2

    import numpy as np

    theta = np.concatenate(([1, 1], np.repeat(0, m-2), [0, 0, 1], np.repeat(0, m-3))) @ theta_w
    z = np.hstack((c[:, :m], cy[:, :m]))
    z[:, 0] = c[:, 0]**2 * 2
    z[:, 1] = c[:, 0] * c[:, 1] * 1.5
    z[:, m+2] = np.sign(cy[:, 2]) * 0.6

    r1 = z @ theta + betav + ep
    rt = z @ theta
    # print(1 - np.sum((r1 - rt)**2) / np.sum((r1 - np.mean(r1))**2))

    pathr = f"{path}{name1}/r2_{M}.csv"
    np.savetxt(pathr, r1, delimiter=",")

    print(M)


