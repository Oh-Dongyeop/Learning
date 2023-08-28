import numpy as np

def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

def HA(x1, x2):
    s = XOR(x1, x2)
    c = AND(x1, x2)
    return np.array([s, c])
    
def FA(x1, x2, c_in):
    y1 = HA(x1, x2)
    y2 = HA(y1[0], c_in)
    s = y2[0]
    c_out = OR(y1[1], y2[1])
    return np.array([s, c_out])


list = [0,1]

for i in list:
    for j in list:
        print('AND (%d, %d) : %d' % (i, j, AND(i,j)))
print("\n")

for i in list:
    for j in list:
        print('NAND (%d, %d) : %d' % (i, j, NAND(i,j)))
print("\n")

for i in list:
    for j in list:
        print('OR (%d, %d) : %d' % (i, j, OR(i,j)))
print("\n")

for i in list:
    for j in list:
        print('XOR (%d, %d) : %d' % (i, j, XOR(i,j)))
print("\n")

for i in list:
    for j in list:
        print('HA (%d, %d) : %d %d (C, S)' % (i, j, HA(i,j)[1], HA(i,j)[0]))
print("\n")

for i in list:
    for j in list:
        for k in list:
            print('FA (%d, %d, %d) : %d %d (C, S)' % (i, j, k, FA(i,j,k)[1], FA(i,j,k)[0]))
print("\n")