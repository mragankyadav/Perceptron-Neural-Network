import numpy as np
import random
import matplotlib.pyplot as plt

images = []

class Perceptron:
    def __init__(self, N):
        xA,yA,xB,yB = [random.uniform(-1, 1) for i in range(4)]
        self.V = np.array([xB*yA-xA*yB, yB-yA, xA-xB])
        self.X = self.generate_points(N)
        
    def andData(self):
        t = 0
        self.V = np.array([t,1,1])
        X = []
        N = len(self.X)
        for i in range(N):
            x1, x2 = [random.uniform(-1, 1) for i in range(2)]
            x = np.array([1, x1, x2])
            if self.V.T.dot(x) >= 0:
                s = 1
            else:
                s = -1
            X.append((x, s))
        self.X = X

    def orData(self):
        t = 0.5
        self.V = np.array([t,1,1])
        X = []
        N = len(self.X)
        for i in range(N):
            x1, x2 = [random.uniform(-1, 1) for i in range(2)]
            x = np.array([1, x1, x2])
            if self.V.T.dot(x) >= 0:
                s = 1
            else:
                s = -1
            X.append((x, s))
        self.X = X


    def xorData(self):
        t = 0
        # self.V = np.array([t,-1,0])
        X = []
        N = len(self.X)
        for i in range(N):
            x1, x2 = [random.uniform(-1, 1) for i in range(2)]
            x = np.array([1, x1, x2])
            if (x1 >= 0 and x2 >= 0) or (x1 < 0 and x2 < 0):
                s = -1
            else:
                s = 1
            X.append((x, s))
        self.X = X

        X = []
        for i in range(N):
            x1,x2 = [random.uniform(-1, 1) for i in range(2)]
            x = np.array([1,x1,x2])
            s = int(np.sign(self.V.T.dot(x)))
            X.append((x, s))
        return X

    def plot(self, mispts=None, vec=None, save=False):
        fig = plt.figure(figsize=(5,5))
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        V = self.V
        a, b = -V[1]/V[2], -V[0]/V[2]
        l = np.linspace(-1,1)
        plt.plot(l, a*l+b, 'k-')
        cols = {1: 'r', -1: 'b'}
        for x,s in self.X:
            plt.plot(x[1], x[2], cols[s]+'o')
        if mispts:
            for x,s in mispts:
                plt.plot(x[1], x[2], cols[s]+'.')
        if vec != None:
            aa, bb = -vec[1]/vec[2], -vec[0]/vec[2]
            plt.plot(l, aa*l+bb, 'g-', lw=2)
        if save:
            if not mispts:
                plt.title('N = %s' % (str(len(self.X))))
            else:
                plt.title('N = %s with %s test points' \
                % (str(len(self.X)),str(len(mispts))))
            plt.savefig('p_N%s' % (str(len(self.X))), \
            dpi=200, bbox_inches='tight')

    def error(self, vec, pts=None):
        if not pts:
            pts = self.X
        M = len(pts)
        n_mispts = 0
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                n_mispts += 1
        error = n_mispts / float(M)
        return error

    def choose_miscl_point(self, vec):
        pts = self.X
        mispts = []
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                mispts.append((x, s))
        return mispts[random.randrange(0,len(mispts))]

    def pla(self, save=False):
    # Initialize the weigths to zeros
        w = np.zeros(3)
        X, N = self.X, len(self.X)
        it = 0
        # Iterate until all points are correctly classified
        while self.error(w) != 0 and it <= 100:
            it += 1
            # Pick random misclassified point
            x, s = self.choose_miscl_point(w)
            # Update weights
            w += s*x
            if save:
                self.plot(vec=w)
                plt.title('N = %s, Iteration %s\n' \
                % (str(N),str(it)))
                plt.savefig('p_N%s_it%s' % (str(N),str(it)), \
                dpi=200, bbox_inches='tight')
                # images.append(plt)
        self.w = w

    def check_error(self, M, vec):
        check_pts = self.generate_points(M)
        return self.error(vec, pts=check_pts)

    def generate_points_XOR(self, N):
        X = []
        for i in range(N):
            x1, x2 = [random.uniform(-1, 1) for i in range(2)]
            x = np.array([1, x1, x2])
            if (x1 >= 0 and x2 >= 0) or (x1 < 0 and x2 < 0):
                s = -1
            else:
                s = 1
            X.append((x, s))
        return X


percept = Perceptron(100)
#percept.orData()
#percept.andData()
percept.xorData()
percept.pla(save=True)

percept.plot(percept.generate_points(500), p.w, save=True)
err = []
for i in range(100):
    err.append(percept.check_error(500, percept.w))
print(np.mean(err))