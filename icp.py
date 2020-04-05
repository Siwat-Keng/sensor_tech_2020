def icp(a, b, max_time = 1):
        import cv2
        import numpy
        import copy
        import pylab
        import time
        import sys
        import sklearn.neighbors
        import scipy.optimize



        def res(p,src,dst):
                T = numpy.matrix([[numpy.cos(p[2]),-numpy.sin(p[2]),p[0]],
                [numpy.sin(p[2]), numpy.cos(p[2]),p[1]],
                [0 ,0 ,1 ]])
                n = numpy.size(src,0)
                xt = numpy.ones([n,3])
                xt[:,:-1] = src
                xt = (xt*T.T).A
                d = numpy.zeros(numpy.shape(src))
                d[:,0] = xt[:,0]-dst[:,0]
                d[:,1] = xt[:,1]-dst[:,1]
                r = numpy.sum(numpy.square(d[:,0])+numpy.square(d[:,1]))
                return r

        def jac(p,src,dst):
                T = numpy.matrix([[numpy.cos(p[2]),-numpy.sin(p[2]),p[0]],
                [numpy.sin(p[2]), numpy.cos(p[2]),p[1]],
                [0 ,0 ,1 ]])
                n = numpy.size(src,0)
                xt = numpy.ones([n,3])
                xt[:,:-1] = src
                xt = (xt*T.T).A
                d = numpy.zeros(numpy.shape(src))
                d[:,0] = xt[:,0]-dst[:,0]
                d[:,1] = xt[:,1]-dst[:,1]
                dUdth_R = numpy.matrix([[-numpy.sin(p[2]),-numpy.cos(p[2])],
                                [ numpy.cos(p[2]),-numpy.sin(p[2])]])
                dUdth = (src*dUdth_R.T).A
                g = numpy.array([  numpy.sum(2*d[:,0]),
                                numpy.sum(2*d[:,1]),
                                numpy.sum(2*(d[:,0]*dUdth[:,0]+d[:,1]*dUdth[:,1])) ])
                return g
        def hess(p,src,dst):
                n = numpy.size(src,0)
                T = numpy.matrix([[numpy.cos(p[2]),-numpy.sin(p[2]),p[0]],
                [numpy.sin(p[2]), numpy.cos(p[2]),p[1]],
                [0 ,0 ,1 ]])
                n = numpy.size(src,0)
                xt = numpy.ones([n,3])
                xt[:,:-1] = src
                xt = (xt*T.T).A
                d = numpy.zeros(numpy.shape(src))
                d[:,0] = xt[:,0]-dst[:,0]
                d[:,1] = xt[:,1]-dst[:,1]
                dUdth_R = numpy.matrix([[-numpy.sin(p[2]),-numpy.cos(p[2])],[numpy.cos(p[2]),-numpy.sin(p[2])]])
                dUdth = (src*dUdth_R.T).A
                H = numpy.zeros([3,3])
                H[0,0] = n*2
                H[0,2] = numpy.sum(2*dUdth[:,0])
                H[1,1] = n*2
                H[1,2] = numpy.sum(2*dUdth[:,1])
                H[2,0] = H[0,2]
                H[2,1] = H[1,2]
                d2Ud2th_R = numpy.matrix([[-numpy.cos(p[2]), numpy.sin(p[2])],[-numpy.sin(p[2]),-numpy.cos(p[2])]])
                d2Ud2th = (src*d2Ud2th_R.T).A
                H[2,2] = numpy.sum(2*(numpy.square(dUdth[:,0])+numpy.square(dUdth[:,1]) + d[:,0]*d2Ud2th[:,0]+d[:,0]*d2Ud2th[:,0]))
                return H
        t0 = time.time()
        init_pose = (0,0,0)
        src = numpy.array([a.T], copy=True).astype(numpy.float32)
        dst = numpy.array([b.T], copy=True).astype(numpy.float32)
        Tr = numpy.array([[numpy.cos(init_pose[2]),-numpy.sin(init_pose[2]),init_pose[0]],
                        [numpy.sin(init_pose[2]), numpy.cos(init_pose[2]),init_pose[1]],
                        [0,                    0,                   1          ]])
        print("src",numpy.shape(src))
        print("Tr[0:2]",numpy.shape(Tr[0:2]))
        src = cv2.transform(src, Tr[0:2])
        p_opt = numpy.array(init_pose)
        T_opt = numpy.array([])
        error_max = sys.maxsize
        first = False
        while not(first and time.time() - t0 > max_time):
                distances, indices = sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto',p = 3).fit(dst[0]).kneighbors(src[0])
                p = scipy.optimize.minimize(res,[0,0,0],args=(src[0],dst[0, indices.T][0]),method='Newton-CG',jac=jac,hess=hess).x
                T  = numpy.array([[numpy.cos(p[2]),-numpy.sin(p[2]),p[0]],[numpy.sin(p[2]), numpy.cos(p[2]),p[1]]])
                p_opt[:2]  = (p_opt[:2]*numpy.matrix(T[:2,:2]).T).A       
                p_opt[0] += p[0]
                p_opt[1] += p[1]
                p_opt[2] += p[2]
                src = cv2.transform(src, T)
                Tr = (numpy.matrix(numpy.vstack((T,[0,0,1])))*numpy.matrix(Tr)).A
                error = res([0,0,0],src[0],dst[0, indices.T][0])

                if error < error_max:
                        error_max = error
                        first = True
                        T_opt = Tr

        p_opt[2] = p_opt[2] % (2*numpy.pi)
        return T_opt, error_max


def main(image0, image1):
        import cv2
        import numpy
        import random
        import matplotlib.pyplot
        n1 = 100
        n2 = 75
        bruit = 1/10
        center = [random.random()*(2-1)*3,random.random()*(2-1)*3]
        radius = random.random()
        deformation = 2

        # template = numpy.array([
        # [numpy.cos(i*2*numpy.pi/n1)*radius*deformation for i in range(n1)], 
        # [numpy.sin(i*2*numpy.pi/n1)*radius for i in range(n1)]
        # ])

        # data = numpy.array([
        # [numpy.cos(i*2*numpy.pi/n2)*radius*(1+random.random()*bruit)+center[0] for i in range(n2)], 
        # [numpy.sin(i*2*numpy.pi/n2)*radius*deformation*(1+random.random()*bruit)+center[1] for i in range(n2)]
        # ])

        template = [[],[]]
        data1 = cv2.imread(image0)
        data1 = cv2.resize(data1,(200,200))
        for x,v_x in enumerate(data1):
                for y,v_y in enumerate(v_x):
                        if sum(v_y) != 0:
                                template[0].append(x-1024)
                                template[1].append(y-1024)          

        data = [[],[]]
        data2 = cv2.imread(image1)
        data2 = cv2.resize(data2,(200,200))
        for x,v_x in enumerate(data2):
                for y,v_y in enumerate(v_x):
                        if sum(v_y) != 0:
                                data[0].append(x-1024)
                                data[1].append(y-1024)          

        template = numpy.array(template)
        data = numpy.array(data)

        print(data)

        T,error = icp(data,template)
        dx = T[0,2]
        dy = T[1,2]
        rotation = numpy.arcsin(T[0,1]) * 360 / 2 / numpy.pi

        print("T",T)
        print("error",error)
        print("rotation°",rotation)
        print("dx",dx)
        print("dy",dy)

        result = cv2.transform(numpy.array([data.T], copy=True).astype(numpy.float32), T).T
        new = [[],[]]
        for point in range(len(template[0])):
                if len(new[0]) == 0:
                        new[0].append(template[0][point])
                        new[1].append(template[1][point])
                elif ( (new[0][len(new[0])-1] - (template[0][point]))**2 + (new[1][len(new[1])-1] - (template[1][point]))**2  )**0.5 < 10:
                        new[0].append(template[0][point])
                        new[1].append(template[1][point])
                else:
                        matplotlib.pyplot.plot(new[0], new[1], 'r')
                        new[0] = []
                        new[1] = []
                        new[0].append(template[0][point])
                        new[1].append(template[1][point])

        new = [[],[]]
        for point in range(len(data[0])):
                if len(new[0]) == 0:
                        new[0].append(data[0][point])
                        new[1].append(data[1][point])
                elif ( (new[0][len(new[0])-1] - (data[0][point]))**2 + (new[1][len(new[1])-1] - (data[1][point]))**2  )**0.5 < 10:
                        new[0].append(data[0][point])
                        new[1].append(data[1][point])
                else:
                        matplotlib.pyplot.plot(new[0], new[1], 'g')
                        new[0] = []
                        new[1] = []
                        new[0].append(data[0][point])
                        new[1].append(data[1][point])                        

        new = [[],[]]
        for point in range(len(result[0])):
                if len(new[0]) == 0:
                        new[0].append(result[0][point])
                        new[1].append(result[1][point])
                elif ( (new[0][len(new[0])-1] - (result[0][point]))**2 + (new[1][len(new[1])-1] - (result[1][point]))**2  )**0.5 < 10:
                        new[0].append(result[0][point])
                        new[1].append(result[1][point])
                else:
                        print("New Line Detected(range) : {}".format(( (new[0][len(new[0])-1] - (result[0][point]))**2 + (new[1][len(new[1])-1] - (result[1][point]))**2  )**0.5))
                        matplotlib.pyplot.plot(new[0], new[1], 'b')
                        new[0] = []
                        new[1] = []
                        new[0].append(result[0][point])
                        new[1].append(result[1][point])

        matplotlib.pyplot.axis('square')
        print("Red : {}".format(image0))
        print("Green : {}".format(image1))
        print("Blue : Result")         
        matplotlib.pyplot.show()
       

if __name__ == "__main__":
        main("map00.png","map01.png")
