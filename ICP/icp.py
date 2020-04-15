import glob
import cv2
import numpy
import random
import matplotlib.pyplot
import copy
import pylab
import time
import sys
import sklearn.neighbors
import scipy.optimize

def icp(a, b, max_time = 1):
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

def convert(img):
        output = [[],[]]
        img = cv2.resize(img,(400,400))
        for x,v_x in enumerate(img):
                for y,v_y in enumerate(v_x):
                        if sum(v_y) != 0:
                                output[0].append(x)
                                output[1].append(y) 
        return numpy.array(output)

def merge():

        fname = glob.glob('*.png')
        images = []
        for name in fname:
                images.append(cv2.imread(name))

        transforms = []     
        for index in range(1, len(images)):
                T,error = icp(convert(images[index]),convert(images[index-1]),max_time=60)
                transforms.append(T)
        
        template = cv2.resize(images[0],(400,400))
        template = numpy.concatenate((template, numpy.zeros(shape=template.shape, dtype=template.dtype)), axis=1)
        template = numpy.concatenate((template, numpy.zeros(shape=template.shape, dtype=template.dtype)), axis=0)           

        for inx,img in enumerate(images[1:]):
                image = convert(img)
                transform = transforms[0]
                for i in range(1, inx+1):
                        transform = transform.dot(transforms[i])
                image = cv2.transform(numpy.array([image.T], copy=True).astype(numpy.float32), transform).T
                cv2.imshow('Before Merge', template)
                cv2.waitKey()
                for i in range(len(image[0])):  
                        template[int(image[0][i][0]), int(image[1][i][0])] = 255
                cv2.imshow('Merged!', template)
                cv2.waitKey()
        
        cv2.destroyAllWindows()
        cv2.imshow('Result', template)
        cv2.waitKey()

if __name__ == "__main__":
        merge()