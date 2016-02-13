from __future__ import division
from random import choice 
import numpy as np 
from pylab import * 

class Perceptron:
    def __init__(self,w=None):
        if w is None:
            w           =   np.random.rand(15)
            self.w      =   w
        self.rate       =   0.2
        self.errors     =   []
        self.testError  =   []
        self.w_vector   =   []
    def update(self,x,error):
        self.w  +=  self.rate*error*x
        self.w_vector.append(self.w)

    def train(self,data,n):
        step_function   = lambda y: -1 if y<0 else 1
        for i in xrange(n):
            x,label     =   choice(data)
            result      =   dot(self.w, x)
            error       =   label - step_function(result)
            self.errors.append(error)
            self.update(x,error)      
    def hypothesis(self,data,n):
        step_function   = lambda y: -1 if y<0 else 1
        #for voted hypothesis
        transpose     =   map(list, zip(*self.w_vector))
        arr     =   np.asarray(transpose)
        axis    =   1
        u, indices = np.unique(arr, return_inverse=True)
        vote    =   u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(arr.shape),None, np.max(indices) + 1), axis=axis)]
        for i in xrange(n):
            x,label     =   choice(data)
            
            #last hypothesis
            #last_w      =   self.w_vector[999]
            #result      =   dot(last_w,x)
            #predict     =   step_function(result)
            #diff        =   label - predict
            #self.testError.append(diff)
        
            #average hypothesis
            #average     =   np.mean(self.w_vector,axis=0) 
            #result      =   dot(average,x)   
            #predict     =   step_function(result)
            #diff        =   label - predict
            #self.testError.append(diff)
            
            #last epoches average
            #last_w_vector   =   self.w_vector[500:1000:1]
            #average     =   np.mean(last_w_vector,axis=0) 
            #result      =   dot(average,x)   
            #predict     =   step_function(result)
            #diff        =   label - predict
            #self.testError.append(diff)
           
            #voted hypothesis
            result      =   dot(vote,x)
            predict     =   step_function(result)
            diff        =   label - predict
            self.testError.append(diff)


            #last epoches vote
            #last_w_vector   =   self.w_vector[500:1000:1]
            #signList=[]
            #for i in range(500):
            #    product     =   dot(last_w_vector[i],x)
            #    temp        =   step_function(product)
            #    signList.append(temp)
            #predict         =   step_function(sum(signList))
            #diff            =   label - predict
            #self.testError.append(diff)

def sign(x): return 1 if x >= 0 else -1

def genData(n,test=False):
    noise = np.random.normal(0,1,15)
    
    if test:
        testSet   =   []
    else:    
        traingSet =   []
    for i in xrange(n):
        vector  =   np.random.choice([-1,1],15)
        #for problem a
        #label   =   vector[0] 
        
        # for problem b
        #count   =   np.sum(vector) 
        #label   =   sign(count)
        
        #for problem c
        new_vec =   vector[0:12:1] 
        portion =   np.sum(new_vec)  
        r       =   choice([-4,4])
        count   =   r + portion
        label   =   sign(count)    
        if test:
            vector+=noise
            for i in xrange(len(vector)):
                if vector[i] > 1: vector[i] = 1
                if vector[i] <-1: vector[i] =-1
            testSet.append((vector , label))
        
        else:
            traingSet.append((vector , label))
    if test:
        return testSet
    else:
        return traingSet

def main():
    #for training set########################################################
    iteration       =   500
    trainingData    =   genData(iteration)
    nueron          =   Perceptron()
    #set the number of epoches for training set
    epochs          =   2
    while(epochs):
        nueron.train(trainingData,iteration)
        epochs-=1
    
    #num_erro_1=[i for i, e in enumerate(nueron.errors[0:499]) if e != 0]
    #num_erro_2=[i for i, e in enumerate(nueron.errors[500:999]) if e != 0]
    #num_erro_3=[i for i, e in enumerate(nueron.errors[1000:1499]) if e != 0]
    #num_erro_4=[i for i, e in enumerate(nueron.errors[1500:1999]) if e != 0]
    
    #print(len(num_erro_1))
    #print(len(num_erro_2))
    #print(len(num_erro_3))
    #print(len(num_erro_4))

    #display errors in graph 
    #ylim([-2,2])
    #plot(nueron.errors)
    #show()    
    
    #for test set################################################
    epochs      =   10
    testData    =   genData(iteration,True)

    while(epochs):
        nueron.hypothesis(testData,iteration)
        epochs-=1
    
    accuracy    =  nueron.testError.count(0)/len(nueron.testError)
    print(accuracy) 
    #ylim([-2,2])
    #plot(nueron.testError)
    #show()    

if __name__ == '__main__':
    main()



