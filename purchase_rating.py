import json,ijson,re,numpy
from math import sqrt,pow
from pprint import pprint
from scipy.stats import pearsonr
import  scipy.sparse as sparse
import scipy.linalg as linalg
import  multiprocessing as mp
import random
from recsys.evaluation.prediction import RMSE
try:
    import cPickle as pickle
except:
    import pickle
    
def main():
    train_data  =   get_temp() 
    test_data   =   get_data(opt=1)
    ###############################
    #start SVD
    svd_init    =   mp.Pool(4)
    svd         =   svd_init.apply(SVDtrainer,args=(train_data,test_data)) 
    #with open("./set.pickle",'wb') as myset:
    #        pickle.dump([svd.userSet,svd.itemSet],myset)
    pu,qi,rmse  =   svd.train()
    #svd_cal     =   mp.Pool(4)
    #pu,qi,rmse  =   svd_cal.apply(svd.train)
    with open("./defactor.pickle",'wb') as myfacotr:
                pickle.dump([pu,qi,rmse],myfacotr,protocal=2)


def get_temp(opt=0):
    try:
        result   =   {}
        if opt  ==  0:
            with open("./sample.json",'rb') as train_rating:
                for obj in train_rating:
                    result=json.loads(obj) 
        return  result
    except IOError as ioerr:
        print("can't find the documents"+str(ioerr)+"\n")
        return(None)

def get_data(opt=0):
    try:
        if opt  ==  0:    
            with open("./train_rating.json",'rb') as train_rating:
                result  =   {}
                for obj in train_rating:
                    dictObj=json.loads(obj)
                    reviewerID  =   dictObj["reviewerID"].strip('U')
                    itemID      =   dictObj["itemID"].strip('I')
                    rating      =   dictObj["rating"]
                    if reviewerID not in result:
                        result[reviewerID]={} 
                    result[reviewerID][itemID]   =   rating
            with open("./sample.json","wb") as outfile:
                json.dump(result,outfile)      
        else:
            with open("./test_rating_label.txt",'r') as test_rating:
                result  =   []
                for line in test_rating:
                    (user,content)  =   line.split("-",1)
                    (item,rating)   =   content.split(",",1)
                    rating          =   rating.strip("\n")
                    obj =[user,item,rating]
                    result.append(obj)
            return  result

    except IOError as ioerr:
        print("can't find the documents"+str(ioerr)+"\n")
        return(None)

class SVDtrainer():
    def __init__(self,train_data,test_data,factorNum=100):
        #training set file
        self.trainfile  =train_data
        self.user       =train_data.keys()
        self.item       =self.getItem(train_data)
        #testing set file
        self.testfile=test_data
        #get factor number
        self.factorNum=factorNum
        #get user number
        self.userNum=self.getUserNum()
        #get item number
        self.itemNum=self.getItemNum(train_data)
        #learning rate
        self.learningRate=0.01
        #the regularization lambda
        self.regularization=0.05
        #initialize the model and parameters
        self.init()

    def init(self):
        self.meanValue  =   self.mean(self.trainfile)
        self.bu         =   [0.0 for i in range(self.userNum)]
        self.bi         =   [0.0 for i in range(self.itemNum)]
        temp            =   sqrt(self.factorNum)
        self.pu=[[(0.1*random.random()/temp) for i in range(self.factorNum)] for j in range(self.userNum)]
        self.qi=[[0.1*random.random()/temp for i in range(self.factorNum)] for j in range(self.itemNum)] 
        print "Initialize end.The user number is:%d,item number is:%d,the average score is:%f" % (self.userNum,self.itemNum,self.meanValue)
           
    def getUserNum(self):
        userSet=set()
        for elem in self.user:
            if elem not in userSet:
                userSet.add(elem)
        self.userSet = list(userSet)
        return len(userSet)
    def getItem(self,train_data):
        temp=[itemID.keys() for itemID in train_data.values()]
        item=[item for sublist in temp for item in sublist]
        return item
    def getItemNum(self,train_data):
        itemSet=set()
        for elem in self.item:
            if elem not in itemSet:
                itemSet.add(elem)
        self.itemSet=list(itemSet)
        return len(itemSet)
    
    def mean(self,train_data):
        result  =   0.0
        #print("create value")
        value        =   train_data.values()
        dataList    =   [ elem.values() for elem in value ]
        data        =   numpy.array([data for sublist in dataList for data in sublist])
        for i in data:
            result+=i
        #print float(result/len(data))
        return float(result/len(data))
    #iter 223272
    def train(self,iterTimes=1000):
        print "Beginning to train the model......"
        trainfile=self.trainfile
        userSet =self.userSet
        itemSet =self.itemSet
        users    =   [elem for elem in userSet for i in range(len(trainfile[elem]))]
        self.userList   =   users
        #users   =   [user  for user in trainfile.keys() for i in range(len(trainfile.values()))]
        values  =   trainfile.values()
        temp    =   [itemID.keys() for itemID in values]
        items    =   [item  for sublist in temp for item in sublist]
        self.itemList   =   items

        data_list    =   [ elem.values() for elem in values]
        data        =   numpy.array([data for sublist in data_list for data in sublist])
        self.dataList   =   data
        preRmse=10000.0
        for idx in range(iterTimes):
            print "times:",idx
            #user and item are entries to the row and column
            user=   userSet.index(users[idx])
            item=   itemSet.index(items[idx])
            rating= data[idx]
            
            #calculate the predict score
            pscore=self.predictScore(self.meanValue,self.bu[user],self.bi[item],self.pu[user],self.qi[item])
            #the delta between the real score and the predict score
            eui=rating-pscore    
            #update parameters bu and bi(user rating bais and item rating bais)
            self.bu[user]+=self.learningRate*(eui-self.regularization*self.bu[user])
            self.bi[item]+=self.learningRate*(eui-self.regularization*self.bi[item])
            for k in range(self.factorNum):
                temp=self.pu[user][k]
                #update pu,qi
                self.pu[user][k]+=self.learningRate*(eui*self.qi[user][k]-self.regularization*self.pu[user][k])
                self.qi[item][k]+=self.learningRate*(temp*eui-self.regularization*self.qi[item][k])
            
            #calculate the current rmse
            curRmse=self.test(self.meanValue,self.bu,self.bi,self.pu,self.qi,idx)
            print "Iteration %d times for train,RMSE is : %f" % (idx+1,curRmse)
            if curRmse>preRmse:
                break
            else:
                preRmse=curRmse
            print "Iteration finished!"
            
        curRmse=self.test(self.meanValue,self.bu,self.bi,self.pu,self.qi,idx,opt=1)
        print "Iteration done,RMSE is : %f" % (curRmse)
 
        return self.pu, self.qi, curRmse

    def test(self,meanValue,bu,bi,pu,qi,idx,opt=0):
        print("start test")
        testfile=self.testfile
        userSet=self.userSet
        itemSet=self.itemSet
        users=self.userList
        items=self.itemList
        data=self.dataList
        rmse=0.0
        cnt=0
        offset  =   223272
        if opt==0:
                cnt+=1
                user=   userSet.index(users[idx+offset])
                item=   itemSet.index(items[idx+offset])
                rating= data[idx+offset]
                pscore=self.predictScore(meanValue,bu[user],bi[item],pu[user],qi[item])
                elem_rmse=pow(float(rating)-float(pscore),2)
                rmse+=elem_rmse

        elif opt==1:
            for test in testfile:
                cnt+=1
                if test[0] in userSet:
                    user=userSet.index(test[0])
                    item=itemSet.index(test[1])
                    score=test[2]
                    pscore=self.predictScore(meanValue,bu[user],bi[item],pu[user],qi[item])
                    elem_rmse=pow(float(score)-float(pscore),2)
                    print "rmse: ",elem_rmse
                    rmse+=elem_rmse
        return sqrt(rmse/cnt)

    def innerProduct(self,v1,v2):
        result=0.0
        for i in range(len(v1)):
            result+=v1[i]*v2[i]
        return result
    def predictScore(self,meanValue,bu,bi,pu,qi):
        pscore=meanValue+bu+bi+self.innerProduct(pu,qi)
        if pscore<1:
            pscore=1
        if pscore>5:
            pscore=5
        return pscore

if __name__ == '__main__':
    main()
