import json,ijson,re,numpy
from math import sqrt,pow
from pprint import pprint
from scipy.stats import pearsonr
import  scipy.sparse as sparse
import scipy.linalg as linalg
import  multiprocessing as mp
from scipy import linalg, mat, dot
from recsys.evaluation.ranking import MeanAveragePrecision
from sklearn.metrics import precision_recall_fscore_support
try:
    import cPickle as pickle
except:
    import pickle
    
def main():
    train_data  =   get_data()
    test_data   =   get_data(opt=1) 
    pu,qi       =   get_matrix()
    userSet,itemSet=get_set()
    print "initilization done" 
############################################
    sim = classfier(train_data,test_data,pu,qi,userSet,itemSet)
    sim.train()
###########################################
def get_matrix():
    with open("./defactor.pickle",'rb') as matrix:
        pu,qi=pickle.load(matrix)
    return pu,qi

def get_set():
    with open("./set.pickle",'rb') as Set:
        userSet,itemSet=pickle.load(Set)
    return userSet,itemSet

def get_data(opt=0):
    try:
        if opt  ==  0:    
            with open("./test_purchase.txt",'rb') as train_rating:
                next(train_rating)
                result  =   []
                for obj in train_rating:
                    user,item    =   obj.split('-')
                    user    =   user.strip('U')
                    item    =   item.strip('I')
                    item    =   item.strip('\n') 
                    result.append([user,item])
        
        else:
            with open("./test_purchase_label.txt",'r') as test_rating:
                next(test_rating)
                result  =   []
                temp    =   []

                for line in test_rating:
                    (user,content)  =   line.split("-",1)
                    (item,rating)   =   content.split(",",1)
                    score          =   rating.strip("\n")
                    user    =   user.strip('U')
                    item    =   item.strip('I')
                    (buy,ranking)   =   score.split(",",1)
                    result.append([user,item,buy,ranking])
               
        return  result

    except IOError as ioerr:
        print("can't find the documents"+str(ioerr)+"\n")
        return(None)
def get_info():
    with open('','rb') as info:
        pu,qi   =   pickle.dump(info)


class classfier():
    def __init__(self,train_data,test_data,pu,qi,userSet,itemSet):
        self.trainfile  =   train_data
        self.testfile   =   test_data
        self.userList       =   self.getUser()        
        self.itemList       =   self.getItem()
        self.purchase       =   self.getPurchase()
        self.rating         =   self.getRating()
        self.pu         =   pu
        self.qi         =   qi
        self.userSet    =   userSet
        self.itemSet    =   itemSet
    def getUser(self):
        test   =   self.testfile
        temp    =   [elem[0] for elem in test]
        users    =   []
        for elem in temp:
            if elem not in users:
                users.append(elem)
        return users
    def getItem(self):
        test   =   self.testfile
        items    =   [elem[1] for elem in test]
        return items
    def getPurchase(self):
        test   =   self.testfile
        purchase    =   [elem[1] for elem in test]
        return purchase

    def getRating(self):
        test   =   self.testfile
        rating    =   [elem[1] for elem in test]
        return rating

    def train(self):
        users=   self.userList
        items=   self.itemList
        pu  =   self.pu
        qi  =   self.qi
        userSet     =   self.userSet
        itemSet     =   self.itemSet
        test_data   =   self.testfile
        train_data  =   self.trainfile
        result      =   []
        buy         =   []
        print ("start to rank")
        for user in users:
            rank    =   []
            score   =   []
            tag     =   []
            rating  =   0
            for elem in train_data:
                cnt =0
                if elem[0] == user:
                    if user in userSet:
                        cnt+=1
                        vec_user    =   pu[userSet.index(elem[0])]
                        vec_item    =   qi[itemSet.index(elem[1])]
                        pred_rating  =   dot(vec_user,vec_item)
                        cosine  =pred_rating/linalg.norm(vec_user)/linalg.norm(vec_item)
                        score.append(cosine)
                        rating+=pred_rating 
            print("calculate cosine similarity")
            temp=sorted(score)
            #find ranking 
            print("finish classfication")
            for elem in score:
                rank.append(temp.index(elem))
            result.append(rank)
            if cnt >1:
                avg=rating/cnt
            # buy or not 
            for elem in train_data:
                if elem[0] == user:
                    if user in userSet:
                        vec_user    =   pu[userSet.index(elem[0])]
                        vec_item    =   qi[itemSet.index(elem[1])]
                        pred_rating  =   dot(vec_user,vec_item)
                        if pred_rating <= avg:
                            tag.append(0)
                        else:
                            tag.append(1)
            MAP(rank)
            metric(tag,rank)

    def MAP(self,rank):
        
        Map = MeanAveragePrecision()
        for elem in rank:
            Map.load(self.rating,elem)
        result=Map.computr()
        print "the MAP of ranking :" ,result
        return result
    def metric(self,tag,rank):
        
        precision,recall,fbeta,support  =   precision_recall_fscore_support(self.purchase,tag)
        print "precision of purchase:",precision
        print "recall of purchase:",recall

        PAN,a,b,c                      =   precision_recall_fscore_support(self.rating,rank)
        
        print "P @ N :",PAN

if __name__ == '__main__':
    main()

