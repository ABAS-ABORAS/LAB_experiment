
# coding: utf-8

# # Experiment 1 : Linear Regression and Gradient Descent

# In[4]:


# Step1: Import Some Library
    
import sklearn.datasets as SK
import numpy as nupy            # import numpy library
import pylab as PLO             # import pylab library to Darw Graph
from sklearn.model_selection import train_test_split # import train_test_split to  Devided The Datase

class Linear(object):
    def __init__(item):#
        item.w= nupy.zeros((14,1))

    def updata(item,stepsize,x,y):
        y = nupy.array(y).reshape((y.size, 1))
        item.pred_y = nupy.array(nupy.dot(x,item.w))
        item.loss=sum(1/(y.size)*(item.pred_y-y)**2)
        derivative=-nupy.dot(x.T,y)+nupy.dot(nupy.dot(x.T,x),item.w)#Derivation loss Function
        item.w = item.w - derivative * stepsize  # Update The Weight Parameters

    def test_loss(item,x,y):
        y=nupy.array(y).reshape((y.size,1))
        item.pred_y = nupy.array(nupy.dot(x,item.w))
        item.testloss=sum(1/(y.size)*(item.pred_y-y)**2)


def run():
    
    #  *****************    load DATA Set *************************
    [x, y] = SK.load_svmlight_file("C:/ABAS/DATA/housing_scale.txt")
    x = x.todense()
    b = nupy.ones(506)  # Number Of Sample = 506 samples
    x = nupy.column_stack((x, b))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.44, random_state = 42)
    
    
    #************************** Training DATA **********************
    example=Linear()
    loss_train=nupy.zeros(100) # Initialize with zeros
    loss_test=nupy.zeros(100)  # Initialize with zeros
    for i in range(100):
        example.updata(0.0001,x_train,y_train) # The value of learning rate = 0.0001
        loss_train[i]=example.loss
        example.test_loss(x_test,y_test)
        loss_test[i]=example.testloss
    M=nupy.arange(100)
    

  #****************  Drawing graph of L_train and L_validation  with the number Of iterations  ****************

    plot1 = PLO.plot(M,loss_test[0:100],color="G",label='Data Test')
    plot2 = PLO.plot(M,loss_train[0:100],color="R", label='Data Train')
   
    PLO.ylabel('Cost')
    PLO.xlabel('Iterations')
    print("************************************************************* ")
    print("This Result Experiment Linear Regression and Gradient Descent ")
    print("************************************************************* ")
    PLO.title("The GRAPH Of Number iterations")
    PLO.legend()
    PLO.show()


if __name__ == '__main__':
    run()

