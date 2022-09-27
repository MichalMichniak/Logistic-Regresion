import numpy as np
import Logistic_Reg
import matplotlib.pyplot as plt


def main():
    x = [[np.random.uniform()*10 - 5,np.random.uniform()*10 -5] for i in range(100)] # np.random.uniform()
    x1 = [[np.random.uniform()*10,np.random.uniform()*10] for i in range(100)]
    y = [1 for i in range(100)]
    y1 = [0 for i in range(100)]
    x = np.array(x+x1)
    y = np.array(y+y1)
    #plt.plot(x[:,0],x[:,1],".b")
    lr = Logistic_Reg.Logistic_reg(0.1,2)
    lr.teach(x,y,10000)
    m,c = lr.get_line_param()
    print(m,c)
    plt.plot([0,100],[c,100*m + c])
    for i in x:
        if lr.predict(i) == 1:
            plt.plot(i[0],i[1],".r")
        else:
            plt.plot(i[0],i[1],".b")
    plt.show()
    pass

if __name__ == '__main__':
    main()