import math
import pandas as pd


def ReLU (x) :    # clear negative values make it 0
    return max(0,x)



def sigmoid (x) : #prediction whether it's near 1 or 0       ~-inf , ~inf --> ~0 , ~1
    return 1/(1+math.e**-x)


def train_test_split (df,test) :
    shuffled_df = df.sample(frac=1)
    n = len(df)
    test = round(test*n)
    
    train = n-test
    
    
    return {'train':shuffled_df[0:train],'test':shuffled_df[train:n]}

def soft_max (array) :
    # print(array)
    new_array = []
    for i in array :
        print(i)
        new_array.append(math.e**i)
    print(new_array)
    sum_array = sum(new_array)
    
    # print('weight with no - and power by e',new_array)
    
    ans = []
    for i in new_array :
        ans.append(round(i/sum_array,3))
    
    return ans
    
