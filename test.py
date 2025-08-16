import pandas as pd
from functions import *
import numpy as np
import random


df = pd.read_csv(r"iris\versions\2\Iris.csv")
df.dropna(inplace=True)

# print(df.dropna(inplace=True))

# result = train_test_split(df,0.3)

# train = result['train']
# # # print(len(train))


# test = result['test']
# print(len(test))

dimention = len(df['Species'].unique())
# print(dimention)

# Id = df['Id']
xs_const = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = np.array(df['Species'])

# for i in df.columns :
#     print(df[i].unique())

encode = {}
arr = []
for i in range(dimention) :
    arr.append(0)
    
# print(arr)

for i in range(dimention) :  #not sure it has to be array though
    my_arr = arr[:]
    my_arr[i] = 1
    encode[df['Species'].unique()[i]] = np.array(my_arr)
    
    # print(df['Species'].unique()[i])

# print(y)

# for i in range(len(y)) :
#     y[i] = encode[y[i]]

# print(y)

# statement = y==df['Species'].unique()[i]
# y[statement] = my_arr

# print(y[y=='Iris-setosa'])

# print(encode['Iris-virginica'])


xs = xs_const.copy()

# print('old :',xs[0:1].values)

n_attribute = len(xs[0:1].values[0])

w_sum = {}
for i in range(dimention) :
    w = []
    for j in range(n_attribute) :   #random ครั้งแรก  ทำครั้งเดียว
        w.append(random.randint(-10,10))
    w_sum[i] = w

step = 10


    

print('w :',w_sum)

# print('old :',xs_const[0:1])
# print('new :',xs[0:1].values[0]*w)
# print('diff :',xs[0:1].values[0]*w-xs_const[0:1])



def guess_weight (adjust_times,index) :
    
    vector_of_values = []
    
    for output_num in range(dimention):

        value = np.sum(sum(xs[index:index+1].values*w_sum[output_num]))
        
        print('value :',value)
        
        vector_of_values.append(value)
    
    print('value_but_metrix',vector_of_values)
    
    print('softmax :',soft_max(vector_of_values))
    
    print("\n")
    
        
    return None





for i in range(150) :
    new_w = guess_weight(10,i)
    # print(new_w)
    
        
    # for j in range(n_attribute) :
    #     w_sum[j] += new_w[j]
    
# print(np.array(w_sum)/150)



# print('old :',xs_const[0:1])
# print('new :',xs[0:1].values[0]*w)
# print('diff :',xs[0:1].values[0]*w-xs_const[0:1])