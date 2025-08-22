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

y_calculate = []

for i in range(len(y)) :
    y_calculate.append(encode[y[i]])

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


# print(y_calculate)

# print('w :',w_sum)

# print('old :',xs_const[0:1])
# print('new :',xs[0:1].values[0]*w)
# print('diff :',xs[0:1].values[0]*w-xs_const[0:1])
w_diff = {}
for outputs in range(dimention) :
    for inputs in range(4) :
        w_diff[f'input_node : {inputs}, output_node : {outputs}'] = []

# print(w_diff)


def guess_weight (adjust_times,rows) :
    
    y_predict = []
    loss = []
    
    
    for index in range(rows) :
    
        vector_of_values = []
        
        for output_num in range(dimention):
            
            # print('\n',xs[index:index+1].values)
            
            xs_times_weights = (xs[index:index+1].values*w_sum[output_num])[0]

            value = sum(xs_times_weights)
            
            # print('value :',value)
            
            vector_of_values.append(value)
            
            for input_num,inputs in enumerate(xs[index:index+1].values[0]) :
                
                wf = y_calculate[index][output_num]*dimention / (inputs)
                
                w_diff[f'input_node : {input_num}, output_node : {output_num}'].append(wf)
                
                # print('yf each :',y_calculate[index][output_num],'x each :',inputs,"what use :",y_calculate[index][output_num]*dimention / (inputs))
                # print('prefered weight :',wf)
        
        # print('value_but_metrix',vector_of_values)
        
        # sf = soft_max(vector_of_values)
        
        
        
        
        # y_predict.append(sf)
        
    # print('softmax :',y_predict)

        # print(w_sum)
    
    # print('w_diff :',w_diff)
    
    w_final = []
    
    for i in w_diff :
        # print(i)
        w_final.append(sum(w_diff[i])/rows)
        # print("\n")

    
    for i in range(rows) :
        # print(w_final)
        final_input = list(xs_const[i:i+1].values[0])*3
        # print(final_input)
        
        model = np.array(w_final)*np.array(final_input)
        
        # print(ans)
        ans = []
        
        for i in range(dimention) :
            ans.append(sum(model[4*i:4*(i+1)])/(dimention*4))
            # print(f'{4*i},{4*(i+1)}')

        # print(ans)
        # print(soft_max(ans))
        print(ans[0]>ans[1]and ans[1]>ans[2])
    
    # print("\n")
    
    
    return None






new_w = guess_weight(10,150)

# print(w_diff)


    # print(new_w)
    
        
    # for j in range(n_attribute) :
    #     w_sum[j] += new_w[j]
    
# print(np.array(w_sum)/150)



# print('old :',xs_const[0:1])
# print('new :',xs[0:1].values[0]*w)
# print('diff :',xs[0:1].values[0]*w-xs_const[0:1])