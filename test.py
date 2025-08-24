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

print('w :',w_sum)

# print('old :',xs_const[0:1])
# print('new :',xs[0:1].values[0]*w)
# print('diff :',xs[0:1].values[0]*w-xs_const[0:1])
w_diff = {}


# print(w_diff)


def guess_weight (adjust_times,rows,adjust_rate) :
    
    y_predict = []
    loss = []
    
    for loop_num in range(adjust_times) :
        # print(adjust_times)
        for outputs in range(dimention) :
            for inputs in range(n_attribute) :
                w_diff[f'input_node : {inputs}, output_node : {outputs}'] = []
        
        for index in range(rows) :
        
            vector_of_y_values = []
            
            for output_num in range(dimention):
                
                # print('\n',xs[index:index+1].values)
                
                xs_times_weights = (xs[index:index+1].values*w_sum[output_num])[0]

                y_value = sum(xs_times_weights)
                
                # print('value :',value)
                
                vector_of_y_values.append(y_value)
                
                # print(vector_of_y_values)
                
                
                for input_num,inputs in enumerate(xs[index:index+1].values[0]) :
                    
                    delta_y = xs_times_weights[input_num]-y_calculate[index][output_num]
                    # print(delta_y)
                    
                    if delta_y == 0 :
                        adjust_w = 0
                    else :
                        adjust_w = inputs*(delta_y/abs(delta_y))*adjust_rate
                    # print(adjust_w)
                    w_diff[f'input_node : {input_num}, output_node : {output_num}'].append(adjust_w)
                    
                    
                    # 
                    

                    
                    # print('yf each :',y_calculate[index][output_num],'x each :',inputs,"what use :",y_calculate[index][output_num]*dimention / (inputs))
                    # print('prefered weight :',wf)
                
                # print(w_sum)
        # print(sum(w_diff['input_node : 1, output_node : 0'])/rows)
        for input_num in range(n_attribute) :
            for output_num in range(dimention) :
                
                w_sum[output_num][input_num] = w_sum[output_num][input_num] - (sum(w_diff[f'input_node : {input_num}, output_node : {output_num}'])/rows)
        
                
                
            
            # print('value_but_metrix',vector_of_y_values)
        

    
    
        
    # print(soft_max(vector_of_y_values))
    # print(list(y_calculate[0:1][0]))
    # print(soft_max(vector_of_y_values)==list(y_calculate[0:1][0]))
        

        
    # print('softmax :',y_predict)

    # print(w_sum)
    
    # print('w_diff :',w_diff)
    

    # print("\n")
    
    
    return w_sum






model = guess_weight(1000,150,0.1)


# print(soft_max(new_w))

# print(model)

r = 0
while r != '' :
    r = int(input('row to check :'))
    a = []
    for i in model :
        x = xs[r:r+1].values[0]
        a.append(sum(model[i]*x))

    print('predict :',soft_max(a))
    print('true :',y_calculate[r:r+1][0])


# print(np.array(w_sum)/150)



# print('old :',xs_const[0:1])
# print('new :',xs[0:1].values[0]*w)
# print('diff :',xs[0:1].values[0]*w-xs_const[0:1])