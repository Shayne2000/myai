import pandas as pd
from functions import *
import numpy as np
import random
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

def browse_files():
    global df
    
    filename = filedialog.askopenfilename(
        title="Select a File",
        initialdir=r"C:\Users\Lenovo\OneDrive\Desktop\aibuild\iris\versions\2\Iris.csv",
        filetypes=(
            ("CSV files", "*.csv"),  # Filter for specific file types
            ("All files (not recoment)", "*.*")
        )
    )
    if not(filename):
        print("file can't open")
        
    df = pd.read_csv(filename)
    df.dropna(inplace=True)
    
    label_showPart.config(text=filename)
    
    comboboxOption_columns = ['ID','LABEL','FEATURES','']
    column_list = {}
    
    for index,value in enumerate(df.columns) :
        comboboxOptionIndex = tk.StringVar()
        tk.Label(screen,text=value).grid(row=index+2,column=0,pady=10)
        obj = ttk.Combobox(screen,values=comboboxOption_columns,textvariable=comboboxOptionIndex)
        obj.grid(row=index+2,column=1)
        column_list[obj] = value
        
    selectData_button = tk.Button(screen,text="SELECT DATA",command=lambda : selectData(column_list,index,selectedColumn_list))
    selectData_button.grid(row=index+3,column=0,pady=10)
    
    selectedColumn_list = {}
    
    
        
    

def selectData (column_list,baseRow,selectedColumn_list) :
    
    feedBack_label.grid(row=baseRow+3,column=1,padx=10,pady=10) #declare in mainprogram
    
    feature_list = []
    label_list = []
    for column in column_list :
        if column.get() == "FEATURES" :
            feature_list.append(column)
        elif column.get() == "LABEL" :
            label_list.append(column)
    
    
    if len(feature_list) <= 0 :
        feedBack_label.config(text="no feature..... select again")
    elif len(label_list) != 1 :
        feedBack_label.config(text="their should be 1 label..... select again")
    else :
        feedBack_label.config(text="select data success.....")
        
        for column in selectedColumn_list :
            selectedColumn_list[column].destroy()
        
        for index,value in enumerate(feature_list) :
            obj = tk.Label(text=column_list[value])
            obj.grid(row=index+baseRow+4,column=0,pady=10)
            selectedColumn_list[index+baseRow+4] = obj
            
            comboboxOption_selectedColumns = ['INTEGER','FLOAT','STRING']
            comboboxOptionIndex = tk.StringVar()
            obj  = ttk.Combobox(screen,values=comboboxOption_selectedColumns,textvariable=comboboxOptionIndex)
            obj.grid(row=index+baseRow+4,column=1,pady=10)
    
    
    


screen = tk.Tk()
header = tk.Label(screen,text="this is ai 123456789",justify=tk.CENTER,font=("Helvetica",20,"bold"))
header.grid(row=0,column=0,padx=10,pady=10,columnspan=2)


file_button = tk.Button(text="select file",command=browse_files)
file_button.grid(row=1,column=0,padx=20)

label_showPart = tk.Label()
label_showPart.grid(row=1,column=1,padx=10,pady=10)

feedBack_label = tk.Label(screen)







df = pd.read_csv(r"iris\versions\2\Iris.csv")
df.dropna(inplace=True)



output_num = len(df['Species'].unique()) #number of classification product

dimentions = [1,2]



xs = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] #x columns
y = np.array(df['Species']) #y column



encode = {}


arr = [] #create sets of 0
for i in range(output_num) :
    arr.append(0)


for i in range(output_num) :  #one hot encoding
    my_arr = arr[:]
    my_arr[i] = 1
    encode[df['Species'].unique()[i]] = my_arr


y_true = []

for i in range(len(y)) :
    y_true.append(encode[y[i]]) #make y an array for classification


n_attribute = len(xs[0:1].values[0]) #numbers if columns

bias = {}
weight = {}

w_diff = {}
b_diff = {}


previous_dimention = n_attribute
for layer in range(len(dimentions)) :

    for furter_node_num in range(dimentions[layer]) :
        for closer_node_num in range(previous_dimention) :   #random ครั้งแรก  ทำครั้งเดียว
            weight[f"(layer,closer,furter) : ({layer},{closer_node_num},{furter_node_num})"] = random.randint(-10,10)
            w_diff[f"(layer,closer,furter) : ({layer},{closer_node_num},{furter_node_num})"] = 0

        bias[f'(layer,node) : ({layer},{furter_node_num})'] = random.randint(-10,10)
        b_diff[f'(layer,node) : ({layer},{furter_node_num})'] = 0
    
    previous_dimention = dimentions[layer]

for furter_node_num in range(output_num) :
    for closer_node_num in range(previous_dimention) :
        weight[f"(layer,closer,furter) : ({len(dimentions)},{closer_node_num},{furter_node_num})"] = random.randint(-10,10)
        w_diff[f"(layer,closer,furter) : ({len(dimentions)},{closer_node_num},{furter_node_num})"] = 0
    
    bias[f'(layer,node) : ({len(dimentions)},{furter_node_num})'] = random.randint(-10,10)
    b_diff[f'(layer,node) : ({layer},{furter_node_num})'] = 0
    







def guess_weight (adjust_times,rows,adjust_rate) :
    
    global output_num
    
    y_predict = []
    loss = []
    
    for loop_num in range(adjust_times) :
        for outputs in range(output_num) :
            for inputs in range(n_attribute) :
                w_diff[f'input_node : {inputs}, output_node : {outputs}'] = []
            b_diff[f'output_node : {outputs}'] = []
            
        print(b_diff)
        for index in range(rows) :
        
            vector_of_y_values = []
            
            for output_num in range(output_num):
                
                ######################################################
                
                #             prediction phase                       #
                
                ######################################################
                
                
                
                # print('\n',xs[index:index+1].values)
                
                xs_times_weights = (xs[index:index+1].values*weight[output_num])[0]

                y_value = sum(xs_times_weights) + bias[f'output_node : {output_num}']
                
                # print('value :',value)
                
                vector_of_y_values.append(y_value)
                
                # print(vector_of_y_values)
                
                
                
                
                
                
                
                
                
                
                
                
                ###########################################################
                
                #      start finding adjust here                          #
                
                ###########################################################
                
                
                for input_num,inputs in enumerate(xs[index:index+1].values[0]) :
                    
                    delta_y = xs_times_weights[input_num]-y_true[index][output_num]
                    
                    # print(y_true[index])
                    # print(delta_y)
                    
                    if delta_y == 0 :
                        adjust_w = 0
                    else :
                        
                        # uq = 1
                        
                        uq = xs[index:index+1].values[0][input_num] #
                        
                        xp = xs_times_weights[input_num] # model prediction
                        
                        yp = y_true[index][output_num] # true value
                        
                        # print(yp)
                        
                        adjust_w = (uq*(xp - yp))*adjust_rate
                        
                        
                        
                    # print(adjust_w)
                    w_diff[f'input_node : {input_num}, output_node : {output_num}'].append(adjust_w)
                    
                    b_diff[f'output_node : {output_num}'].append(adjust_rate*(y_value - y_true[index][output_num]))
                    # print(w_diff)
                    

                    
                    # print('yf each :',y_true[index][output_num],'x each :',inputs,"what use :",y_true[index][output_num]*output_num / (inputs))
                    # print('prefered weight :',wf)
                
        
                
                
                # print(weight)
        # print(sum(w_diff['input_node : 1, output_node : 0'])/rows)
        # print(w_diff)
        for output_num in range(output_num) :
            for input_num in range(n_attribute) :
                
                weight[output_num][input_num] = weight[output_num][input_num] - (sum(w_diff[f'input_node : {input_num}, output_node : {output_num}'])/rows)
                
            bias[f'output_node : {output_num}'] = bias[f'output_node : {output_num}'] - (sum(b_diff[f'output_node : {output_num}'])/rows)
        
                
                
            
            # print('value_but_metrix',vector_of_y_values)
        

    
    
        
    # print(soft_max(vector_of_y_values))
    # print(list(y_true[0:1][0]))
    # print(soft_max(vector_of_y_values)==list(y_true[0:1][0]))
        

        
    # print('softmax :',y_predict)

    # print(weight)
    
    # print('w_diff :',w_diff)
    

    # print("\n")
    
    
    return [weight,bias]






# model = guess_weight(10,150,0.01)


# # print(soft_max(new_w))

# print(model)

# r = 0
# while r != 'q' :
#     r = int(input('row to check :'))
#     a = []
#     for i in model[0] :
#         x = xs[r:r+1].values[0]
#         # print(x,model[i])
#         a.append(sum(model[0][i]*x)+model[1][i])

#     # print("a :",a)
#     print('predict :',soft_max(a))
#     print('true :',y_true[r:r+1][0],"\n")


# print(np.array(weight)/150)



# print('old :',xs_const[0:1])
# print('new :',xs[0:1].values[0]*w)
# print('diff :',xs[0:1].values[0]*w-xs_const[0:1])

tk.mainloop()