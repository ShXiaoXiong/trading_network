import csv
import numpy
import scipy.special #激活函数

#准备实际数据
data_file=open('data_restructured.csv','r')#每个新行表示一个新的数据库行，每个数据库行由一个或多个以逗号分隔的字段组成
data_list=data_file.readlines()#转换为列表
data_list = [x.strip() for x in data_list if x.strip() != '']#不等于空就移除字符串头尾指定的字符（默认为空格或换行符）
data_list = [x.strip('"') for x in data_list if x.strip() != '']#不等于空就移除字符串头尾指定的字符（默认为空格或换行符）

data_file.close()

#准备target数据
with open('train.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    column1 = [row[2]for row in reader]
closing_prices=column1[1:]#去除标签，保持一致

#打开神经网络数据
f1=open("in_matrix.csv","rb")
matrix1=numpy.loadtxt(f1,delimiter=',',skiprows=0)
f1.close()
ihw=numpy.array(matrix1)

f2=open("out_matrix.csv","rb")
matrix2=numpy.loadtxt(f2,delimiter=',',skiprows=0)
f2.close()
how=numpy.array(matrix2)

activation_function=lambda x:scipy.special.expit(x)

scoreboard=[]
accuracies=[]
    
actions=[]
net_values=[]

t=251#第一个预测值的位置
cash=6.99#t251的收盘价
equity=0

#计算
for record in data_list:

    all_values=record.split(',')    
    inputs=numpy.asfarray(all_values)
    
    inputs=numpy.array(inputs,ndmin=2).T#传递列表，转换为二维数组，转置

    #计算过程
    hidden_inputs=numpy.dot(ihw,inputs)#点乘
    hidden_outputs=activation_function(hidden_inputs)#使用激活函数

    final_inputs=numpy.dot(how,hidden_outputs)#点乘
    final_outputs=activation_function(final_inputs)#使用激活函数

    
    #对比，记录结果1：神经网络输出
    networks_label=numpy.argmax(final_outputs)#取出最大值对应的索引值，0或1

    closing_0=numpy.array(closing_prices[t],ndmin=2)
    closing_1=numpy.array(closing_prices[t+1],ndmin=2)    

    actions.append(networks_label)

    if closing_1>=closing_0:#涨了，应当满仓
        correct_label=1
    else:#跌了，应该空仓
        correct_label=0

    if networks_label==correct_label:
        scoreboard.append(1)
    else:
        scoreboard.append(0)

    #记录结果2，预测正确率
    accuracy= sum(scoreboard)/len(scoreboard)
    accuracies.append(accuracy)

    #记录输出3：净值
    if networks_label==0 and cash==0: #按前一个交易日价格卖出
        cash = equity
        equity=0
    if networks_label==0 and cash!=0:#保持空仓
        net_value += 0
    if networks_label==1 and cash==0:#保持满仓
        equity = equity*float(closing_prices[t+1])/float(closing_prices[t]) #重新赋值
    if networks_label==1 and cash!=0:#按前一个交易日价格买入
        equity=cash/float(closing_prices[t])*float(closing_prices[t+1])
        cash=0

    net_value=cash + equity
    net_values.append(net_value)

    t+=1
    
#输出结果

import matplotlib.pyplot as plt

plt.subplot(311)
plt.title('Accuracies of Predictor_20180112-20200108')
y=accuracies
x=range(len(accuracies))
plt.plot(x,y,linewidth=2,color='r',marker='o', markerfacecolor='yellow',markersize=3) 
 
plt.subplot(312)
plt.title('Outputs of Predictor')
y=actions
x=range(len(actions))
plt.plot(x,y,linewidth=2,color='r',marker='o', markerfacecolor='yellow',markersize=3) 

with open('train.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    column1 = [row[2]for row in reader]
closing_prices=column1[1:]#去除标签，保持一致
closing_prices=closing_prices[252:]#t+1
closing_prices=list(map(float, closing_prices))
real_closing_prices=[]	
for jj in closing_prices:
	jj=10*jj
	real_closing_prices.append(jj)	


#图一
plt.subplot(313)
plt.title('Price from 6.99 to 5.08 VS Net Value from 6.99 to 52.40')
x1=range(len(closing_prices))
plt.plot(x1,real_closing_prices,linewidth=2,color='g',marker='o', markerfacecolor='blue',markersize=3)
x2=range(len(net_values))
plt.plot(x2,net_values,linewidth=2,color='r',marker='o', markerfacecolor='red',markersize=3) 
plt.show()

