import numpy as np #数组功能
import scipy.special #激活函数
import matplotlib.pyplot as plt #可视化
import csv

class Neuralnetworks:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #待传递节点数量
        self.inodes=inputnodes#根据input的量决定
        self.hnodes=hiddennodes#隐藏层强制神经网络进行总结和归纳，代表其能力
        self.onodes=outputnodes#这个案例中，输出的是具体的数字识别
        #待传递学习率
        self.lr=learningrate #过低的学习率限制了步长、限制了梯度下降发生的速度，对性能造成了损害。过高的学习率会导致在梯度下降过程中超调及来回跳动
        #设定初始连接权重：使用正态概率分布采样权重，也可以使用其他更为复杂的方法
        self.ihw=np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.how=np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #设定激活函数:使用sigmoid函数，一个常用的非线性激活函数，接受任何数值，输出0到1之间的某个值，但不包含0和1
        self.activation_function=lambda x:scipy.special.expit(x)
     
        pass

    def query(self,inputs):
        
        hidden_inputs=np.dot(self.ihw,inputs)#点乘
        hidden_outputs=self.activation_function(hidden_inputs)#使用激活函数

        final_inputs=np.dot(self.how,hidden_outputs)#点乘
        final_outputs=self.activation_function(final_inputs)#使用激活函数

        return final_outputs#如果不写return，会返回一个None对象

    def train(self,inputs_list,targets_list):
        #反馈调节权重的过程/反向传播误差——告知如何优化权重
        
        #完全相同的计算，因此在循环中要重写
        inputs=np.array(inputs_list,ndmin=2).T#传递列表，转换为二维数组，转置
        
        hidden_inputs=np.dot(self.ihw,inputs)#点乘
        hidden_outputs=self.activation_function(hidden_inputs)#使用激活函数

        final_inputs=np.dot(self.how,hidden_outputs)#点乘
        final_outputs=self.activation_function(final_inputs)#使用激活函数

        targets=np.array(targets_list,ndmin=2).T#传递列表，转换为二维数组
        
        output_errors=targets-final_outputs#计算误差

        #隐藏层误差
        hidden_errors=np.dot(self.how.T,output_errors)#点乘
        #反向传递，更新how权重
        self.how += self.lr * np.dot((output_errors * final_outputs* (1-final_outputs)),np.transpose(hidden_outputs))#点乘
        #反向传递，更新ihw权重
        self.ihw += self.lr * np.dot((hidden_errors * hidden_outputs* (1-hidden_outputs)),np.transpose(inputs))#点乘
        pass

###数据部分
#读取数据并转化为二维数组，去掉表头和时间
data=np.genfromtxt("original_data.csv",delimiter=",",skip_header=1)[:,1:]
#归一化数据（使用广播）
normalization1=[] 
normalization2=[] 

lie=0
for xx in range(data.shape[1]):
	normalization1.append(np.mean(data[:,lie]))
	normalization2.append(data[:,lie].max()-data[:,lie].min())
	lie+=1

new_data = (data-normalization1)/normalization2

#创建神经网络
n=Neuralnetworks(250*np.shape(data)[1] ,200,2,0.015)

#计分板：每一代的最终净值、两个矩阵
final_scores=[]
in_matrix=[]
out_matrix=[]

#80%断点
duandian=int(0.8*len(data))


for e in range(100):#epochs

    #初始条件
    t=251#第一个预测值的位置
    cash=10000
    equity=0
    net_value=cash + equity

    #计分板
    action=[]
    scoreboard=[]
    accuracies=[]
    net_values=[]

    #训练
    while t <= duandian: #预测器能运动到断点
    
        inputs=new_data[t-249:t+1,:].reshape(1,-1).T#预测器的第一个索引位置是249，输入数据的位置是[0:250]。转置
        
        #计算过程
        hidden_inputs=np.dot(n.ihw,inputs)#点乘
        hidden_outputs=n.activation_function(hidden_inputs)#使用激活函数

        final_inputs=np.dot(n.how,hidden_outputs)#点乘
        final_outputs=n.activation_function(final_inputs)#使用激活函数

        #神经网络输出
        networks_label=np.argmax(final_outputs)#取出最大值对应的索引值，0或1
        action.append(networks_label)
        
        #对比
        closing_0= data[t+1][0]
        closing_1= data[t+2][0]

        if closing_1>=closing_0:
            correct_label=1
        else:
            correct_label=0
  
        if networks_label==correct_label:
            scoreboard.append(1)
        else:
            scoreboard.append(0)

        #计分板1：预测正确率
        accuracy= sum(scoreboard)/len(scoreboard)
        accuracies.append(accuracy)

        #计分板2：净值
        
        if networks_label==0 and cash==0: #按前一个交易日价格卖出
            cash = equity
            equity=0
        if networks_label==0 and cash!=0:#保持空仓
            net_value += 0
        if networks_label==1 and cash==0:#保持满仓
            equity = equity*closing_1/closing_0 #重新赋值
        if networks_label==1 and cash!=0:#按前一个交易日价格买入
            equity=cash/closing_0*closing_1
            cash=0

        net_value=cash + equity
        net_values.append(net_value)

        #target值
        targets=np.zeros(2)+0.01 

        if closing_1>=closing_0:#涨了，应当满仓
            targets[1]=0.99
        else:#跌了，应该空仓
            targets[0]=0.99

        #训练
        targets=np.array(targets,ndmin=2).T#传递列表，转换为二维数组
        output_errors=targets-final_outputs#计算误差

        #隐藏层误差
        hidden_errors=np.dot(n.how.T,output_errors)#点乘
        #反向传递，更新how权重
        n.how += n.lr * np.dot((output_errors * final_outputs* (1-final_outputs)),np.transpose(hidden_outputs))#点乘
        #反向传递，更新ihw权重
        n.ihw += n.lr * np.dot((hidden_errors * hidden_outputs* (1-hidden_outputs)),np.transpose(inputs))#点乘
    
        t+=1
           
    #输出结果
    print(net_value)
    final_scores.append(net_value)
    out_matrix.append(n.how) 
    in_matrix.append(n.ihw)

print(accuracies)    

#取分最高的索引，并输出
suoying=final_scores.index(max(final_scores))

with open('in_matrix.csv', 'w', newline='') as csvfile:
    writer  = csv.writer(csvfile)
    for row in in_matrix[suoying]:
        writer.writerow(row)

with open('out_matrix.csv', 'w', newline='') as csvfile:
    writer  = csv.writer(csvfile)
    for row in out_matrix[suoying]:
        writer.writerow(row)