import numpy #数组功能
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
        self.ihw=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.how=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #设定激活函数:使用sigmoid函数，一个常用的非线性激活函数，接受任何数值，输出0到1之间的某个值，但不包含0和1
        self.activation_function=lambda x:scipy.special.expit(x)
     
        pass

    def query(self,inputs_list):
        #计算输出的过程 
        inputs=numpy.array(inputs_list,ndmin=2).T#传递列表，转换为二维数组，转置
        
        hidden_inputs=numpy.dot(self.ihw,inputs)#点乘
        hidden_outputs=self.activation_function(hidden_inputs)#使用激活函数

        final_inputs=numpy.dot(self.how,hidden_outputs)#点乘
        final_outputs=self.activation_function(final_inputs)#使用激活函数

        return final_outputs#如果不写return，会返回一个None对象

    def train(self,inputs_list,targets_list):
        #反馈调节权重的过程/反向传播误差——告知如何优化权重
        
        #完全相同的计算，因此在循环中要重写
        inputs=numpy.array(inputs_list,ndmin=2).T#传递列表，转换为二维数组，转置
        
        hidden_inputs=numpy.dot(self.ihw,inputs)#点乘
        hidden_outputs=self.activation_function(hidden_inputs)#使用激活函数

        final_inputs=numpy.dot(self.how,hidden_outputs)#点乘
        final_outputs=self.activation_function(final_inputs)#使用激活函数

        targets=numpy.array(targets_list,ndmin=2).T#传递列表，转换为二维数组
        
        output_errors=targets-final_outputs#计算误差

        #隐藏层误差
        hidden_errors=numpy.dot(self.how.T,output_errors)#点乘
        #反向传递，更新how权重
        self.how += self.lr * numpy.dot((output_errors * final_outputs* (1-final_outputs)),numpy.transpose(hidden_outputs))#点乘
        #反向传递，更新ihw权重
        self.ihw += self.lr * numpy.dot((hidden_errors * hidden_outputs* (1-hidden_outputs)),numpy.transpose(inputs))#点乘
        pass



n=Neuralnetworks(1750,200,2,0.02)

##输入为7*250个交易日，准备训练数据
data_file=open('data_restructured.csv','r')#每个新行表示一个新的数据库行
data_list=data_file.readlines()#转换为列表
data_list = [x.strip() for x in data_list if x.strip() != '']#不等于空就移除字符串头尾指定的字符（默认为空格或换行符）
data_list = [x.strip('"') for x in data_list if x.strip() != '']#不等于空就移除字符串头尾指定的字符（"）
data_file.close()

#准备target数据
with open('train.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    column1 = [row[2]for row in reader]
closing_prices=column1[1:]#去除标签行，保持一致

#计分板：对应的，净值额、两个矩阵
final_scores=[]
in_matrix=[]
out_matrix=[]

for e in range(100):#epochs

    scoreboard=[]
    accuracies=[]
    actions=[]

    t=251#第一个预测值的位置
    cash=10000
    equity=0

    #训练并对比
    for record in data_list:
        all_values=record.split(',')    
        inputs=numpy.asfarray(all_values) 
        inputs=numpy.array(inputs,ndmin=2).T#传递列表，转换为二维数组，转置

        ###计算过程
        hidden_inputs=numpy.dot(n.ihw,inputs)#点乘
        hidden_outputs=n.activation_function(hidden_inputs)#使用激活函数

        final_inputs=numpy.dot(n.how,hidden_outputs)#点乘
        final_outputs=n.activation_function(final_inputs)#使用激活函数


    
        ###对比
        networks_label=numpy.argmax(final_outputs)#取出最大值对应的索引值，0或1
        actions.append(networks_label)
        
        closing_0=numpy.array(closing_prices[t],ndmin=2)
        closing_1=numpy.array(closing_prices[t+1],ndmin=2)    

        if closing_1>=closing_0:#涨了，应当满仓
            correct_label=1
        else:#跌了，应该空仓
            correct_label=0

        if networks_label==correct_label:
            scoreboard.append(1)
        else:
            scoreboard.append(0)

        #记录结果2：预测正确率
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

        #每轮有一个新的target值
        targets=numpy.zeros(2)+0.01 

        if closing_1>=closing_0:#涨了，应当满仓
            targets[1]=0.99
        else:#跌了，应该空仓
            targets[0]=0.99

        #训练过程
        targets=numpy.array(targets,ndmin=2).T#传递列表，转换为二维数组
        output_errors=targets-final_outputs#计算误差

        #隐藏层误差
        hidden_errors=numpy.dot(n.how.T,output_errors)#点乘
        #反向传递，更新how权重
        n.how += n.lr * numpy.dot((output_errors * final_outputs* (1-final_outputs)),numpy.transpose(hidden_outputs))#点乘
        #反向传递，更新ihw权重
        n.ihw += n.lr * numpy.dot((hidden_errors * hidden_outputs* (1-hidden_outputs)),numpy.transpose(inputs))#点乘
    
        t+=1
    
    #输出结果
    print(net_value)   
    final_scores.append(net_value)
    out_matrix.append(n.how) 
    in_matrix.append(n.ihw)

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