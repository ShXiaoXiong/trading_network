import numpy #数组功能
import scipy.special #激活函数
import matplotlib.pyplot as plt #可视化

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


#按250个交易日往前推

#构建神经网络实例
#输入为7*250个交易日
n=Neuralnetworks(1750,100,2,0.3)

#训练并对比
for record in range(484):#训练次数=735-251+1+1=485次，最后一组用于计算最后一次的target
    
    #每次都重新组成收入
    inputs_list=[]#全部转换成列表。
    for record in data_list[t-250:t-1]:#往前推250个交易日数据，一共250行，250*7=1750个数据
        b = record.split(',')#指定分隔符‘，’，对字符串进行切片，返回一个列表
        closing_prices += b[2]#列表中的第【2】，是closing price
        inputs_list +=b#列表拼接

    inputs=numpy.array(inputs_list,ndmin=2).T#传递列表，转换为二维数组，转置


    #计算过程
    hidden_inputs=numpy.dot(n.ihw,inputs)#点乘
    hidden_outputs=n.activation_function(hidden_inputs)#使用激活函数

    final_inputs=numpy.dot(n.how,hidden_outputs)#点乘
    final_outputs=n.activation_function(final_inputs)#使用激活函数

    
    #对比
    networks_label=numpy.argmax(final_outputs)#取出最大值对应的索引值

    #输出结果，计算净值
    
    closing_0=numpy.array(data_list[t-1],ndmin=2)[3]
    closing_1=numpy.array(data_list[t],ndmin=2)    
    
    net_value=10000/closing_0*closing_1


    if networks_label=correct_label:#涨
        correct_label=1
    else:#跌
        correct_label=0
    
   
    #每轮有一个新的targets值
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
    pass


#最终结果可视化        
import matplotlib.pyplot as plt

y=accuracies
x=range(len(accuracies))
plt.plot(x,y,label='Frist line',linewidth=3,color='r',marker='o', markerfacecolor='blue',markersize=12) 
plt.show()
