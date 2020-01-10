import csv

#加载训练数据并转为列表
data_file=open('train.csv','r')#每个新行表示一个新的数据库行，每个数据库行由一个或多个以逗号分隔的字段组成
data_list=data_file.readlines()#转换为列表
data_file.close()

t=251#第一个预测值的位置
closing_prices=[]
inputs_list=[]#全部转换成列表

for xx in range(734-251):#跺步次数
    new_input=[]
    for record in data_list[t-250:t-1]:#往前推250个交易日数据，一共250行，250*7=1750个数据
        
        b = record.split(',')#指定分隔符‘，’，对字符串进行切片，返回一个列表
        new_input+=b#列表拼接

    inputs_list.append(new_input)
    pass

with open('data_restructured.csv', 'w', newline='') as csvfile:
    writer  = csv.writer(csvfile)
    for row in inputs_list:
        writer.writerow(row)
