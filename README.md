# Naive_Bayes_Text_Classification
Naive Bayes model used to make classifications of texts
Qiyang_NLP
        Simple_Bayes.py(模块)
                 Simple_Bayes_wordCount(类)
                            _init_(data,labels,test_data,test_labels,laplas=1)#data:训练数据，labels:训练数据标签，test_data：测试数据，test_labels:测试标签，laplas:拉普拉斯正则化参数
                            predict(data)#给定数据data，预测出数据对应的标签label
                 Simple_Bayes_paragraghCount(类)
                            _init_(data,labels,test_data,test_labels,laplas=1)#data:训练数据，labels:训练数据标签，test_data：测试数据，test_labels:测试标签，laplas:拉普拉斯正则化参数
                            predict(data)#给定数据data，预测出数据对应的标签label
                 Simple_Bayes_wordweightCount(类)
                            _init_(data,labels,test_data,test_labels,laplas=1,,generate_weight=generate_wordweight)#data:训练数据，labels:训练数据标签，test_data：测试数据，test_labels:测试标签，laplas:拉普拉斯正则化参数
                            predict(data)#给定数据data，预测出数据对应的标签label

Qiyang_NLP_test#用于实验结果的获取
        Simple_Bayes.py(模块)
                 Simple_Bayes_wordCount(类)
                            _init_(data,labels,test_data,test_labels,laplas=1)#data:训练数据，labels:训练数据标签，test_data：测试数据，test_labels:测试标签，laplas:拉普拉斯正则化参数
                            predict(data)#给定数据data，预测出数据对应的标签label
                 Simple_Bayes_paragraghCount(类)
                            _init_(data,labels,test_data,test_labels,laplas=1)#data:训练数据，labels:训练数据标签，test_data：测试数据，test_labels:测试标签，laplas:拉普拉斯正则化参数
                            predict(data)#给定数据data，预测出数据对应的标签label
                 Simple_Bayes_wordweightCount(类)
                            _init_(data,labels,test_data,test_labels,laplas=1,,generate_weight=generate_wordweight)#data:训练数据，labels:训练数据标签，test_data：测试数据，test_labels:测试标签，laplas:拉普拉斯正则化参数
                            predict(data)#给定数据data，预测出数据对应的标签label
