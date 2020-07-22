"""
作者：乔家阳
学校：上海交通大学
描述：本模块包含了多个不同构思的朴素贝叶斯类，用于文本的分类。
"""
import numpy as np
import  string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
class Simple_Bayes_wordCount:
    def __init__(self,data,labels,laplas=1):
        """
        :param data:构建模型所使用的模型
        :param labels: 数据对应的标签
        """
        self.word_list = []  # 单词列表，不重复地记录训练数据种出现的单词,并使用下标代表相应的单词
        self.c_list = []  # 类别列表，记录类别的标号，列表中的值为labels当中的标识，序号作为程序使用的类别标识
        self.c_count = []  # 记录每一个类别的文本的数量
        self.c_w = np.mat([])  # 矩阵，行号对应各个类别，列号对应各个单词(不重复)
        stop_words = set(stopwords.words('english'))
        for i in range(len(data)):  # 对每一个文本进行遍历
            str_lower = str.lower((data.iloc[i]))  # 将第i个文本转换为小写
            str_pure = str_lower.translate(str.maketrans('', '', string.punctuation))
            # 去除没有具体词义的标点符号,我认为感叹号可以留下
            word_tokens = word_tokenize(str_pure)
            # word_tokens是一个列表对象
            list_ = [w for w in word_tokens if not w in stop_words]
            # 完成停用词的处理，

            w_list = np.zeros(len(self.word_list))
            # 建立该文本的单词列表，记录该文本中出现了哪些单词及其出现次数。1*|w|的数组，
            # 第i个单元的值表示i所对应的单词的出现的次数

            if labels.iloc[i] in self.c_list:  # 判断当前的文本i的类别是否在c_list当中
                c_idex = self.c_list.index(labels.iloc[i])  # 将文本当中的标号兑换成c_list
                # 下标作为的类别标号。

                self.c_count[c_idex] += 1
                # 对应类别的文本的数目增加1，用于计算P(c)，即某一类文本出现的概率。应当是基于
                # 文本的数量进行统计，绝非对应类别单词的数量！！！
            else:  # 倘若列表当中还没有记录这一类别
                c_idex = len(self.c_list)  # c_idex是新类别的标号，c_list的数组下标
                self.c_list.append(labels.iloc[i])  # 将新的标号添加到数组当中                self.c_count.append(1)
                self.c_w = np.concatenate((self.c_w, np.mat(np.zeros(self.c_w.shape[1]))), axis=0)
                # 由于检测到新的文本类别出现，我们在c_w矩阵当中需要增加一个新的行
            for word in list_:  # 对该文本当中的单词进行遍历
                if word not in self.word_list:  # 检查单词是否不在在单词列表当中
                    self.word_list.append(word)  # 将新的单词添加到单词列表当中
                    w_list = np.append(w_list, 1)  # 该文本对应得单词记录表，也新添加一个位置
                else:  # 若单词在该文本当中
                    word_index = self.word_list.index(word)  # 获取单词的序号
                    w_list[word_index] += 1  # 将相应的单词计数加1，应该置为1就好了
                    # 对于单个文本，不应该对单词进行重复计数。
            if len(w_list) > self.c_w.shape[1]:  # 判断是否在该文本中出现了新的单词
                self.c_w = np.concatenate((self.c_w, np.mat(np.zeros((self.c_w.shape[0], len(w_list) - self.c_w.shape[1])))), axis=-1)
                # 对c_w矩阵，补加上新单词的位置
            self.c_w[c_idex, :] = self.c_w[c_idex, :] + w_list
            # 将该i号文本的记录，添加到c_w矩阵的对应行当中
            print(i)
        self.c_w = np.delete(self.c_w, self.c_w.shape[0] - 1, axis=0)
        # 消除最后一行，这是我程序的缺陷之处，c_w有原始的一行，最后去除即可
        c_num = np.sum(self.c_w,axis=-1)
        #每个类别的单词总量
        p_c_w = np.divide(self.c_w + laplas, np.array(c_num).reshape(-1, 1) + len(self.word_list)*laplas)
        # 这个操作叫作拉普拉斯正则化，因为某些单词在一些类别当中根本没出现过，但是，按照
        # 朴素贝叶斯的计算方式，有p(w1|c)*p(w2|c)...p(wn|c),若某一个单词w没在c类别当
        # 中出现过的话，对应p(w|c)就是0.这是很不合理的事情，我们在文本分类这样的任务当中，
        # 不能因为有一些在c当中没有统计到的单词，就完全扼杀了该文本属于该类别的可能。拉普拉斯
        # 会使得每一个单词对应的p(w|c)都大于0，同时还保证了所有的p(w|c)相加，概率和仍为1.还
        # 有，我们会把概率相乘换成对数相加的形式，因为当一个文本当中的单词过多，多个p(w|c)相乘
        # 会得到一个非常小的数，计算机的浮点数据类型都无法准确表示，最终得到0.这显然给我们的
        # 最后的比较造成很大的困扰，所以，拉普拉斯正则化十分重要。

        p_c = np.divide(np.array(self.c_count), len(labels))
        # 每个类别的文本的数量，处以总的文本的数量，获得p(c)即某一类别的文本出现的概率

        w_num = np.sum(self.c_w, axis=0)
        # 我们对c_w矩阵的每一行进行相加的操作，得到每一个类别统计到的单词数目。统计的方式很
        # 考究，单个文本当中的单词，不应该进行重复统计，不同文本的相同单词要重复统计。我们既
        # 防范了个别文本过度使用某词语所造成的破坏，也兼顾到了单词出现频率的计量。

        p_w = np.divide(w_num, np.sum(w_num))
        # 每个单词出现的次数除以总的单词出现次数，这里对单词数量的统计也很考究。我还是建议，单
        # 个文本当中的单词，不要重复计量，或者压缩重复计量的比例，除了第一次计量之后，第二次就
        # 累加0.1等操作，第三，第四......，我认为，这也是一种不错的方法,这些方法，我都会尝试
        # 一下。我们这样操作，将过度使用某单词的影响进行了削减，也实际地考虑到了，过度使用和只
        # 使用一次也是有区别的，就比如你在某个文本当中，写了一个like，说明你喜欢，是正面的。而
        # 你写了5个like，实际上体现了你特别喜欢，喜欢的程度更深了，对于分类，应该有影响。也有一
        # 些am is are, 我们最好不去统计这些单词，我们不重复计量，也是消除这些没有情感的单词的
        # 影响，我们可以做一下统计，看这几种方式，是否让这些单词在不同类别之间分布得更加均匀。
        # 了。P_w不影响最终的分类结果，因为对于w进行分类的时候，p(w)都是相同的。影响的是p_c
        self.p_c_w_log = np.log(p_c_w)
        self.p_c_log = np.log(p_c)
        self.p_w_log = np.log(p_w)
        # 我们把概率值，进行log运算，它不影响我们最终的比较结果，而且计算机能够准确地显示出log
        # 函数的运算结果

    def predict(self,data):
        # 所谓的朴素贝叶斯文本分类器，不过是我们利用训练数据得到的一些变量而已。在分类的
        # 时候，也只是使用这些变量，计算p(c1|w),p(c2|w),p(c3|w)...的值，值最大的对应
        # 的ci,就是我们要分类的类别。
        pwlog = self.p_w_log
        word_list = self.word_list
        pcwlog = self.p_c_w_log
        pclog = self.p_c_log
        c_list = self.c_list
        sumplogw = np.sum(pwlog)
        # 这个值就是log(p(w)),它对于分类结果，没有任何影响，但为了体现模型的完整性，我还是
        # 把这一部分的内容给添加上了。

        answers = []
        # answers用于存放每个样本的分类结果
        for i in range(len(data)):  # 用于遍历所有的文本,i是文本的标号
            str_lower = str.lower((data.iloc[i]))
            # 将该文本处理为小写的,为什么要小写呢？因为，大写和小写，在情感的分类上是同等地位
            # 我们没有必要再多花一份存储的开销，而且可能导致分类结果的波动，因为大写的单词，在
            # 不同的类别的占比可能差异很大，所以把大写改成小写，好处真的很多。不仅仅是节省存储
            # 空间，也让我们在对结果分类的时候，分类的结果更加合理，

            str_pure = str_lower.translate(str.maketrans('', '', string.punctuation))
            # 将文本的标点去除掉
            list_ = str_pure.split()
            # 以空格来将文本分割成数组
            word_vector = np.zeros(len(word_list))
            # 提取该文本的词向量，这里也很考究，是记录是否出现，还是记录出现的次数？还是以
            # 0到1的权重来计入词向量。这里，我们也应该采用不同的方法来尝试一下。
            for word in list_:  # 对该文本对应的数组进行遍历
                if word in word_list:  # 判断文本当中的单词，是否在单词列表当中，只处理模型认识的单词
                    word_index = word_list.index(word)
                    # 获取单词在单词表当中的位置
                    word_vector[word_index] += 1
                    # 将相应的词向量的位置的值置为1，而不是加1.
            word_vector = np.reshape(word_vector, (-1, 1))
            # 将词向量的尺寸进行修改，将行向量修改为列向量，方便矩阵的乘法运算
            pclog = np.reshape(pclog, (-1, 1))
            # 将pclog从行向量修改为列向量
            ans0 = (pcwlog * word_vector)
            # c*w * w*1 = c*1，我们这样就完成了每个类别的的log(p(w|c))的运算
            ans1 = ans0 + pclog
            # 完成了log(p(w|c)*p(c))的计算
            ans = ans1 - sumplogw
            # 我还多此一举地把log(p(w))给捎带上了，它对于结果而言，可有可无，但是，我
            # 不是搞工程，我是要分析这项简单技术的细节。
            category_belongs = c_list[np.argmax(ans)]
            # c*1的列向量，值最大的序号,对应于c_list的相应序号位置的值就是分类的结果
            answers.append(category_belongs)
            # 将分类的结果添加到answers当中进行记录
            print(i)
            # 给程序设计师，显示当前的文本位置。
        return answers
class Simple_Bayes_paragraghCount:
    def __init__(self,data,labels,laplas=1):
        """
        :param data:构建模型所使用的模型
        :param labels: 数据对应的标签
        """
        self.word_list = []  # 单词列表，不重复地记录训练数据种出现的单词
        self.c_list = []  # 类别列表，记录类别的标号
        self.c_count = []  # 记录每一个类别的文本的数量
        self.c_w = np.mat([])  # 矩阵，行号对应各个类别，列号对应各个单词(不重复)
        stop_words = set(stopwords.words('english'))
        print(self.c_w.shape)
        for i in range(len(data)):  # 对每一个文本进行遍历
            str_lower = str.lower((data.iloc[i]))  # 将第i个文本转换为小写
            str_pure = str_lower.translate(str.maketrans('', '', string.punctuation))
            # 去除没有具体词义的标点符号,我认为感叹号可以留下
            word_tokens = word_tokenize(str_pure)
            # word_tokens是一个列表对象
            list_ = [w for w in word_tokens if not w in stop_words]
            # 完成停用词的处理，

            w_list = np.zeros(len(self.word_list))
            # 建立该文本的单词列表，记录该文本中出现了哪些单词及其出现次数。1*|w|的数组，
            # 第i个单元的值表示i所对应的单词的出现的次数

            if labels.iloc[i] in self.c_list:  # 判断当前的文本i的类别是否在c_list当中
                c_idex = self.c_list.index(labels.iloc[i])  # 将文本当中的标号兑换成c_list
                # 下标作为的类别标号。

                self.c_count[c_idex] += 1
                # 对应类别的文本的数目增加1，用于计算P(c)，即某一类文本出现的概率。应当是基于
                # 文本的数量进行统计，绝非对应类别单词的数量！！！
            else:  # 倘若列表当中还没有记录这一类别
                c_idex = len(self.c_list)  # c_idex是新类别的标号，c_list的数组下标
                self.c_list.append(c_idex)  # 将新的标号添加到数组当中
                self.c_count.append(1)
                self.c_w = np.concatenate((self.c_w, np.mat(np.zeros(self.c_w.shape[1]))), axis=0)
                # 由于检测到新的文本类别出现，我们在c_w矩阵当中需要增加一个新的行
            for word in list_:  # 对该文本当中的单词进行遍历
                if word not in self.word_list:  # 检查单词是否不在在单词列表当中
                    self.word_list.append(word)  # 将新的单词添加到单词列表当中
                    w_list = np.append(w_list, 1)  # 该文本对应得单词记录表，也新添加一个位置
                else:  # 若单词在该文本当中
                    word_index = self.word_list.index(word)  # 获取单词的序号
                    w_list[word_index] = 1  # 将相应的单词计数加1，应该置为1就好了
                    # 对于单个文本，不应该对单词进行重复计数。
            if len(w_list) > self.c_w.shape[1]:  # 判断是否在该文本中出现了新的单词
                self.c_w = np.concatenate((self.c_w, np.mat(np.zeros((self.c_w.shape[0], len(w_list) - self.c_w.shape[1])))), axis=-1)
                # 对c_w矩阵，补加上新单词的位置
            self.c_w[c_idex, :] = self.c_w[c_idex, :] + w_list
            # 将该i号文本的记录，添加到c_w矩阵的对应行当中
            print(i)
        self.c_w = np.delete(self.c_w, self.c_w.shape[0] - 1, axis=0)
        # 消除最后一行，这是我程序的缺陷之处，c_w有原始的一行，最后去除即可
        c_num = np.sum(self.c_w,axis=-1)
        p_c_w = np.divide(self.c_w + laplas, np.array(c_num).reshape(-1, 1) + len(self.word_list)*laplas)
        # 这个操作叫作拉普拉斯正则化，因为某些单词在一些类别当中根本没出现过，但是，按照
        # 朴素贝叶斯的计算方式，有p(w1|c)*p(w2|c)...p(wn|c),若某一个单词w没在c类别当
        # 中出现过的话，对应p(w|c)就是0.这是很不合理的事情，我们在文本分类这样的任务当中，
        # 不能因为有一些在c当中没有统计到的单词，就完全扼杀了该文本属于该类别的可能。拉普拉斯
        # 会使得每一个单词对应的p(w|c)都大于0，同时还保证了所有的p(w|c)相加，概率和仍为1.还
        # 有，我们会把概率相乘换成对数相加的形式，因为当一个文本当中的单词过多，多个p(w|c)相乘
        # 会得到一个非常小的数，计算机的浮点数据类型都无法准确表示，最终得到0.这显然给我们的
        # 最后的比较造成很大的困扰，所以，拉普拉斯正则化十分重要。

        p_c = np.divide(np.array(self.c_count), len(labels))
        # 每个类别的文本的数量，处以总的文本的数量，获得p(c)即某一类别的文本出现的概率

        w_num = np.sum(self.c_w, axis=0)
        # 我们对c_w矩阵的每一行进行相加的操作，得到每一个类别统计到的单词数目。统计的方式很
        # 考究，单个文本当中的单词，不应该进行重复统计，不同文本的相同单词要重复统计。我们既
        # 防范了个别文本过度使用某词语所造成的破坏，也兼顾到了单词出现频率的计量。

        p_w = np.divide(w_num, np.sum(w_num))
        # 每个单词出现的次数除以总的单词出现次数，这里对单词数量的统计也很考究。我还是建议，单
        # 个文本当中的单词，不要重复计量，或者压缩重复计量的比例，除了第一次计量之后，第二次就
        # 累加0.1等操作，第三，第四......，我认为，这也是一种不错的方法,这些方法，我都会尝试
        # 一下。我们这样操作，将过度使用某单词的影响进行了削减，也实际地考虑到了，过度使用和只
        # 使用一次也是有区别的，就比如你在某个文本当中，写了一个like，说明你喜欢，是正面的。而
        # 你写了5个like，实际上体现了你特别喜欢，喜欢的程度更深了，对于分类，应该有影响。也有一
        # 些am is are, 我们最好不去统计这些单词，我们不重复计量，也是消除这些没有情感的单词的
        # 影响，我们可以做一下统计，看这几种方式，是否让这些单词在不同类别之间分布得更加均匀。
        # 了。P_w不影响最终的分类结果，因为对于w进行分类的时候，p(w)都是相同的。影响的是p_c
        self.p_c_w_log = np.log(p_c_w)
        self.p_c_log = np.log(p_c)
        self.p_w_log = np.log(p_w)
        # 我们把概率值，进行log运算，它不影响我们最终的比较结果，而且计算机能够准确地显示出log
        # 函数的运算结果

    def predict(self,data):
        # 所谓的朴素贝叶斯文本分类器，不过是我们利用训练数据得到的一些变量而已。在分类的
        # 时候，也只是使用这些变量，计算p(c1|w),p(c2|w),p(c3|w)...的值，值最大的对应
        # 的ci,就是我们要分类的类别。
        pwlog = self.p_w_log
        word_list = self.word_list
        pcwlog = self.p_c_w_log
        pclog = self.p_c_log
        c_list = self.c_list
        sumplogw = np.sum(pwlog)
        # 这个值就是log(p(w)),它对于分类结果，没有任何影响，但为了体现模型的完整性，我还是
        # 把这一部分的内容给添加上了。

        answers = []
        # answers用于存放每个样本的分类结果
        for i in range(len(data)):  # 用于遍历所有的文本,i是文本的标号
            str_lower = str.lower((data.iloc[i]))
            # 将该文本处理为小写的,为什么要小写呢？因为，大写和小写，在情感的分类上是同等地位
            # 我们没有必要再多花一份存储的开销，而且可能导致分类结果的波动，因为大写的单词，在
            # 不同的类别的占比可能差异很大，所以把大写改成小写，好处真的很多。不仅仅是节省存储
            # 空间，也让我们在对结果分类的时候，分类的结果更加合理，

            str_pure = str_lower.translate(str.maketrans('', '', string.punctuation))
            # 将文本的标点去除掉
            list_ = str_pure.split()
            # 以空格来将文本分割成数组
            word_vector = np.zeros(len(word_list))
            # 提取该文本的词向量，这里也很考究，是记录是否出现，还是记录出现的次数？还是以
            # 0到1的权重来计入词向量。这里，我们也应该采用不同的方法来尝试一下。
            for word in list_:  # 对该文本对应的数组进行遍历
                if word in word_list:  # 判断文本当中的单词，是否在单词列表当中
                    word_index = word_list.index(word)
                    # 获取单词在单词表当中的位置
                    word_vector[word_index] = 1
                    # 将相应的词向量的位置的值置为1，而不是加1.
            word_vector = np.reshape(word_vector, (-1, 1))
            # 将词向量的尺寸进行修改，将行向量修改为列向量，方便矩阵的乘法运算
            pclog = np.reshape(pclog, (-1, 1))
            # 将pclog从行向量修改为列向量
            ans0 = (pcwlog * word_vector)
            # c*w * w*1 = c*1，我们这样就完成了每个类别的的log(p(w|c))的运算
            ans1 = ans0 + pclog
            # 完成了log(p(w|c)*p(c))的计算
            ans = ans1 - sumplogw
            # 我还多此一举地把log(p(w))给捎带上了，它对于结果而言，可有可无，但是，我
            # 不是搞工程，我是要分析这项简单技术的细节。
            category_belongs = c_list[np.argmax(ans)]
            # c*1的列向量，值最大的序号,对应于c_list的相应序号位置的值就是分类的结果
            answers.append(category_belongs)
            # 将分类的结果添加到answers当中进行记录
            print(i)
            # 给程序设计师，显示当前的文本位置。
        return answers
class Simple_Bayes_wordweightCount:
    def generate_wordweight(wordc_list,word_index):
        if wordc_list[word_index]>1:
           return 1
        else:
            return 0.1
    def __init__(self,data,labels,laplas=1,generate_weight=generate_wordweight):
        """
        :param data:构建模型所使用的模型
        :param labels: 数据对应的标签
        :param generate_weight:产生单词计数时的加权值,使用者需要自己
        设计相应的函数，设计的规则是，第一个参数为记录当前各个单词出现的
        次数的列表，第二个参数是单词序号，用于访问列表当中的单词,我也提供
        了一个很简单的计算权重的默认函数。
        """
        self.word_list = []  # 单词列表，不重复地记录训练数据种出现的单词
        self.c_list = []  # 类别列表，记录类别的标号
        self.c_count = []  # 记录每一个类别的文本的数量
        self.c_w = np.mat([])  # 矩阵，行号对应各个类别，列号对应各个单词(不重复)
        self.generate_weight =generate_weight
        stop_words = set(stopwords.words('english'))
        for i in range(len(data)):  # 对每一个文本进行遍历
            str_lower = str.lower((data.iloc[i]))  # 将第i个文本转换为小写
            str_pure = str_lower.translate(str.maketrans('', '', string.punctuation))
            # 去除没有具体词义的标点符号,我认为感叹号可以留下
            word_tokens = word_tokenize(str_pure)
            # word_tokens是一个列表对象
            list_ = [w for w in word_tokens if not w in stop_words]
            # 完成停用词的处理，

            w_list = np.zeros(len(self.word_list))
            # 建立该文本的单词列表，记录该文本中出现了哪些单词及其出现次数。1*|w|的数组，
            # 第i个单元的值表示i所对应的单词的权重结果
            wc_list = np.zeros(len(self.word_list))
            if labels.iloc[i] in self.c_list:  # 判断当前的文本i的类别是否在c_list当中
                c_idex = self.c_list.index(labels.iloc[i])  # 将文本当中的标号兑换成c_list
                # 下标作为的类别标号。

                self.c_count[c_idex] += 1
                # 对应类别的文本的数目增加1，用于计算P(c)，即某一类文本出现的概率。应当是基于
                # 文本的数量进行统计，绝非对应类别单词的数量！！！
            else:  # 倘若列表当中还没有记录这一类别
                c_idex = len(self.c_list)  # c_idex是新类别的标号，c_list的数组下标
                self.c_list.append(c_idex)  # 将新的标号添加到数组当中
                self.c_count.append(1)
                self.c_w = np.concatenate((self.c_w, np.mat(np.zeros(self.c_w.shape[1]))), axis=0)
                # 由于检测到新的文本类别出现，我们在c_w矩阵当中需要增加一个新的行
            for word in list_:  # 对该文本当中的单词进行遍历
                if word not in self.word_list:  # 检查单词是否不在在单词列表当中
                    self.word_list.append(word)  # 将新的单词添加到单词列表当中
                    w_list = np.append(w_list, 1)  # 该文本对应得单词记录表，也新添加一个位置
                    wc_list = np.append(wc_list,1) #用于计数文本当中，单词出现次数的列表
                else:  # 若单词在该文本当中
                    word_index = self.word_list.index(word)  # 获取单词的序号
                    wc_list[word_index] += 1
                    w_list[word_index] += generate_weight(wc_list,word_index)  # 将相应的单词计数加1，应该置为1就好了
                    # 对于单个文本，不应该对单词进行重复计数。
            if len(w_list) > self.c_w.shape[1]:  # 判断是否在该文本中出现了新的单词
                self.c_w = np.concatenate((self.c_w, np.mat(np.zeros((self.c_w.shape[0], len(w_list) - self.c_w.shape[1])))), axis=-1)
                # 对c_w矩阵，补加上新单词的位置
            self.c_w[c_idex, :] = self.c_w[c_idex, :] + w_list
            # 将该i号文本的记录，添加到c_w矩阵的对应行当中
            print(i)
        self.c_w = np.delete(self.c_w, self.c_w.shape[0] - 1, axis=0)
        # 消除最后一行，这是我程序的缺陷之处，c_w有原始的一行，最后去除即可

        c_num = np.sum(self.c_w,axis=-1)
        p_c_w = np.divide(self.c_w + laplas, np.array(c_num).reshape(-1, 1) + len(self.word_list)*laplas)
        # 这个操作叫作拉普拉斯正则化，因为某些单词在一些类别当中根本没出现过，但是，按照
        # 朴素贝叶斯的计算方式，有p(w1|c)*p(w2|c)...p(wn|c),若某一个单词w没在c类别当
        # 中出现过的话，对应p(w|c)就是0.这是很不合理的事情，我们在文本分类这样的任务当中，
        # 不能因为有一些在c当中没有统计到的单词，就完全扼杀了该文本属于该类别的可能。拉普拉斯
        # 会使得每一个单词对应的p(w|c)都大于0，同时还保证了所有的p(w|c)相加，概率和仍为1.还
        # 有，我们会把概率相乘换成对数相加的形式，因为当一个文本当中的单词过多，多个p(w|c)相乘
        # 会得到一个非常小的数，计算机的浮点数据类型都无法准确表示，最终得到0.这显然给我们的
        # 最后的比较造成很大的困扰，所以，拉普拉斯正则化十分重要。

        p_c = np.divide(np.array(self.c_count), len(labels))
        # 每个类别的文本的数量，处以总的文本的数量，获得p(c)即某一类别的文本出现的概率

        w_num = np.sum(self.c_w, axis=0)
        # 我们对c_w矩阵的每一行进行相加的操作，得到每一个类别统计到的单词数目。统计的方式很
        # 考究，单个文本当中的单词，不应该进行重复统计，不同文本的相同单词要重复统计。我们既
        # 防范了个别文本过度使用某词语所造成的破坏，也兼顾到了单词出现频率的计量。

        p_w = np.divide(w_num, np.sum(w_num))
        # 每个单词出现的次数除以总的单词出现次数，这里对单词数量的统计也很考究。我还是建议，单
        # 个文本当中的单词，不要重复计量，或者压缩重复计量的比例，除了第一次计量之后，第二次就
        # 累加0.1等操作，第三，第四......，我认为，这也是一种不错的方法,这些方法，我都会尝试
        # 一下。我们这样操作，将过度使用某单词的影响进行了削减，也实际地考虑到了，过度使用和只
        # 使用一次也是有区别的，就比如你在某个文本当中，写了一个like，说明你喜欢，是正面的。而
        # 你写了5个like，实际上体现了你特别喜欢，喜欢的程度更深了，对于分类，应该有影响。也有一
        # 些am is are, 我们最好不去统计这些单词，我们不重复计量，也是消除这些没有情感的单词的
        # 影响，我们可以做一下统计，看这几种方式，是否让这些单词在不同类别之间分布得更加均匀。
        # 了。P_w不影响最终的分类结果，因为对于w进行分类的时候，p(w)都是相同的。影响的是p_c
        self.p_c_w_log = np.log(p_c_w)
        self.p_c_log = np.log(p_c)
        self.p_w_log = np.log(p_w)
        # 我们把概率值，进行log运算，它不影响我们最终的比较结果，而且计算机能够准确地显示出log
        # 函数的运算结果

    def predict(self,data):
        # 所谓的朴素贝叶斯文本分类器，不过是我们利用训练数据得到的一些变量而已。在分类的
        # 时候，也只是使用这些变量，计算p(c1|w),p(c2|w),p(c3|w)...的值，值最大的对应
        # 的ci,就是我们要分类的类别。
        pwlog = self.p_w_log
        word_list = self.word_list
        pcwlog = self.p_c_w_log
        pclog = self.p_c_log
        c_list = self.c_list
        sumplogw = np.sum(pwlog)
        # 这个值就是log(p(w)),它对于分类结果，没有任何影响，但为了体现模型的完整性，我还是
        # 把这一部分的内容给添加上了。

        answers = []
        # answers用于存放每个样本的分类结果
        for i in range(len(data)):  # 用于遍历所有的文本,i是文本的标号
            str_lower = str.lower((data.iloc[i]))
            # 将该文本处理为小写的,为什么要小写呢？因为，大写和小写，在情感的分类上是同等地位
            # 我们没有必要再多花一份存储的开销，而且可能导致分类结果的波动，因为大写的单词，在
            # 不同的类别的占比可能差异很大，所以把大写改成小写，好处真的很多。不仅仅是节省存储
            # 空间，也让我们在对结果分类的时候，分类的结果更加合理，

            str_pure = str_lower.translate(str.maketrans('', '', string.punctuation))
            # 将文本的标点去除掉
            list_ = str_pure.split()
            # 以空格来将文本分割成数组
            word_vector = np.zeros(len(word_list))
            word_cvector = np.zeros(len(word_list))
            # 提取该文本的词向量，这里也很考究，是记录是否出现，还是记录出现的次数？还是以
            # 0到1的权重来计入词向量。这里，我们也应该采用不同的方法来尝试一下。
            for word in list_:  # 对该文本对应的数组进行遍历
                if word in word_list:  # 判断文本当中的单词，是否在单词列表当中
                    word_index = word_list.index(word)
                    # 获取单词在单词表当中的位置
                    word_cvector[word_index] += 1
                    word_vector[word_index] +=self.generate_weight(word_cvector,word_index)
                    # 将相应的词向量的位置的值置为1，而不是加1.
            word_vector = np.reshape(word_vector, (-1, 1))
            # 将词向量的尺寸进行修改，将行向量修改为列向量，方便矩阵的乘法运算
            pclog = np.reshape(pclog, (-1, 1))
            # 将pclog从行向量修改为列向量
            ans0 = (pcwlog * word_vector)
            # c*w * w*1 = c*1，我们这样就完成了每个类别的的log(p(w|c))的运算
            ans1 = ans0 + pclog
            # 完成了log(p(w|c)*p(c))的计算
            ans = ans1 - sumplogw
            # 我还多此一举地把log(p(w))给捎带上了，它对于结果而言，可有可无，但是，我
            # 不是搞工程，我是要分析这项简单技术的细节。
            category_belongs = c_list[np.argmax(ans)]
            # c*1的列向量，值最大的序号,对应于c_list的相应序号位置的值就是分类的结果
            answers.append(category_belongs)
            # 将分类的结果添加到answers当中进行记录
            print(i)
            # 给程序设计师，显示当前的文本位置。
        return answers