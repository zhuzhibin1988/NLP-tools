NLP-tools
==
本项目旨在通过Tensorflow基于BiLSTM+CRF实现中文分词、词性标注、命名实体识别（NER）。

模型架构：输入层-->嵌入层-->双向长短记忆网络CELL-->输出层。

模型从网络中自我学习嵌入矩阵（非预训练）提高效率。

欢迎各位大佬吐槽。

模型训练
--
corpus文件下放入语料集（该语料集未上传至github，只有部分样例，可通过互联网找到。若找不到可email me），语料格式：人民网/nz 1月4日/t 讯/ng 据/p [法国/nsf 国际/n。

执行python train.py 开始训练（分词与Pos训练只需调整输出数据类型，seg_training_data.pkl或是pos_training_data.pkl既可）。

生成word2id字典存入data/seg_trining_data.pkl。(data/pos_trining_data.pkl。)

训练生成checkpoint存入ckpt/seg/。(ckpt/pos/)


模型超参数
--
* 分词标记方式：4tags 

* 嵌入层向量长度：100

* BiLstm层数：2

* 隐藏层节点数：256

* 最大迭代次数：9 (POS训练只需到6次迭代acc达到99%)

* Batch宽度：128

* 初始学习率：1e-3（采用动态形式，随训练进行而减小步长）
    
模型测试
--
将待分词项写入test/test.txt文件中，执行python seg_model_test.py，生成结果存入test/seg_result.txt。
执行 python pos_model_test.py，生成结果存入test/pos_result.txt。

现状
--
目前模型尚处于初步测试成功，分词部分完成，正确率97%。
                            词性标注训练完成，正确率99%，代码尚需整理。

后期陆续整理出NER以及Parse功能。 

参考
--

本项目模型BiLSTM+CRF参考论文：http://www.aclweb.org/anthology/N16-1030
