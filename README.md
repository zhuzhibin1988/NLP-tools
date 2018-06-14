NLP-tools
==
本项目旨在通过Tensorflow基于BiLSTM+CRF实现中文分词、词性标注、命名实体识别（NER）。

模型架构：输入层-->嵌入层-->双向长短记忆网络CELL-->输出层。

模型从网络中自我学习嵌入矩阵（非预训练）提高效率。

项目持续优化中，最终提供不同领域完善的训练方式并提供稳定的HTTP接口。

欢迎各位大佬吐槽。

说明
--
下期更新：
1、各种语料集预处理.py

环境配置：创建新的conda环境

     $ conda env create -f environment.yaml

模型训练
--
语料放到CORPUS_PATH下（该语料集未上传至github，只有部分样例，可通过互联网找到。若找不到可email me），语料格式：人民网/nz 1月4日/t 讯/ng 据/p [法国/nsf 国际/n。

     分词预处理：
     $ python data_read_seg_RMRB.py #人民日报
     词性标注预处理：
     $ python data_read_pos_RMRB.py
     实体命名预处理：
     $ python data_preprocess.py # 实体命名代码与分词基本类似，只是再4Tags的基础上加了实体标签。可在sentence.py中查看。


生成word2id字典存入DICT_PATH。

     $ python train.py 
     [-h] [--corpus_path CORPUS_PATH] [--dict_path DICT_PATH]
          [--ckpt_path CKPT_PATH] [--embed_size EMBED_SIZE]
          [--layer_num LAYER_NUM] [--hidden_size HIDDEN_SIZE]
          [--batch_size BATCH_SIZE] [--epoch EPOCH] [--lr LR]

训练生成checkpoint存入CKPT_PATH。


模型默认超参数
--

* 嵌入层向量长度：300

* BiLstm层数：2

* 隐藏层节点数：256

* 最大迭代次数：9 (POS训练只需到6次迭代acc达到99%)

* Batch宽度：128

* 初始学习率：1e-4（采用动态形式，随训练进行而减小步长）
    
模型测试
--

    $ python seg_model_test.py # 分词
    [-h] [--dict_path DICT_PATH] [--model_path MODEL_PATH] 
         [--test_file TEST_FILE] [--test_result TEST_RESULT]

    $ python pos_model_test.py # 词性标注 
    [-h] [--dict_path DICT_PATH] [--model_path MODEL_PATH] 
         [--test_file TEST_FILE] [--test_result TEST_RESULT]

注 DICT_PATH、MODEL_PATH 选择预处理数据文件和相应模型文件。TEST_FILE 待测试文件目录，TEST_RESULT 预测结果保持目录。

HTTP接口
--
一个简单的web server

     $ python web_server.py
     [-h] [--dict_path DICT_PATH] [--model_path MODEL_PATH]

选择预处理数据文件 和 模型文件。执行python，默认本机测试代码：

     $ curl -i -H "Content-Type: application/json" -X POST -d '{"text":"\u5f20\u51cc\u745e\u3002"}' http://localhost:5000/ner_text

现状
--
目前模型尚处于初步测试成功，分词部分完成，正确率97%。
                            
词性标注训练完成，正确率99%，代码尚需整理。

后期陆续整理出NER以及Parse功能。 

参考
--

本项目模型BiLSTM+CRF参考论文：http://www.aclweb.org/anthology/N16-1030
