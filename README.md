NLP-tools
==
本项目旨在通过Tensorflow基于BiLSTM+CRF实现字符级序列标注模型。

功能：
1、对未登录字（词）识别能力
2、Http接口
3、可快速实现分词、词性标注、NER、SRL等序列标注模型
欢迎各位大佬吐槽。

说明
--

环境配置：创建新的conda环境

     $ conda env create -f environment.yaml

语料处理
--

不同标注语料格式不同，需额外处理，在example/DataPreprocessing.ipynb中提供了人民日报2014预处理过程（该语料集未上传至github，只有部分样例于corpus，可通过互联网找到。若找不到可email me），语料格式：人民网/nz 1月4日/t 讯/ng 据/p [法国/nsf 国际/n。

生成word2id字典和训练数据于data/xx.pkl中。

模型训练
--

     $ python train.py 
     [-h] [--dict_path DICT_PATH] [--train_data TRAIN_DATA]
          [--ckpt_path CKPT_PATH] [--embed_size EMBED_SIZE]
          [--hidden_size HIDDEN_SIZE] [--batch_size BATCH_SIZE] 
          [--epoch EPOCH] [--lr LR]
          [--save_path SAVE_PATH]

训练生成checkpoint存入SAVE_PATH, CKPT_PATH用于模型做finetune。


模型默认超参数
--

* 嵌入层向量长度：256

* BiLstm层数：2

* 隐藏层节点数：512

* Batch宽度：128

* 初始学习率：1e-3 （不同任务需做finetune）
    
模型测试
--

模型测试示例位于Modeltest.ipynb中。

HTTP接口
--

一个简单的web server

     $ python app.py

执行python，默认本机测试代码：(linux和windows下格式不同)

     $ curl -i -H "Content-Type: application/json" -X POST -d '{"text":"\u5f20\u51cc\u745e\u3002"}' http://localhost:7777/cws

现状
--

在人民日报上的分词能达到正确率97%，词性标注能达到正确率96%。

通过对该模型在上亿条句子上的训练结果测试，将CWS、POS、NER标签做成end2end的融合标签，综合正确率能达到96%，且对未登录字（词）识别能力佳，拥有对语义的捕获能力。

（在Modeltest.ipynb中列举了一些例子）

参考
--

本项目模型BiLSTM+CRF参考论文：http://www.aclweb.org/anthology/N16-1030
