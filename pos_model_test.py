#encoding=utf8
import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import data_read as readData
import bi_lstm_model as modelDef
from tensorflow.contrib import crf


class BiLSTMTest(object):
    def __init__(self, data=None, model_path='.\ckpt\pos',test_file='.\\test\\seg_result.txt', test_result='.\\test\pos_result.txt'):
        self.data = data
        self.model_path = model_path
        self.test_file = test_file
        self.test_result = test_result
        self.sess = tf.Session()
        self.isload = self.loadModel(self.sess)
        self.seq_len = 200 # 神经网络输入大小

    # 读取测试文件并且写入测试文件
    def testfile(self):     
        pos = pd.read_csv('./pos.csv') # 读取词性标签以及对应id
        id2tag = {ii: label for ii, label in enumerate(list(pos['POS']),1)}
        id2tag[0]=''
        
        isFile = os.path.isfile(self.test_file)
        if isFile:
            with open(self.test_result, "w", encoding='utf-8') as out:  # 读写文件默认都是UTF-8编码的
                with open(self.test_file, "r", encoding='utf-8') as fp:
                    for line in fp.readlines(): # line为段
                        
                        words_ints = []
                        word_ints = []  
                        line = line.strip()
                        line = line.replace('/ ','').replace('/Date ','').replace('/','')
                        line = line.strip()
                        word_ = line.split(' ')
                        for word in word_:
                            if word in self.data.word2id: 
                                word_ints.append(self.data.word2id[word]) # 文字转id
                            else:
                                word_ints.append('126') # 未登录词处理
                        words_ints.append(word_ints)
                        
                        features_ = np.zeros((len(words_ints), self.seq_len), dtype=int)
                        for i, row in enumerate(words_ints):
                            features_[i, :len(row)] = np.array(row)[:self.seq_len]
                          
                        X_batch = features_ 
                        
                        fetches = [self.model.scores, self.model.length, self.model.transition_params]
                        feed_dict = {self.model.X_inputs: X_batch, self.model.lr: 1.0, self.model.batch_size: 1,
                                     self.model.keep_prob: 1.0}
                        test_score, test_length, transition_params = self.sess.run(fetches, feed_dict)
                        tags, _ = crf.viterbi_decode(test_score[0][:test_length[0]], transition_params)

                        tags = [id2tag[i] for i in tags]
                        
                        result = ''
                        for i in range(len(word_)):
                            result = result + word_[i] + '/' +tags[i] + ' '
                        print(result)
                        out.write("%s\n" % (result))

    # 加载还原模型
    def loadModel(self, sess=None):
        isload = False

        self.model = modelDef.BiLSTMModel(vocab_size=self.data.word2id.__len__(), class_num=self.data.tag2id.__len__())
        ckpt = tf.train.latest_checkpoint(self.model_path)
        print(ckpt)
        saver = tf.train.Saver()
        if ckpt:
            saver.restore(sess, ckpt)
            isload = True
        return isload


if __name__ == '__main__':

    data = readData.DataHandler(save_path = 'data/POS_training_data.pkl')
    data.loadData()
    print('加载字典(词性标注)完成')

    test = BiLSTMTest(data)
    if test.isload:
        test.testfile()
