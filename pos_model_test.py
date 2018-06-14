#encoding=utf8
import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import data_read as readData
import bi_lstm_model as modelDef
from tensorflow.contrib import crf

tf.app.flags.DEFINE_string('dict_path', 'data/POS_training_data.pkl', 'dict path')
tf.app.flags.DEFINE_string('model_path', 'ckpt/pos', 'model path')
tf.app.flags.DEFINE_string('test_file', 'test/seg_result', 'test file path')
tf.app.flags.DEFINE_string('test_result', 'test/pos_result.txt', 'test result path')

FLAGS = tf.app.flags.FLAGS

class BiLSTMTest(object):
    def __init__(self, data=None, model_path=FLAGS.model_path, 
                 test_file=FLAGS.test_file, test_result=FLAGS.test_result):
        self.data = data
        self.model_path = model_path
        self.test_file = test_file
        self.test_result = test_result
        self.sess = tf.Session()
        self.isload = self.load_model(self.sess)
        self.seq_len = 200 

    def cut_line(self, line):
        not_cuts = re.compile(u'[。？！\?!]')
        result = []
        start = 0
        
        for seg_sign in not_cuts.finditer(line):
            result.extend(self.word_pos(line[start:seg_sign.end()], self.sess))
            start = seg_sign.end()+1
        
        return result

    def word_pos(self, sentence, sess=None):
        pos = pd.read_csv('./pos.csv')
        id2tag = {ii: label for ii, label in enumerate(list(pos['POS']),1)}
        id2tag[0]='' # 占位符0置空
        print(sentence)
        word_ints = []
        words_ints = []
        word_ = sentence.split('/')
        for word in word_:
            if word in self.data.word2id: 
                word_ints.append(self.data.word2id[word]) 
            else:
                word_ints.append('126') # 未登录词暂处理成w
        words_ints.append(word_ints)
        
        X_batch = np.zeros((len(words_ints), self.seq_len), dtype=int)
        for i, row in enumerate(words_ints):
            X_batch[i, :len(row)] = np.array(row)[:self.seq_len]
                        
        fetches = [self.model.scores, self.model.length, self.model.transition_params]
        feed_dict = {self.model.X_inputs: X_batch, self.model.lr: 1.0, self.model.batch_size: 1,
                     self.model.keep_prob: 1.0}
        test_score, test_length, transition_params = sess.run(fetches, feed_dict)
        tags, _ = crf.viterbi_decode(test_score[0][:test_length[0]], transition_params)

        tags = [id2tag[i] for i in tags]

        return tags
    
    def format_standardization(self, text):
        pass

    def testfile(self):     
        
        isFile = os.path.isfile(self.test_file)
        if isFile:
            with open(self.test_result, "w", encoding='utf-8') as out: 
                with open(self.test_file, "r", encoding='utf-8') as fp:
                    for line in fp.readlines(): 
                        line = line.encode('utf-8').decode('utf-8-sig')
                        line = line.strip().strip('/') 
                        print(line)  
                        tags = self.cut_line(line)
                        word_ = line.split('/')
                        print(len(word_))
                        print(len(tags))                    
                        result = ''
                        for i in range(len(word_)):
                            result = result + word_[i] + '/' +tags[i] + ' '
                        print(result)
                        out.write("%s\n" % (result))

    def load_model(self, sess=None):
        isload = False

        self.model = modelDef.BiLSTMModel(vocab_size=self.data.word2id.__len__(), class_num=self.data.tag2id.__len__())
        ckpt = tf.train.latest_checkpoint(self.model_path)
        print(u'\n 加载模型checkpoint完成,路径为：'+self.model_path+' ...\n')
        saver = tf.train.Saver()
        if ckpt:
            saver.restore(sess, ckpt)
            isload = True
        return isload

def main(_):
    data = readData.DataHandler(save_path=FLAGS.dict_path)
    data.loadData()
    print(u'\n 加载字典完成，路径为：'+FLAGS.dict_path+' ...\n')

    test = BiLSTMTest(data)
    if test.isload:
        test.testfile()

if __name__ == '__main__':
    tf.app.run()
   
