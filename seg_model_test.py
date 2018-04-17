#encoding=utf8
import re
import os
import numpy as np
import tensorflow as tf
import data_preprocess as readData
import bi_lstm_model as modelDef
from sentence import TagSurfix
from tensorflow.contrib import crf

tf.app.flags.DEFINE_string('dict_path', 'data/data_ner_0409.pkl', 'dict path')
tf.app.flags.DEFINE_string('model_path', 'ckpt/ner0409', 'model path')
tf.app.flags.DEFINE_string('test_file', 'test/jyaq.csv', 'test file path')
tf.app.flags.DEFINE_string('test_result', 'test/ner_result.ext', 'test result path')

FLAGS = tf.app.flags.FLAGS

class BiLSTMTest(object):
    def __init__(self, data=None, model_path=FLAGS.model_path,
                 test_file=FLAGS.test_file, test_result=FLAGS.test_result):
        self.data = data
        self.model_path = model_path
        self.test_file = test_file
        self.test_result = test_result
        self.sess = tf.Session()
        self.isload = self.loadModel(self.sess)
 
    def text2ids(self, text):
        words = list(text)
        ids = list(self.data.word2id[words].fillna(self.data.word2id['<NEW>']))  

        if len(ids) >= self.data.max_len: 
            print(u'输入句长超过%d，无法完成处理！' % (self.data.max_len))
            ids = ids[:self.data.max_len]
            ids = np.asarray(ids).reshape([-1, self.data.max_len])
            return ids
        else:
            ids.extend([0] * (self.data.max_len - len(ids))) 
            ids = np.asarray(ids).reshape([-1, self.data.max_len])
            return ids

    def simple_cut(self, text, sess=None):
        if text:
            X_batch = self.text2ids(text) 
     
            fetches = [self.model.scores, self.model.length, self.model.transition_params]
            feed_dict = {self.model.X_inputs: X_batch, self.model.lr: 1.0, self.model.batch_size: 1,
                         self.model.keep_prob: 1.0}
            test_score, test_length, transition_params = sess.run(fetches, feed_dict) 
            tags, _ = crf.viterbi_decode(
                test_score[0][:test_length[0]], transition_params)
            

            tags = list(self.data.id2tag[tags])
            
            # force change which the begin is not [s] or [b] , this error may show the RNN mistakes.
            if tags[0] != TagSurfix.S.value and not tags[0].endswith(TagSurfix.B.value): 
                tags[0] = TagSurfix.S.value
 
            words = []
            for i in range(len(tags)):
           	 
                if tags[i] == TagSurfix.S.value or tags[i].endswith(TagSurfix.B.value):
                    if tags[i].endswith('_' + TagSurfix.B.value):
                        words.append([text[i], tags[i][:tags[i].find('_')]]) 
                    else:
                        words.append(text[i])
                else:
                    if isinstance(words[-1], list):
                        words[-1][0] += text[i]
                    else:
                        words[-1] += text[i]

            return words
        else:
            return []

    # 分句方式 1
    def cut_word(self, sentence):
        not_cuts = re.compile(u'[。.？！\?!]')
        result = []
        start = 0

        for seg_sign in not_cuts.finditer(sentence):
            result.extend(self.simple_cut(sentence[start:seg_sign.start()], self.sess))
            result.append(sentence[seg_sign.start():seg_sign.end()]) 
            start = seg_sign.end()
        result.extend(self.simple_cut(sentence[start:], self.sess))
        
        return result
    
    # 分句方式 2
    def cut_sentence(self, line, max_len):
        newLine = []
        tags_result = []
        # 小写转大写、去空格、去/n /t
        line = line.replace(' ','').strip('"').replace('"','“').replace('!','！').replace('?','？').strip()
        print(' ')
        print('TEXT:',line)
        # 去段尾句号
        line = line.strip('.').strip('。') 
        # 分句，还原句号
        line = re.split(r'[。？！]', line)
        line = [i + '。' for i in line]
        
            # 超过最大句长max_len进行逗号分句
        def segSent(sentence):
            sli = sentence.split('，')
            lenSli = [ len(i) for i in sli]
            sLen = 0
            for i in lenSli:
                sLen += i+1
                if sLen > max_len:
                    sLen -= i+1
                    break
            newLine.append(sentence[:sLen])
            if len(sentence[sLen:]) > max_len:
                segSent(sentence[sLen:])
            else:
                newLine.append(sentence[sLen:])
            return newLine

        for sentence in line:
            if len(sentence) > max_len:
                segSent(sentence)
            else:
                newLine.append(sentence)
        
        for sentence in newLine: 
            tags_result.extend(self.simple_cut(sentence, self.sess))
    
        return tags_result


    def testfile(self):
        isFile = os.path.isfile(self.test_file)
        if isFile:
            with open(self.test_result, "w", encoding='utf-8') as out: 
                with open(self.test_file, "r", encoding='utf-8') as fp:
                    for line in fp.readlines():
                        line = line.encode('utf-8').decode('utf-8-sig')
                        if len(line.strip()) > 1: # 阻断空白文档
                       # 分句方式 1 
                       # line = line.strip('"')
                       # line = line.replace('"','“').replace('"','”').replace(' ','')
                       # result = self.cut_word(line.strip())
                       #
                       # 分句方式 2 
                            result = self.cut_sentence(line, self.data.max_len)
                            

                            rss = ''
                            DAT = []
                            LOC = []
                            PEO = []

                            for each in result: 
                                if isinstance(each, list):     
                                    # rss = rss +each[0] + '/' # no NER
                                    rss = rss + each[0] + ' /' + each[1] + ' ' # NER

                                    if each[1] == 'DAT':
                                        DAT.append(each[0])
                                    elif each[1] == 'LOC':
                                        LOC.append(each[0])
                                    elif each[1] == 'PEO':
                                        PEO.append(each[0])

                                else:
                                    rss = rss + each + '/'
                            
                            # 去重
                            PEO = list(set(PEO))
                            PE0 = ''
                            for i in PEO:
                                PE0 = PE0 + i + ' '

                            DAT = DAT
                            DAt = ''
                            for i in DAT:
                                if i != '。':
                                    DAt = DAt + i + ' '

                            LOC_ = list(set(LOC))
                            LOC_.sort(key = LOC.index)
                            L0C = ''
                            for i in LOC_:
                                L0C = L0C + i + ' '

                            print('DATE:',DAt)
                            print('LOCTION:',L0C) 
                            print('PEOPLE:',PE0)
                            print(' ')
                            #print(rss)
                            #out.write("%s\n" % (rss))

    def loadModel(self, sess=None):
        isload = False
        self.model = modelDef.BiLSTMModel(vocab_size=self.data.word2id.__len__(), class_num=self.data.tag2id.__len__())
        ckpt = tf.train.latest_checkpoint(self.model_path)
        print(u'加载模型checkpoint完成，路径为：'+ckpt+'\n')
        saver = tf.train.Saver()
        if ckpt:
            saver.restore(sess, ckpt)
            isload = True
        return isload

def main(_):
    data = readData.DataHandler(save_path = FLAGS.dict_path)
    data.loadData()
    print(u'加载字典完成！\n')
    test = BiLSTMTest(data)

    if test.isload:
        test.testfile()

if __name__ == '__main__':
    tf.app.run()
    
