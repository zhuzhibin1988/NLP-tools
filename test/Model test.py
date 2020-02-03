#!/usr/bin/env python
# coding: utf-8

# ###### 模型加载

# In[1]:


from cws.segmenter import BiLSTMSegmenter

segmenter = BiLSTMSegmenter(data_path='D:/workspace/trsnlp/data/cws_pos/cws_pos_dict.pkl', model_path='D:/workspace/trsnlp/checkpoints/cws_pos_1107')

# ###### 模型预测

# In[2]:


# 示例1
segmenter.predict('2003年10月15日，杨利伟乘由长征二号F火箭运载的神舟五号飞船首次进入太空，象征着中国太空事业向前迈进一大步，起到了里程碑的作用。')

# ###### 示例2、3说明“记录”一词在不同环境下为不同的词性

# In[3]:


# 示例2
print('示例2：', segmenter.predict('我和林长开的通话记录'))
# 示例3
print('示例3：', segmenter.predict('记录一段文字'))

# ###### 示例4、5说明模型对罕见人名、少数名族姓名识别能力

# In[4]:


# 示例4
print('示例4：', segmenter.predict('车雨菲在挣扎中被对方用锐器划伤颈部'))
# 示例5
# 对比Hanlp(http://hanlp.com/)“吉和日布”识别错误
print('示例5：', segmenter.predict('2002年5月23日下午，胡志勇在新市镇新市街遇到吉和日布等人'))

# ###### 示例6新字（词）识别能力

# In[5]:


# 示例6
# “漹”字未出现在训练集
print('示例6：', segmenter.predict('漹城镇'))