#!flask/bin/python
#coding=utf-8
from flask import Flask, jsonify, request, abort
import seg_model_test
import data_read as readData
import bi_lstm_model as modelDef
import tensorflow as tf

tf.app.flags.DEFINE_string('dict_path', 'data/data_ner_0409.pkl', 'dict path')
tf.app.flags.DEFINE_string('model_path', 'ckpt/ner0409', 'model path')

FLAGS = tf.app.flags.FLAGS

app = Flask(__name__)

@app.route('/ner_text', methods=['POST'])
def create_task():
    if not request.json or not 'text' in request.json:
        abort(400)

    task = {
        
        'text': request.json['text'],
        'description': request.json.get('description', ""),
        'done': False
    }
    
    text = task['text']
    
    result = test.cut_word(text.strip())

    rss = ''
    for each in result:
        if isinstance(each, list):
            # rss = rss +each[0] + '/' # no NER
            rss = rss + each[0] + ' /' + each[1] + ' ' # NER
        else:
            rss = rss + each + '/'

    print(text)
    print(rss)
    
    return jsonify({'out': rss}), 201

def main(_):
    data = readData.DataHandler(save_path=FLAGS.dict_path)
    data.loadData()
    test = seg_model_test.BiLSTMTest(data, model_path=FLAGS.model_path)  
    print(u'预测模型服务启动完成')   
    app.run(debug=True)   

if __name__ == '__main__':
    tf.app.run()

# curl -i -H "Content-Type: application/json" -X POST -d '{"text":"\u5f20\u51cc\u745e\u3002"}' http://localhost:5000/ner_text
