# encoding=utf8
from flask import Flask, request, abort
from cws.segmenter import BiLSTMSegmenter


app = Flask(__name__)

@app.route('/cws', methods=['POST'])
def segment():
    if not request.json:
        abort(400)
    text = request.json['text']
    if not text or text == '':
        abort(400)
    return segmenter.predict(text)


if __name__ == '__main__':
    segmenter = BiLSTMSegmenter(data_path='data/your_dict.pkl', model_path='checkpoints/cws.ckpt/')
    app.run(host='0.0.0.0', port=7777, debug=True, use_reloader=False)
