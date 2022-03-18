from flask import Flask, jsonify, request
import numpy as np

from nlp_utils.text_processing import predict_topic
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/topic_model', methods=['GET', 'POST'])
def topic_model():
    model_path = 'models/model_8_symmetric_0.91.gensim'
    sent = request.get_json()['sent']
    results = predict_topic(sent, model_path)
    # we're incrementing the zero-based index to 1-based to match the viz
    result_stringified = {k + 1 :(str(round(v,5)) if isinstance(v,np.float32) else v) for (k,v) in results.items()}
    return jsonify(result_stringified)


if __name__ == '__main__':
    app.run()
