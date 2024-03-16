import warnings
warnings.filterwarnings("ignore")

from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS

from model import init_model, Q_A

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def init():
    return jsonify({"message" : "Hi"}), 200

@app.route('/api/question', methods=['POST'])
def post_question():
    json = request.get_json(silent=True)
    question = json['question']

    resp = Q_A(question)
    
    print(resp)

    data = {
        'question' : json['question'],
        'answer' : resp
        }

    return jsonify(data), 200

if __name__ == '__main__':
    init_model()
    app.run(debug=True)




