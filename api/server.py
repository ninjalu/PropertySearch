from flask import Flask, request
import json

app = Flask(__name__)


@app.route('/')  # if somebody accesses host:post/ this function will run
def root():
    print('hello')
    return json.dumps({
        'response': 'hello world'
    })


@app.route('/predict', methods=['POST'], header={'content_type': 'application/json'})
def predict():
    print(request)
    x = request.get_json()
    return 'hello'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
