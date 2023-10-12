from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- Import CORS

app = Flask(__name__)
CORS(app)  # <-- Enable CORS for the app

@app.route('/receive_json', methods=['POST'])
def receive_json():
    data = request.json
    print(data)
    return jsonify({"message": "JSON received!"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

