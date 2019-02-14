from flask import Flask, jsonify
from flask_restful import Api
from flask_jwt import JWT
import firebase_admin
from firebase_admin import credentials
from item import Predict

app = Flask(__name__)
#app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

app.secret_key = 'now'
api=Api(app)

#jwt = JWT(app, authenticate, identity)

api.add_resource(Predict, '/predict', methods=['POST'])


app.run(port=5000)




