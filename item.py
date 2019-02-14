import pandas as pd
import numpy as np
import json

import sklearn
from sklearn.externals import joblib

from flask_restful import Resource, reqparse

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from kerascapture import training


ALLOWED_EXTENSIONS = set(['csv'])


def allowed_file(filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class Predict(Resource):

    def post(self):
        parser=reqparse.RequestParser()
        parser.add_argument('option',type=str,required=True,help="field cannot be empty")
        parser.add_argument('name',type=str,required=True,help="field cannot be empty")
        data=parser.parse_args()

        if 'xfile' not in request.files or 'yfile' not in request.files:
            return { 'Message' : 'xfile and yfile required!!' }, 404
        x_file = request.files['xfile']
        y_file = request.files['yfile']
        
        if not allowed_file(x_file.filename) or not allowed_file(y_file.filename):
            return { 'Message' : 'File type not allowed!!' }, 404
        
        #delete this file
        x_file.save('dataset/train.csv')
        y_file.save('dataset/test.csv')
        try:
            result=training(data.option)
            response=json.dumps({'name':data.name,'prediction':np.array(result).tolist()})
        except:
            return {'Message':'Error in prediction'},500
        finally:
            import os
            os.remove('dataset/train.csv')
            os.remove('dataset/test.csv')
        return response,202 
    
