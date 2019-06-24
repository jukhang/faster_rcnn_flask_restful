from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import cv2
import os
from lib.faster_rcnn_api import im_detect 

app = Flask(__name__)
api = Api(app)

RE = {
    'get' : {'DOC' : 'Faster Rcnn Images Recognition Restful Api'}
}

parser = reqparse.RequestParser()
parser.add_argument('image', type=FileStorage, location='files')

class FsterRcnn(Resource):
    def get(self):
        return RE['get']

    def post(self):
        args = parser.parse_args()
        
        # store image file
        im_file = args.get('image')
        im_name = secure_filename(im_file.filename)
        im_file.save(os.path.join('/data/wxh/www/image/', im_name))
        
        # faster rcnn
        im_file = os.path.join('/data/wxh/www/image/',im_name)
        im = cv2.imread(im_file)
        im_detect(im, im_name)
        return 201


api.add_resource(FsterRcnn, '/faster-rcnn')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8383, debug=True)
