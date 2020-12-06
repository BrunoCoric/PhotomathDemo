from flask import Flask,jsonify,request, render_template
import io
from PIL import Image
from combined import get_prediction, transform_image, getExpression

app = Flask(__name__,static_url_path='/static')


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def render_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('image')
        rot = request.form['rotation']
        if file is None or file.filename == "":
            return render_template('index.html',err= "Invalid file upload")
        if not allowed_file(file.filename):
            return render_template('index.html',err= "Format not supported")

        try:
            img_bytes = file.read()
            image = io.BytesIO(img_bytes)
            image = Image.open(image)
            if rot == "leftRotation":
                image = image.rotate(-90)
            elif rot == "rightRotation":
                image = image.rotate(90)
            image = getExpression(image)
            tensor = transform_image(image)
            prediction,ocitano = get_prediction(tensor)
            return render_template('rezultat.html', rez=prediction,ocitano = ocitano)
        except:
            return render_template('index.html',err= "Error during prediction")