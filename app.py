from flask import Flask, request, render_template, send_from_directory
import os
from classifier import predict

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_image = None
    preprocessed_image = None
    prediction = None
    confidence_level = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            preprocessed_path, result_label, result_confidence = predict(filepath)
            uploaded_image = file.filename
            preprocessed_image = os.path.basename(preprocessed_path)
            prediction = result_label
            confidence_level = result_confidence
    return render_template(
        'index.html',
        uploaded_image=uploaded_image,
        preprocessed_image=preprocessed_image,
        prediction=prediction,
        confidence_level=confidence_level
    )

@app.route('/static/uploads/<filename>')
def send_uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/results/<filename>')
def send_preprocessed_image(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
