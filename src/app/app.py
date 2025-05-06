import os
from flask import Flask, render_template, request
from utils.predict_classification import predict_classification
from utils.predict_segmentation import segment_tumor

UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def is_brats_file(filename):
    return filename.lower().endswith(('.nii', '.nii.gz'))

def is_image_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png'))

@app.route('/', methods=['GET', 'POST'])
def index():
    result_image = None
    input_image = None
    prediction_label = None
    prediction_accuracy = None
    prediction_details = None

    if request.method == 'POST':
        uploaded_file = request.files['file']
        gt_file = request.files.get('gtmask')  # Ground truth file
        model_type = request.form['model']

        if uploaded_file and uploaded_file.filename != '':
            filename = uploaded_file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)

            # Save ground truth if available
            gt_path = None
            if gt_file and gt_file.filename != '':
                gt_filename = "gtmask.png"
                gt_path = os.path.join(app.config['UPLOAD_FOLDER'], gt_filename)
                gt_file.save(gt_path)

            try:
                if model_type in ['unet', 'attention'] and not is_brats_file(filename):
                    prediction_label = "Dataset Type Error"
                    prediction_details = {
                        "Error": "Segmentation model requires NIfTI",
                        "Solution": "Try to upload a .nii or .nii.gz file"
                    }
                elif model_type in ['resnet', 'vit'] and not is_image_file(filename):
                    prediction_label = "Dataset Type Error"
                    prediction_details = {
                        "Error": "You uploaded 3D volume but selected classification model",
                        "Solution": "For Classification Models choose JPG/PNG Images"
                    }
                else:
                    if is_brats_file(filename):
                        result = segment_tumor(file_path, model_type, gt_path=gt_path)
                        if result and isinstance(result, tuple) and len(result) == 3:
                            input_path, result_path, metrics = result
                            input_image = os.path.basename(input_path)
                            result_image = os.path.basename(result_path)
                            prediction_label = "Segmentation Completed"
                            prediction_details = metrics
                    else:
                        result = predict_classification(file_path, model_type)
                        prediction_label = result.get('label')
                        prediction_accuracy = result.get('accuracy')
                        prediction_details = result.get('details')

            except Exception as e:
                prediction_label = "Segmentation/Classification Failed"
                prediction_details = {"Error": str(e)}
                prediction_accuracy = 0

    return render_template(
        'index.html',
        input_image=input_image,
        result_image=result_image,
        prediction_label=prediction_label,
        prediction_accuracy=prediction_accuracy,
        prediction_details=prediction_details
    )

if __name__ == '__main__':
    app.run(debug=True)
