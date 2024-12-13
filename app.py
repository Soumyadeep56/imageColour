from flask import Flask, request, render_template, send_file
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import joblib
import imutils
import uuid

app = Flask(__name__)

# Set paths for uploaded and output images
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/outputs/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load pre-trained KMeans model
MODEL_PATH = 'kmeans_model.pkl'
kmeans = joblib.load(MODEL_PATH)

def process_image(image_path, clusters=5):
    """
    Processes an image to extract dominant colors and generate an output image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image file.")

    original_img = img.copy()
    img = imutils.resize(img, height=200)
    flat_img = np.reshape(img, (-1, 3))

    labels = kmeans.predict(flat_img)
    dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint')

    percentages = (np.unique(labels, return_counts=True)[1]) / flat_img.shape[0]
    p_and_c = sorted(zip(percentages, dominant_colors), reverse=True)

    rows = 1000
    cols = int((original_img.shape[0] / original_img.shape[1]) * rows)
    img_resized = cv2.resize(original_img, dsize=(rows, cols), interpolation=cv2.INTER_LINEAR)

    copy = img_resized.copy()
    cv2.rectangle(copy, (rows // 2 - 250, cols // 2 - 90), (rows // 2 + 250, cols // 2 + 110), (255, 255, 255), -1)

    final_image = cv2.addWeighted(img_resized, 0.1, copy, 0.9, 0)
    cv2.putText(final_image, 'Most Dominant Colors in the Image', (rows // 2 - 230, cols // 2 - 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    start = rows // 2 - 220
    for i in range(clusters):
        end = start + 70
        final_image[cols // 2:cols // 2 + 70, start:end] = p_and_c[i][1]
        cv2.putText(final_image, str(i + 1), (start + 25, cols // 2 + 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        start = end + 20

    return final_image

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part in the request", 400

        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        # Save the uploaded file
        file_id = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_FOLDER, file_id + '.jpg')
        file.save(input_path)

        # Process the image and save the output
        try:
            output_img = process_image(input_path)
            output_path = os.path.join(OUTPUT_FOLDER, file_id + '_output.jpg')
            cv2.imwrite(output_path, output_img)
            return render_template('index.html', output_image=output_path)
        except Exception as e:
            return f"Error processing the image: {e}", 500

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
