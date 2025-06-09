from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load YOLO models
cell_model = YOLO('yolo.pt')  # Cell detector
cell_classify_model = YOLO('good_bad_classifier_v1.pt')  # Good/Bad classifier
bad_type_model = YOLO('best (1).pt')  # Multiclass classifier

# Define class names for the bad_type_model manually
bad_type_class_names = ['branch3', 'hotspot3', 'line3']

# Create folders if not exist
original_image_dir = 'original_images'
processed_image_dir = 'processed_images'
for directory in [original_image_dir, processed_image_dir]:
    os.makedirs(directory, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/cell-detection', methods=['POST'])
def cell_detection():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Save original image
    original_filename = f"original_{uuid.uuid4().hex}.jpg"
    original_path = os.path.join(original_image_dir, original_filename)
    cv2.imwrite(original_path, image)

    # Create image copies
    detected_img = image.copy()
    classified_img = image.copy()
    multiclass_img = image.copy()

    # Detect cells
    results = cell_model(image)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
            label = f"{cell_model.names[int(box.cls[0])]} {box.conf[0]:.2f}"

            # Draw basic detection
            cv2.rectangle(detected_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(detected_img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Classify Good/Bad
            result_cls = cell_classify_model(crop)
            class_id = int(result_cls[0].probs.top1)
            class_label = cell_classify_model.names[class_id].lower()

            color = (0, 255, 0) if class_label == "good" else (0, 0, 255)
            final_label = class_label

            # If bad, further classify
            if class_label == "bad":
                result_bad_type = bad_type_model(crop)
                bad_type_id = int(result_bad_type[0].probs.top1)
                bad_type_label = bad_type_class_names[bad_type_id]
                final_label = f"{class_label} - {bad_type_label}"

                # Draw on multiclass image
                cv2.rectangle(multiclass_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(multiclass_img, bad_type_label, (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw on classified image
            cv2.rectangle(classified_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(classified_img, final_label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save all images
    detected_filename = f"detected_{uuid.uuid4().hex}.jpg"
    classified_filename = f"classified_{uuid.uuid4().hex}.jpg"
    multiclass_filename = f"multiclass_{uuid.uuid4().hex}.jpg"

    cv2.imwrite(os.path.join(processed_image_dir, detected_filename), detected_img)
    cv2.imwrite(os.path.join(processed_image_dir, classified_filename), classified_img)
    cv2.imwrite(os.path.join(processed_image_dir, multiclass_filename), multiclass_img)

    return jsonify({
        'original_image': original_filename,
        'processed_image': detected_filename,
        'classified_image': classified_filename,
        'multiclass_image': multiclass_filename
    })

@app.route('/original/<path:filename>')
def serve_original(filename):
    return send_from_directory(original_image_dir, filename)

@app.route('/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(processed_image_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)
