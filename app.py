from flask import Flask, jsonify, request
from flask_cors import CORS
import UrinoxCode  # Replace with your actual module

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Assuming the image is sent as a file in the request
        print(request)
        
        # Check if 'image' is in the request.files
        if 'image' not in request.files:
            raise ValueError("No 'image' file provided in the request.")
        
        image_file = request.files['image']
        print(image_file.filename)
        
        # Ensure that the uploaded file has a valid filename
        if not image_file.filename:
            raise ValueError("Uploaded file has no filename.")
        
        # Call your OpenCV processing function
        result_text = UrinoxCode.fun(image_file.filename)
        
        return jsonify({'result': result_text})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

