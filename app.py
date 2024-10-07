from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import utlis

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process_image", methods=["POST"])
def process_image():
    data = request.json
    image_data = data["image"]
    threshold_value = int(data["threshold"])

    # Convert base64 image data to OpenCV format
    img_str = base64.b64decode(image_data.split(",")[1])
    np_arr = np.frombuffer(img_str, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Set height and width for the image
    heightImg, widthImg = 640, 480
    img = cv2.resize(img, (widthImg, heightImg))

    # Begin the same processing steps as in Main.py
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to gray scale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Add Gaussian Blur
    imgThreshold = cv2.Canny(
        imgBlur, threshold_value, threshold_value
    )  # Apply Canny for edges
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # Apply Dilation
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # Apply Erosion

    # Find all contours
    contours, _ = cv2.findContours(
        imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    imgContours = img.copy()
    cv2.drawContours(
        imgContours, contours, -1, (0, 255, 0), 10
    )  # Draw all detected contours

    # Find the biggest contour
    biggest, _ = utlis.biggestContour(contours)  # Get the biggest contour
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        pts1 = np.float32(biggest)  # Prepare points for perspective transformation
        pts2 = np.float32(
            [[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]
        )
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # Remove 20 pixels from each side
        imgWarpColored = imgWarpColored[
            20 : imgWarpColored.shape[0] - 20, 20 : imgWarpColored.shape[1] - 20
        ]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

        # Apply adaptive threshold
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(
            imgWarpGray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        # Encode the final processed image to return it to the frontend
        _, buffer = cv2.imencode(".jpg", imgWarpColored)
        processed_image = base64.b64encode(buffer).decode("utf-8")

        return jsonify({"processed": processed_image})
    else:
        # Return an error if no document is found
        return jsonify({"error": "No document found"})


# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
