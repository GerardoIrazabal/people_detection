import cv2
import imutils
import numpy as np

def nms(boxes, scores, threshold):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep

# Load YOLOv4 model
weights_path = "the folder path/yolov4.weights"
config_path = "the folder path/yolov4.cfg"

net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten().tolist()]

# Load classes
with open("the folder path/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open video file
video_path = "the folder path/video1.mp4"
cap = cv2.VideoCapture()
cap.open(video_path, cv2.CAP_FFMPEG)

ret = True

while cap.isOpened() and ret:
    # Reading the video stream
    ret, img = cap.read()
    if ret:
        # Resto del código
        img = imutils.resize(img, width=min(800, img.shape[1]))
        height, width, _ = img.shape

        # Preprocessing the image for YOLOv4
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and class_id == 0:  # 0 is the index for the "person" class
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, x + w, y + h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = nms(np.array(boxes), np.array(confidences), 0.5)

        for i in indices:
            x, y, x2, y2 = boxes[i]
            cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)
            text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Resize the output image
        output_width = 800
        output_height = int(img.shape[0] * output_width / img.shape[1])
        img_resized = cv2.resize(img, (output_width, output_height))

        # Showing the output Image
        cv2.imshow("Image", img_resized)

        # Wait for 1 millisecond and check for 'q' key press
        key = cv2.waitKey(5000)
        if key & 0xFF == ord('q'):
            break
    else:
        break

    cap.release()
    cv2.destroyAllWindows()
