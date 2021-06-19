from flask import Flask, render_template, Response
import cvlib
import cv2
import numpy as np

webcam = cv2.VideoCapture(0)


def gender_detection():
    padding = 20
    while webcam.isOpened():
        _, frame = webcam.read()
        face, confidence = cvlib.detect_face(frame)  # inbuilt cvlib facial detection algo
        for idx, f in enumerate(face):
            x0, y0 = max(0, f[0] - padding), max(0, f[1] - padding)  # x0 xy coordiated of the face - padding no -ve cords
            x1, y1 = min(frame.shape[1] - 1, f[2] + padding), min(frame.shape[0] - 1, f[3] + padding)  # x1 y1
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)  # color of rect box and thickness
            face_crop = np.copy(frame[y0:y1, x0:x1])  # extracting the face
            (label, confidence) = cvlib.detect_gender(face_crop)  # probs values of male and female in confidence
            idx = np.argmax(confidence)
            label = label[idx]
            label = "{}: {:.2f}%".format(label, confidence[idx] * 100)
            y = y0 - 10 if y0 - 10 > 10 else y0 + 10
            cv2.putText(frame, label, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Gender detection", frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        if cv2.waitKey(1) & 0xFF == ord('s'):  # press s to break the loop
            break
    webcam.release()
    cv2.destroyAllWindows()


app=Flask(__name__)


@app.route('/', methods=['POST', 'GET', ])
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST', 'GET', ])
def result():
    return Response(gender_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)