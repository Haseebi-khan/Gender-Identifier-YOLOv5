from flask import *
import cv2
import torch
import math
# seaborn, matlib, numpy, pandas and ultralytics/yolov5


app = Flask(__name__)

camera = cv2.VideoCapture(0) 

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
# Corrected path using a raw string (r'...')
camModel = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\NoNeo\Desktop\MLProject\bestModel.pt', force_reload=True)

# i tool the dataset from roboflow adn train model on kaggle gpu tasla p100,
# kaggle provide 100 free hour gpu to students  


faceLimit = 1
faceCount = 0

# video stream frames
def gen_frames():
    global faceLimit
    global faceCount  
    while True:
        
        success, frame = camera.read()  
        if not success:
            break
        else:
            faceCount = 0
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = camModel(frameRGB)
            for box in results.xyxy[0]:
                if faceCount < faceLimit:
                    faceCount += 1 
                    if box[5] == 1:
                        className = "Male:"
                        bgr =(230, 216, 173) # A light beige/tan color
                    elif box[5] == 0:
                        className = "Female:"
                        bgr =(203, 192, 255) # A light lavender color
                    
                    conf = math.floor(box[4] * 100)
                    xA, yA, xB, yB = map(int, box[:4]) # Unpack coordinates more cleanly
                        
                    cv2.rectangle(frame, (xA, yA), (xB, yB), (bgr), 4)
                    # Adjust text background rectangle to fit text and confidence
                    text_size_class = cv2.getTextSize(className, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
                    text_size_conf = cv2.getTextSize(str(conf), cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
                    # Calculate width needed for both text and confidence
                    total_text_width = text_size_class[0] + text_size_conf[0] + 10 # 10 for spacing
                    cv2.rectangle(frame, (xA, yA - 50), (xA + total_text_width, yA), (bgr), -1)
                    
                    cv2.putText(frame, className, (xA, yA - 15), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2)
                    cv2.putText(frame, str(conf), (xA + text_size_class[0] + 5, yA - 15), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2)
                else:
                    break

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    # CORRECTED: Flask's render_template expects a template name relative to the 'templates' folder,
    # not an absolute path. Assuming index.html is in C:\Users\NoNeo\Desktop\MLProject\templates\
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=50000, debug=True)
