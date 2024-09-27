import cv2
import numpy as np
from djitellopy import tello

# cap = cv2.VideoCapture(0)

classNames = []
with open('utils/resources/coco.names', 'r') as f:
    classNames = f.read().splitlines()
# print(len(classNames))
print(classNames)

model = 'utils/resources/yolov3-tiny.weights'
config = 'utils/resources/yolov3-tiny.cfg'
net = cv2.dnn.readNet(model, config)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

me = tello.Tello()
me.connect()
print("Battery: ", me.get_battery(), " %")

me.streamoff()
me.streamon()

me.takeoff()
me.move_up(120)

x_cord = 760
y_cord = 0

cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)
cv2.moveWindow("Video Feed", x_cord, y_cord)


def find_object(outputs, img):
    height, width, c = img.shape
    bboxes = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            conf = score[class_id]
            if conf > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                bboxes.append([x, y, w, h])
                class_ids.append(class_id)
                confidences.append(float(conf))
    # print(len(bboxes), len(classIds))
    # Non maximum suppression (NMS)
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, 0.5, 0.2)

    for i in indices:
        x, y, w, h = bboxes[i]
        label = str(classNames[class_ids[i]]).upper()
        confidence = str(round(confidences[i] * 100, 2))
        color = (255, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + ' ' + confidence, (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)


while True:
    img = me.get_frame_read().frame

    # _, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    # print(len(layerNames))
    # print(net.getUnconnectedOutLayers())
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    outputs = net.forward(outputNames)
    # print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
    # outputs: cx cy w h conf prob_score_for_80_class

    find_object(outputs, img)

    me.send_rc_control(0, 0, 0, 0)

    cv2.imshow("Video Feed", img)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        me.move_forward(50)
        print("Forward Command Has Been Executed.")
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        me.move_back(50)
        print("Back Command Has Been Executed.")
    elif cv2.waitKey(1) & 0xFF == ord('a'):
        me.move_left(25)
        print("Left Command Has Been Executed.")
    elif cv2.waitKey(1) & 0xFF == ord('d'):
        me.move_right(25)
        print("Right Command Has Been Executed.")
    elif cv2.waitKey(1) & 0xFF == ord('e'):
        me.move_up(25)
        print("Up Command Has Been Executed.")
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        me.move_down(25)
        print("Down Command Has Been Executed.")
    elif cv2.waitKey(1) & 0xFF == ord('j'):
        me.rotate_clockwise(90)
        print("Rotate Clockwise Command Has Been Executed.")
    elif cv2.waitKey(1) & 0xFF == ord('k'):
        me.rotate_counter_clockwise(90)
        print("Rotate Counter-Clockwise Command Has Been Executed.")
    elif cv2.waitKey(1) & 0xFF == ord('x'):
        print("Exit Command Has Been Executed.")
        break

cv2.destroyAllWindows()
me.land()
me.reboot()
