import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

confidence_threshold = 0.55
with open("..\\..\\models\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

path_prefix = "..\\..\\models"
model_cfg = f"{path_prefix}\\yolov3-608.cfg"
model_weights = f"{path_prefix}\\yolov3-608.weights"
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)

def get_output_layers(net):
    layer_names = net.getLayerNames()
    # print(layer_names)
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    # print(output_layers)

    return output_layers

def object_detection(image):
    blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(608, 608), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    hT, wT, sT = image.shape
    
    detections = []
    blank_image = np.zeros_like(image)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                detections.append({"class_id": classes[class_id], "confidence": confidence})
                # Calculate bounding box coordinates
                # Draw bounding box
                if (classes[class_id] == "person"):
                    w, h = int(detection[2] * wT), int(detection[3] * hT)
                    x, y = int((detection[0] * wT) - w / 2), int((detection[1] * hT) - h / 2)
                    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.rectangle(blank_image, (x, y), (x+w, y+h), (0, 0, 0), -1)
    
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rotated_blank_image = cv2.rotate(blank_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    plt.imshow(rotated_image)
    plt.show()


def main():
    print("Image Analysis - NFL Defenses")
    nfl_pictures = glob.glob("nfl_coaches_film/*.png")
    
    for picture in nfl_pictures:
        print(picture)
        image = cv2.imread(picture)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        object_detection(rgb_image)

if __name__ == "__main__":
    main() 