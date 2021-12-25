import spacy
from anytree import Node, RenderTree, PreOrderIter
# YOLO PACKAGES
import numpy as np
import cv2
import sys
import csv


lambdaList = []
updatedLambdaList = []
root = None
itemList = []
probabilityDic = {}
totalObjects = 0
totalFoundObjects = 0
resultProbability = 0
objectCache = {}

def findObjectsThroughYolo(image, count):
    global itemList
    global objectCache
    confidenceThreshold = 0.8
    threshold = 0.3
    # load the COCO class labels our YOLO model was trained on
    labelsPath = 'yolo\\yolo-coco\\coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")
    # print(LABELS)

    # initialize a list of colors to represent each possible class label
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    # paths to the YOLO weights and model configuration
    weightsPath = 'yolo\\yolo-coco\\yolov3.weights'
    configPath = 'yolo\\yolo-coco\\yolov3.cfg'

    # load our YOLO object detector trained on COCO dataset (80 classes)
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load our input image and grab its spatial dimensions
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidenceThreshold:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold,
        threshold)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
            # if (LABELS[classIDs[i]] not in itemList):
            #     itemList.append(LABELS[classIDs[i]])
            #     # UPDATE PRBABILITY OF THE OBJECTS IN OUR ORIGINAL G3 MAP
            #     updateG3Map(LABELS[classIDs[i]])
            #     cv2.imwrite(str(count) +".jpg", image)
            updateG3 = False
            if (LABELS[classIDs[i]] not in objectCache):
                objectCache[LABELS[classIDs[i]]] = 1
                # updateG3Map(LABELS[classIDs[i]])
                # cv2.imwrite(str(count) +".jpg", image)
            else:
                objectCache[LABELS[classIDs[i]]] += 1
                if (objectCache[LABELS[classIDs[i]]] == 5):
                    updateG3 = True
            if (updateG3):
                # UPDATE PRBABILITY OF THE OBJECTS IN OUR ORIGINAL G3 MAP
                updateG3Map(LABELS[classIDs[i]])
                cv2.imwrite(str(count) +".jpg", image)
                
    return image

def buildG3Graph(text, actionlist):
    global lambdaList, probabilityDic, root, totalObjects, totalFoundObjects
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    lambda_current = ""
    lstNodes = []
    root = Node("")
    lstNodes.append(root)
    tempVerbNode = None
    for token in doc:
        # THE/A/YELLOW/etc.
        if token.pos_ == "DET" or token.pos_ == "ADJ":
            if len(lambda_current) == 0:
                lambda_current = token.text
            else:
                lambda_current += " " + token.text
        elif token.pos_ == "VERB":
            lambdaList.append(token.text)
            if len(lambda_current) == 0:
                lambda_current = token.text + "(EVENT)"
            else:
                lambda_current += " " + token.text + "(EVENT)"
            probability = 0
            totalObjects + 1
            if (token.text.lower() in actionlist):
                probability = 1
                totalFoundObjects += 1
            tempNode = Node(lambda_current, parent=lstNodes[-1])
            tempVerbNode = tempNode
            probabilityDic[lambda_current] = probability
            lstNodes.append(tempNode)
            lambda_current = ""
        elif token.pos_ == "NOUN":
            lambdaList.append(token.text)
            totalObjects + 1
            if len(lambda_current) == 0:
                lambda_current = token.text + "(OBJECT)"
            else:
                lambda_current += " " + token.text + "(OBJECT)"
            tempNode = Node(lambda_current, parent=lstNodes[-1])
            probabilityDic[lambda_current] = 0
            lstNodes.append(tempNode)
            lambda_current = ""
        elif token.pos_ == "ADP":
            # ADP SHOWED UP, NO NEED TO KEEP TRACK OF THE ORIGINAL VERB WORD
            totalObjects + 1
            lambdaList.append(token.text)
            if len(lambda_current) == 0:
                lambda_current = token.text + "(PLACES OR PATHS)"
            else:
                lambda_current += " " + token.text + "(PLACES OR PATHS)"
            if tempVerbNode == None:
                tempNode = Node(lambda_current, parent=lstNodes[-1])
            else:
                tempNode = Node(lambda_current, parent=tempVerbNode)
            probabilityDic[lambda_current] = 1
            totalFoundObjects += 1
            lstNodes.append(tempNode)
            lambda_current = ""
            tempVerbNode = None

# UPDATE PROBABILITY
def updateG3Map(lambdaDetected):
    global probabilityDic, root, totalFoundObjects
    # for pre, fill, node in RenderTree(root):
    updated = False
    for node in PreOrderIter(root):
        # print()
        # OBJECT
        if (lambdaDetected in node.name and probabilityDic[node.name] == 0):
            probabilityDic[node.name] = 1
            updated = True
            showTreeGraph(root)
    if updated:
        answer = 1
        totalFoundObjects += 1
        for i in probabilityDic:
            answer = answer*probabilityDic[i]
        if (answer == 1):
            print("We know that given command can be executed!")


def captureVideo(src):
    cap = cv2.VideoCapture(src)
    if cap.isOpened() and src=='0':
        ret = cap.set(3,640) and cap.set(4,480)
        if ret==False:
            print( 'Cannot set frame properties, returning' )
            return
    else:
        frate = cap.get(cv2.CAP_PROP_FPS)
        print( frate, ' is the framerate' )
        # waitTime = int( 1000/frate )
        waitTime = 3
        print(1)

    if src == 0:
        waitTime = 1
    if cap:
        print( 'Succesfully set up capture device' ) 
    else:
        print( 'Failed to setup capture device' ) 

    windowName = 'Camera Window'
    cv2.namedWindow(windowName)
    timeCount = 0
    results = []
    while(True):
        # Capture frame-by-frame
        ret, image = cap.read()
        if ret==False:
            break

        # YOLO DETECT OBJECTS
        image = findObjectsThroughYolo(image, timeCount)
        # print(itemlist)
        cv2.imshow(windowName, image )                                        
        inputKey = cv2.waitKey(waitTime) & 0xFF
        timeCount += 1
        results.append([timeCount, resultProbability, totalFoundObjects, totalObjects])
        if inputKey == ord('q'):
            break   

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    with open('data.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(results)

def showTreeGraph(root):
    global resultProbability
    probability = 1
    for pre, fill, node in RenderTree(root):
        print("%s%s" % (pre, node.name), end = ' ')
        if len(node.name) > 0:
            print("Corresponding Variance Phi: ", probabilityDic[node.name])
            probability *= int(probabilityDic[node.name])
            resultProbability = probability
    print("Probability of language command: ", probability)

if __name__ == '__main__':
    arglist = sys.argv
    src = 0
    command = ""
    print( 'Argument count is ', len(arglist) ) 
    if len(arglist) == 3:
        # VIDEO
        src = arglist[1]
        # COMMAND
        command = arglist[2]
        # ROBOT ACTION LIST
        actionList = ["put", "go", "pick", "drop"]
        buildG3Graph(command, actionList)
        print("Semantic Language G3 Map - Before Sensing")
        showTreeGraph(root)
        captureVideo(src)
    else:
        src = 0
        print('No video source input or No command input!')
