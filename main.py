import cv2
import matplotlib.pyplot as plt
import numpy as np

model = cv2.face.EigenFaceRecognizer_create()
#model = cv2.face.LBPHFaceRecognizer_create()

face_db = [
    #Angelina
    ["db/img1.jpg", "db/img2.jpg", "db/img4.jpg", "db/img5.jpg", "db/img6.jpg", "db/img7.jpg", "db/img10.jpg", "db/img11.jpg"],
    #Jennifer
    ["db/img3.jpg", "db/img12.jpg", "db/img53.jpg","db/img54.jpg","db/img55.jpg","db/img56.jpg"],
    #Scarlet
    ["db/img8.jpg", "db/img9.jpg", "db/img47.jpg", "db/img48.jpg", "db/img49.jpg", "db/img50.jpg", "db/img51.jpg"],
    # Mark
    ["db/img13.jpg", "db/img14.jpg", "db/img15.jpg", "db/img57.jpg", "db/img58.jpg", "db/img68.jpg", "db/img69.jpg", "db/img70.jpg"],
    ["db/img30.jpg"],
    ["db/img44.jpg"]
]

''' Cria o identificador de faces'''
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

''' Detecta face na imagem, e faz o corte para ampliar somente o rosto'''
def detect_face(img_path):
    img = cv2.imread(img_path)
    detect_faces = detector.detectMultiScale(img, 1.3, 5)
    x, y, w, h = detect_faces[0]

    img = img[y:y+h, x:x+w] #focus on the detected area
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img

# TREINAMENTO 
###############################################################################
faces = []; ids = []
index = 0
for img_path in face_db:
    for item in img_path:
        detected_face = detect_face(item)
        faces.append(detected_face)
        ids.append(index)
    
    index = index + 1
    
ids = np.array(ids)

model.train(faces, ids)
model.save("my_model.yml")
################################################################################

# FIND FACES
###############################################################################
target_path = "target/mark.jpg"
target = detect_face(target_path)

idx, confidence = model.predict(target)
found_path = face_db[idx]
found = detect_face(found_path[0])

plt.imshow(target)
plt.show()
plt.imshow(found)
plt.show()
