import face_recognition
import cv2
from PIL import Image, ImageDraw
import numpy as np

def reconhece_pessoas(frame):
  face_locations = face_recognition.face_locations(frame)
  return face_locations

def cria_conhecidos():
  for i in range(1,6):
    image_of_michel = face_recognition.load_image_file('michel'+str(i)+'.jpg')
    michel_face_encoding = []
    michel_face_encoding.append(face_recognition.face_encodings(image_of_michel)[0]) 

    image_of_janet = face_recognition.load_image_file('janet'+str(i)+'.jpg')
    janet_face_encoding = []
    janet_face_encoding.append(face_recognition.face_encodings(image_of_janet)[0]) 

    image_of_claire = face_recognition.load_image_file('claire'+str(i)+'.jpg')
    claire_face_encoding = []
    claire_face_encoding.append(face_recognition.face_encodings(image_of_claire)[0]) 

    image_of_junior = face_recognition.load_image_file('junior'+str(i)+'.jpg')
    junior_face_encoding = []
    junior_face_encoding.append(face_recognition.face_encodings(image_of_junior)[0]) 


    image_of_kady = face_recognition.load_image_file('kady'+str(i)+'.jpg')
    kady_face_encoding = []
    kady_face_encoding.append(face_recognition.face_encodings(image_of_kady)[0]) 

  #  Create arrays of encodings and names
  known_face_encodings = [
    michel_face_encoding,
    janet_face_encoding,
    claire_face_encoding,
    junior_face_encoding,
    kady_face_encoding
  ]

  known_face_names = [
    "Michel",
    "Janet",
    "Claire",
    "Junior",
    "Kady"
  ]

  return known_face_encodings,known_face_names

def marca_pessoas(frame,face_locations,known_face_encodings,known_face_names):
  
  # Find faces in test image
  face_locations = face_recognition.face_locations(frame)
  face_encodings = face_recognition.face_encodings(frame, face_locations)

  # Convert to PIL format
  pil_image = Image.fromarray(frame)

  # Create a ImageDraw instance
  draw = ImageDraw.Draw(pil_image)

  # Loop through faces in test image
  for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown Person"
    matches = np.array(matches)
    # If match
    if matches.any():
      first_match_index = np.where(matches==True,)[0][0]
      name = known_face_names[first_match_index]
  
    # Draw box
    draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))

    # Draw label
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))

  del draw

  return pil_image

def main():
  known_face_encodings,known_face_names = cria_conhecidos()
  EPC = cv2.VideoCapture('EPC-Pai-do-Ano.mp4')
  if (EPC.isOpened()== False): 
    print("Erro abertura da camera")
  i = 0
  while(True):
    # Take each frame
    ret , frame = EPC.read()
    frame_width = int(EPC.get(3))
    frame_height = int(EPC.get(4))
    cv2.imshow("teste",frame)
    vid_writer = cv2.VideoWriter('saida.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (frame_width,frame_height))
    if ret:
      frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)        
      face_locations = reconhece_pessoas(frame)
      pil_image = marca_pessoas(frame,face_locations,known_face_encodings,known_face_names)
      # pil_image.save(str(i)+".jpg")
      res = np.array(pil_image)
      vid_writer.write(res)
      cv2.waitKey(30)
    else:
      break
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
      EPC.release()
      break
    i += 1
    if 7*i < int(EPC.get(cv2.CAP_PROP_FRAME_COUNT)):
      EPC.set(cv2.CAP_PROP_POS_FRAMES, 7*i)
    else:
      EPC.set(cv2.CAP_PROP_POS_FRAMES, int(EPC.get(cv2.CAP_PROP_FRAME_COUNT)))
    print('Position:', int(EPC.get(cv2.CAP_PROP_POS_FRAMES)))
  vid_writer.release()
  cv2.destroyAllWindows()

main()