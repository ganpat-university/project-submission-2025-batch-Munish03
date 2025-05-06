from ultralytics import YOLO
from playsound import playsound
import time
import os
import cv2
import smtplib
from email.message import EmailMessage
import firebase_admin
from firebase_admin import credentials, firestore, storage
from datetime import datetime
from twilio.rest import Client
# Initialize Firebase Admin SDK
cred = credentials.Certificate("firewatch-a23d0-firebase-adminsdk-fbsvc-9d0f6b02f4.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'firewatch-a23d0.firebasestorage.app'  # Updated bucket name
})
db = firestore.client()
bucket = storage.bucket()

account_sid = 'ACc455443567b7789e1e2da1b6bc5d7bb7'
auth_token = 'd229e0d1e017570b0aa8b748628f8606'
client = Client(account_sid, auth_token)

cap = cv2.VideoCapture(0)

output_dir = 'output_images'
os.makedirs(output_dir,exist_ok=True)
img_counter = 1

current_time = time.time()
run_counter=1
fire_counter=0
last_email_time = 0  # Track when the last email was sent

def email_alert(subject,body,to):
  msg = EmailMessage()
  msg.set_content(body)
  msg['subject'] = subject
  msg['to'] = 'munish03patwa@gmail.com'

  user = 'munishdpatwa21@gnu.ac.in'
  msg['from'] = user
  password = 'qqervjnnsrkbrujo'

  server = smtplib.SMTP("smtp.gmail.com",587)
  server.starttls()
  server.login(user,password)
  server.send_message(msg)
  server.quit()

while True:


    ret,frame = cap.read()

    cv2.imshow('Webcam',frame)

    k=cv2.waitKey(1)

    if k%256 == 27:
        print('Escape Hit')
        break

    elif time.time() > current_time + 5:
        current_time = time.time()
        img_name = os.path.join(output_dir,"opencv_frame_{}.png".format(img_counter))
        cv2.imwrite(img_name,frame)
        print("{} written".format(img_name))
        img_counter +=1
        run_counter=1

    else:
        if run_counter==1:
            img_detect = "output_images/opencv_frame_{}.png".format(img_counter-1)
            model = YOLO('best (2).pt')
            results = model.predict(source=img_detect,imgsz=650,conf=0.6, show=True)
            detected_classes = [result.names[0] for result in results for class_id in result.boxes.cls]

            # Upload image to Firebase Storage
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            blob = bucket.blob(f'detection_images/{timestamp}.png')
            blob.upload_from_filename(img_detect)
            # Make the image publicly accessible and get the URL
            blob.make_public()
            image_url = blob.public_url

            # Create log entry with image URL
            log_data = {
                'timestamp': datetime.now(),
                'fire_detected': 'fire' in detected_classes,
                'email': 'munishdpatwa21@gmail.com',
                'image_url': image_url
            }
            
            # Add log to Firestore
            db.collection('fire_detection_logs').add(log_data)

            if 'fire' in detected_classes:
                print("Fire detected!")
                fire_counter+=1
            else:
                fire_counter=0

            run_counter=0

        if fire_counter>=3:
            current_time = time.time()
            # Check if 5 minutes (300 seconds) have passed since last email
            if current_time - last_email_time >= 300:
                email_alert('FIRE IN THE HOUSE!!!!!','FIRE DETECTED CALL STATION','munish03patwa@gmail.com')
                last_email_time = current_time  # Update the last email time
                print("Email alert sent")
                message = client.messages.create(
                from_='+19702938179',
                body='FIRE IN YOUR APARTMENT',
                to='+919106956367'
                )

                print(message.sid)


cap.release()
cv2.destroyAllWindows()
   



message = client.messages.create(
  from_='+19702938179',
  body='FIRE IN YOUR APARTMENT',
  to='+919106956367'
)

