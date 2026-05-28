import cv2
import mediapipe as mp
import numpy as np #pa los angulos bb

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calc_angulo(a,b,c):
    #vamo a calcular el angulo B que esta entre un punto A y C conocidos
    #cada punto es un array de coordenada [x,y]

    a = np.array(a) 
    b = np.array(b) #aqui es donde claculamos el angulp
    c = np.array(c)

    ba = a-b #vector b a a
    bc = c - b #vector b a c

    #arctan2 nos da el angylo de cada vector en funcion al eje x
    angulo = np.degrees(
        np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
    )

    #pa no tener angulo negatuvo
    angulo = abs(angulo)
    #pa siempre etar enrre los 0-180 grados
    if angulo>180:
        angulo = 360 - angulo
    
    return angulo

def get_landmark_coords(landmarks, index):
    landmark = landmarks[index]
    return [landmark.x, landmark.y]

#contador de reps
reps = 0
state = 'up'


#abrir camara
cap = cv2.VideoCapture(0)

#detector de forma
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        #ret es True si se abre la camara y FRAME es la imaen
        ret, frame = cap.read()

        if not ret:
            print("NO se consiguio una camara, verifica si esta coectada")
            break

        #mediapipe trbaja en rgb y cv en bgr, conversion:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        #ahora si la deteccion
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        #AJA, el skeleton pue
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            #los puntos q necesitamos pa squta
            hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value)
            knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value)
            ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value)
            shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)

            #calculamos los angulos
            knee_angle = calc_angulo(hip, knee, ankle)
            hip_angle = calc_angulo(shoulder, hip, knee)

            #pa las reps
            if knee_angle>160:
                state='up'


            if knee_angle<90 and state=='up':
                state='down'

            if knee_angle>150 and state=='down':
                state='up'
                reps+=1

            #aca va el poas 4, las alertas de si ta bien o ta mal
            warnings = [] #array dodne guardmos las advertencias
            if state == 'down': #solo chequea si la persona ya esta posicion baja
                if knee_angle > 100:
                    warnings.append("Baja mas")
                if hip_angle < 80:
                    warnings.append("Espalda recta")

            #mostramos los angulos
            h, w = frame.shape[:2]

            #contador de reps
            cv2.rectangle(image_rgb,(0,0),(240,80),(40,40,40),-1)

            cv2.putText(image_rgb, 'REPS',(10,25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(150,150,150),1
                        )
            cv2.putText(image_rgb,str(reps),(10,65),
                        cv2.FONT_HERSHEY_SIMPLEX,1.8,(255,255,255),3
                        )
            
            #mostamos estado
            cv2.putText(image_rgb, "ESTADO", (120,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,(150,150,150),1
                        )
            cv2.putText(image_rgb,state.upper(),(120,55),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,250,0),2)

            knee_px = (int(knee[0]*w),int(knee[1]*h))
            hip_px = (int(hip[0]*w),int(hip[1]*h))

            cv2.putText(image_rgb, f"{int(knee_angle)} grados",
                        knee_px,
                        cv2.FONT_HERSHEY_SIMPLEX,0.7, (255,255,0),2
                        )
            
            cv2.putText(image_rgb, f"{int(hip_angle)} grados",
                        hip_px,
                        cv2.FONT_HERSHEY_SIMPLEX,0.7, (255,255,0),2
                        )
            
            #alertas... o advertencias, como prefieras, dunno
            y_warn = 120 #aja, altura vertical

            #loop para mostrar todas las advertencias
            for warning in warnings:
                cv2.rectangle(image_rgb, (0, y_warn - 25), (300, y_warn + 10), (0, 0, 180), -1)#mostranis el texto en un resctagylo rojo
                cv2.putText(image_rgb, warning, (10, y_warn),#mostramos el texto
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_warn += 45 #movemos la siguiente advertencia pa bajo


            mp_drawing.draw_landmarks(
                image_rgb, #la imagen donde c va a draw
                results.pose_landmarks, #los 33 puntos del cuerpo
                mp_pose.POSE_CONNECTIONS #los huesitos jiji
            )
        


        #pa verlo
        cv2.imshow('IA COACH', image_rgb)
        frame_resized = cv2.resize(image_rgb, (1280, 720))
        cv2.imshow('IA COACH', frame_resized)

        #pa salir
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
