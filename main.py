import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ─────────────────────────────────────────────
# VOZ — corregida
# ─────────────────────────────────────────────

class VoiceCoach:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)

        voices = self.engine.getProperty('voices')
        for v in voices:
            if 'spanish' in v.name.lower() or 'es_' in v.id.lower():
                self.engine.setProperty('voice', v.id)
                break

        self.lock        = threading.Lock()
        self.is_speaking = False
        self.last_said   = {}     # {mensaje: timestamp cuando SE DIJO de verdad}
        self.cooldown    = 5.0

    def say(self, messages):
        if self.is_speaking:
            return

        now    = time.time()
        to_say = [m for m in messages
                  if now - self.last_said.get(m, 0) >= self.cooldown]

        if not to_say:
            return

        # Registrar timestamp SOLO de los que vamos a decir
        for m in to_say:
            self.last_said[m] = now

        texto = ". ".join(to_say)

        def speak():
            with self.lock:
                self.is_speaking = True
                self.engine.say(texto)
                self.engine.runAndWait()
                self.is_speaking = False

        threading.Thread(target=speak, daemon=True).start()


# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def calc_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    angulo = np.degrees(np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0]))
    angulo = abs(angulo)
    if angulo > 180:
        angulo = 360 - angulo
    return angulo

def get_coords(landmarks, punto):
    lm = landmarks[mp_pose.PoseLandmark[punto].value]
    return [lm.x, lm.y]

def get_visibility(landmarks, punto):
    return landmarks[mp_pose.PoseLandmark[punto].value].visibility

# Zona muerta entre estados — evita warnings mientras la persona transita
# Solo se evalúa forma si el ángulo está claramente en zona arriba o abajo
def en_zona_arriba(angulo, angulo_up, margen=15):
    return angulo > angulo_up - margen

def en_zona_abajo(angulo, angulo_down, margen=15):
    return angulo < angulo_down + margen


# ─────────────────────────────────────────────
# CLASE BASE
# ─────────────────────────────────────────────

class Ejercicio:
    def __init__(self, nombre, emoji, descripcion, angulo_up, angulo_down, color):
        self.nombre      = nombre
        self.emoji       = emoji
        self.descripcion = descripcion
        self.angulo_up   = angulo_up
        self.angulo_down = angulo_down
        self.color       = color
        self.reps        = 0
        self.state       = 'arriba'

    def get_angulo_principal(self, landmarks): raise NotImplementedError
    def get_angulo_secundario(self, landmarks): return None
    def check_warnings(self, angulo, angulo2, state, en_arriba, en_abajo): return []

    def update(self, landmarks):
        angulo  = self.get_angulo_principal(landmarks)
        angulo2 = self.get_angulo_secundario(landmarks)

        # Máquina de estados con zona intermedia
        if angulo > self.angulo_up:
            if self.state == 'abajo':
                self.reps += 1
            self.state = 'arriba'
        elif angulo < self.angulo_down:
            self.state = 'abajo'
        # zona intermedia: no cambia estado

        # Flags de zona para los warnings
        arriba = en_zona_arriba(angulo, self.angulo_up)
        abajo  = en_zona_abajo(angulo, self.angulo_down)

        warnings = self.check_warnings(angulo, angulo2, self.state, arriba, abajo)
        return angulo, angulo2, warnings

    def reset(self):
        self.reps  = 0
        self.state = 'arriba'


# ─────────────────────────────────────────────
# EJERCICIOS — warnings solo en zonas válidas
# ─────────────────────────────────────────────

class Squat(Ejercicio):
    def __init__(self):
        super().__init__("Sentadilla", "SQ",
                         "Rodilla hasta 90 grados",
                         160, 90, (86, 180, 233))

    def get_angulo_principal(self, landmarks):
        return calc_angulo(get_coords(landmarks, 'LEFT_HIP'),
                           get_coords(landmarks, 'LEFT_KNEE'),
                           get_coords(landmarks, 'LEFT_ANKLE'))

    def get_angulo_secundario(self, landmarks):
        # Ángulo de inclinación del torso: hombro→cadera→rodilla
        return calc_angulo(get_coords(landmarks, 'LEFT_SHOULDER'),
                           get_coords(landmarks, 'LEFT_HIP'),
                           get_coords(landmarks, 'LEFT_KNEE'))

    def check_warnings(self, a, a2, state, en_arriba, en_abajo):
        w = []
        # "baja más" solo cuando está claramente abajo y aún no llegó a 90
        if en_abajo and a > self.angulo_down + 10:
            w.append("Baja más!")
        # espalda solo cuando está en el fondo de la sentadilla
        if en_abajo and a2 is not None and a2 < 55:
            w.append("Espalda muy inclinada!")
        return w


class Deadlift(Ejercicio):
    def __init__(self):
        super().__init__("Peso Muerto", "DL",
                         "Bisagra de cadera, espalda recta",
                         155, 65, (230, 159, 0))

    def get_angulo_principal(self, landmarks):
        # Ángulo de la cadera: hombro→cadera→rodilla
        return calc_angulo(get_coords(landmarks, 'LEFT_SHOULDER'),
                           get_coords(landmarks, 'LEFT_HIP'),
                           get_coords(landmarks, 'LEFT_KNEE'))

    def get_angulo_secundario(self, landmarks):
        # Rodilla — no debe doblarse demasiado
        return calc_angulo(get_coords(landmarks, 'LEFT_HIP'),
                           get_coords(landmarks, 'LEFT_KNEE'),
                           get_coords(landmarks, 'LEFT_ANKLE'))

    def check_warnings(self, a, a2, state, en_arriba, en_abajo):
        w = []
        if en_abajo and a2 is not None and a2 < 120:
            w.append("No dobles tanto las rodillas!")
        if en_arriba and a < 150:
            w.append("Extiende bien la cadera!")
        return w


class Lunge(Ejercicio):
    def __init__(self):
        super().__init__("Zancada", "LU",
                         "Rodilla al frente a 90 grados",
                         160, 90, (0, 158, 115))

    def get_angulo_principal(self, landmarks):
        return calc_angulo(get_coords(landmarks, 'LEFT_HIP'),
                           get_coords(landmarks, 'LEFT_KNEE'),
                           get_coords(landmarks, 'LEFT_ANKLE'))

    def get_angulo_secundario(self, landmarks):
        # Torso vertical: hombro→cadera→rodilla
        return calc_angulo(get_coords(landmarks, 'LEFT_SHOULDER'),
                           get_coords(landmarks, 'LEFT_HIP'),
                           get_coords(landmarks, 'LEFT_KNEE'))

    def check_warnings(self, a, a2, state, en_arriba, en_abajo):
        w = []
        if en_abajo and a > self.angulo_down + 15:
            w.append("Baja más la rodilla!")
        if en_abajo and a2 is not None and a2 < 65:
            w.append("Mantén el torso erguido!")
        return w


class PushUp(Ejercicio):
    def __init__(self):
        super().__init__("Flexión", "PU",
                         "Vista frontal — cuerpo recto",
                         155, 85, (213, 94, 0))

    def _lado(self, landmarks):
        """Elige el lado más visible para el ángulo del codo"""
        vl = get_visibility(landmarks, 'LEFT_ELBOW')
        vr = get_visibility(landmarks, 'RIGHT_ELBOW')
        return 'LEFT' if vl >= vr else 'RIGHT'

    def get_angulo_principal(self, landmarks):
        lado = self._lado(landmarks)
        return calc_angulo(
            get_coords(landmarks, f'{lado}_SHOULDER'),
            get_coords(landmarks, f'{lado}_ELBOW'),
            get_coords(landmarks, f'{lado}_WRIST')
        )

    def get_angulo_secundario(self, landmarks):
        # Alineación del cuerpo desde el frente:
        # diferencia de altura normalizada entre hombros y caderas
        # En una flexión bien hecha esto debe ser pequeño (cuerpo horizontal recto)
        ls = get_coords(landmarks, 'LEFT_SHOULDER')
        rs = get_coords(landmarks, 'RIGHT_SHOULDER')
        lh = get_coords(landmarks, 'LEFT_HIP')
        rh = get_coords(landmarks, 'RIGHT_HIP')
        cy_hombros = (ls[1] + rs[1]) / 2
        cy_caderas = (lh[1] + rh[1]) / 2
        # Retornamos diferencia * 100 en escala 0-100
        return abs(cy_hombros - cy_caderas) * 100

    def check_warnings(self, a, a2, state, en_arriba, en_abajo):
        w = []
        if en_abajo and a > self.angulo_down + 15:
            w.append("Baja más el pecho!")
        # Alineación: solo alertar si la diferencia es muy grande (>45 puntos)
        # y estamos en movimiento (no solo parado quieto)
        if state == 'abajo' and a2 is not None and a2 > 45:
            w.append("Cuerpo recto, no eleves la cadera!")
        return w


class BicepCurl(Ejercicio):
    def __init__(self):
        super().__init__("Curl de Bícep", "BC",
                         "Codo dobla hasta 40 grados",
                         155, 50, (204, 121, 167))

    def get_angulo_principal(self, landmarks):
        return calc_angulo(get_coords(landmarks, 'LEFT_SHOULDER'),
                           get_coords(landmarks, 'LEFT_ELBOW'),
                           get_coords(landmarks, 'LEFT_WRIST'))

    def check_warnings(self, a, a2, state, en_arriba, en_abajo):
        w = []
        # Solo advertir si está claramente en la zona correspondiente
        if en_arriba and a < 145:
            w.append("Extiende el brazo completamente!")
        if en_abajo and a > 65:
            w.append("Sube más el peso!")
        return w


class ShoulderPress(Ejercicio):
    def __init__(self):
        super().__init__("Press de Hombro", "SP",
                         "Brazos arriba, espalda recta",
                         160, 85, (0, 114, 178))

    def get_angulo_principal(self, landmarks):
        return calc_angulo(get_coords(landmarks, 'LEFT_SHOULDER'),
                           get_coords(landmarks, 'LEFT_ELBOW'),
                           get_coords(landmarks, 'LEFT_WRIST'))

    def get_angulo_secundario(self, landmarks):
        # Inclinación lateral del torso: hombro izq → cadera izq → cadera der
        # Mide si la persona se está arqueando hacia atrás
        l_shoulder = get_coords(landmarks, 'LEFT_SHOULDER')
        l_hip      = get_coords(landmarks, 'LEFT_HIP')
        r_hip      = get_coords(landmarks, 'RIGHT_HIP')
        return calc_angulo(l_shoulder, l_hip, r_hip)

    def check_warnings(self, a, a2, state, en_arriba, en_abajo):
        w = []
        if en_arriba and a < 155:
            w.append("Extiende bien los brazos!")
        # Arqueo de espalda: hombro→cadera→cadera debería estar cerca de 90°
        # Si se aleja mucho de 90 hacia arriba (>120) es que se arquea atrás
        if a2 is not None and a2 > 120:
            w.append("No arquees la espalda!")
        return w


# ─────────────────────────────────────────────
# PANTALLA: MENÚ
# ─────────────────────────────────────────────

VENTANA_W = 1280
VENTANA_H = 720

def draw_menu(canvas, ejercicios, hover_idx):
    h, w = canvas.shape[:2]
    canvas[:] = (18, 18, 18)

    cv2.putText(canvas, "IA COACH", (w // 2 - 175, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 3)
    cv2.putText(canvas, "Selecciona un ejercicio", (w // 2 - 155, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 1)
    cv2.line(canvas, (80, 128), (w - 80, 128), (45, 45, 45), 1)

    cols   = 3
    card_w = 280
    card_h = 145
    gap_x  = 50
    gap_y  = 35
    total_w = cols * card_w + (cols - 1) * gap_x
    sx = (w - total_w) // 2
    sy = 155

    rects = []
    for i, ej in enumerate(ejercicios):
        row = i // cols
        col = i % cols
        x1  = sx + col * (card_w + gap_x)
        y1  = sy + row * (card_h + gap_y)
        x2  = x1 + card_w
        y2  = y1 + card_h
        rects.append((x1, y1, x2, y2))

        hover = (i == hover_idx)
        bg    = (38, 38, 38) if hover else (26, 26, 26)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        bcolor = ej.color if hover else (55, 55, 55)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bcolor, 2 if hover else 1)
        cv2.rectangle(canvas, (x1, y1), (x2, y1 + 6), ej.color, -1)

        cv2.putText(canvas, ej.emoji, (x1 + 18, y1 + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, ej.color, 2)
        cv2.putText(canvas, ej.nombre, (x1 + 18, y1 + 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1)
        cv2.putText(canvas, ej.descripcion, (x1 + 18, y1 + 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (110, 110, 110), 1)
        if hover:
            cv2.putText(canvas, "Clic para empezar  >", (x1 + 18, y1 + 134),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, ej.color, 1)

    cv2.putText(canvas, "Q: salir", (w // 2 - 30, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)
    return rects


# ─────────────────────────────────────────────
# PANTALLA: EJERCICIO
# ─────────────────────────────────────────────

def draw_exercise_ui(canvas, cam_frame, ejercicio, angulo, angulo2, warnings):
    h_c, w_c = canvas.shape[:2]
    h_f, w_f = cam_frame.shape[:2]
    canvas[:] = (10, 10, 10)

    # Cámara centrada con proporción
    scale  = min(w_c / w_f, h_c / h_f)
    new_w  = int(w_f * scale)
    new_h  = int(h_f * scale)
    off_x  = (w_c - new_w) // 2
    off_y  = (h_c - new_h) // 2
    resized = cv2.resize(cam_frame, (new_w, new_h))
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized

    color = ejercicio.color

    # Header
    cv2.rectangle(canvas, (0, 0), (w_c, 52), (12, 12, 12), -1)
    cv2.rectangle(canvas, (0, 49), (w_c, 52), color, -1)
    cv2.putText(canvas, ejercicio.nombre.upper(), (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
    cv2.putText(canvas, "R: reiniciar   B: volver al menu   Q: salir",
                (w_c - 420, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)

    # Panel REPS
    cv2.rectangle(canvas, (0, 57), (140, 155), (20, 20, 20), -1)
    cv2.rectangle(canvas, (0, 57), (140, 155), (40, 40, 40), 1)
    cv2.putText(canvas, "REPS", (12, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1)
    cv2.putText(canvas, str(ejercicio.reps), (12, 142),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 255, 255), 3)

    # Panel ESTADO
    s_color = (80, 210, 80) if ejercicio.state == 'arriba' else color
    cv2.rectangle(canvas, (145, 57), (310, 155), (20, 20, 20), -1)
    cv2.rectangle(canvas, (145, 57), (310, 155), (40, 40, 40), 1)
    cv2.putText(canvas, "ESTADO", (157, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1)
    cv2.putText(canvas, ejercicio.state.upper(), (157, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, s_color, 2)

    # Panel ANGULOS
    cv2.rectangle(canvas, (315, 57), (500, 155), (20, 20, 20), -1)
    cv2.rectangle(canvas, (315, 57), (500, 155), (40, 40, 40), 1)
    cv2.putText(canvas, "ANGULOS", (327, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1)
    cv2.putText(canvas, f"A1: {int(angulo)}", (327, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 1)
    if angulo2 is not None:
        cv2.putText(canvas, f"A2: {int(angulo2)}", (327, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 170, 0), 1)

    # Barra rango de movimiento
    bar_y = h_c - 28
    cv2.rectangle(canvas, (0, bar_y - 4), (w_c, h_c), (18, 18, 18), -1)
    pct  = float(np.clip(
        (angulo - ejercicio.angulo_down) / max(ejercicio.angulo_up - ejercicio.angulo_down, 1),
        0, 1))
    fill = int(w_c * pct)
    cv2.rectangle(canvas, (0, bar_y + 2), (fill, bar_y + 18), color, -1)
    cv2.putText(canvas, f"Rango de movimiento: {int(pct * 100)}%",
                (10, h_c - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)

    # Advertencias
    y_w = 175
    for msg in warnings:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (off_x, y_w - 24), (off_x + 390, y_w + 14),
                      (0, 0, 140), -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        cv2.circle(canvas, (off_x + 14, y_w - 4), 5, (0, 60, 255), -1)
        cv2.putText(canvas, msg, (off_x + 26, y_w),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)
        y_w += 48


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

EJERCICIOS = [
    Squat(), Deadlift(), Lunge(),
    PushUp(), BicepCurl(), ShoulderPress(),
]

screen           = 'menu'
ejercicio_actual = None
hover_idx        = -1
card_rects       = []

voice  = VoiceCoach()
canvas = np.zeros((VENTANA_H, VENTANA_W, 3), dtype=np.uint8)

def on_mouse(event, x, y, flags, param):
    global hover_idx, screen, ejercicio_actual
    if screen != 'menu':
        return
    hover_idx = -1
    for i, (x1, y1, x2, y2) in enumerate(card_rects):
        if x1 <= x <= x2 and y1 <= y <= y2:
            hover_idx = i
            break
    if event == cv2.EVENT_LBUTTONDOWN and hover_idx != -1:
        ejercicio_actual = EJERCICIOS[hover_idx]
        ejercicio_actual.reset()
        screen = 'ejercicio'

cv2.namedWindow('IA COACH', cv2.WINDOW_NORMAL)
cv2.resizeWindow('IA COACH', VENTANA_W, VENTANA_H)
cv2.setMouseCallback('IA COACH', on_mouse)

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

        if screen == 'menu':
            card_rects = draw_menu(canvas, EJERCICIOS, hover_idx)
            cv2.imshow('IA COACH', canvas)

        elif screen == 'ejercicio':
            if key == ord('b'):
                screen    = 'menu'
                hover_idx = -1
                continue
            if key == ord('r'):
                ejercicio_actual.reset()

            ret, frame = cap.read()
            if not ret:
                break

            # ── Voltear imagen horizontalmente ──
            frame = cv2.flip(frame, 1)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            results = pose.process(img_rgb)
            img_rgb.flags.writeable = True
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            angulo, angulo2, warnings = 0, None, []

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                angulo, angulo2, warnings = ejercicio_actual.update(landmarks)

                if warnings:
                    voice.say(warnings)

                mp_drawing.draw_landmarks(
                    img_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            draw_exercise_ui(canvas, img_bgr, ejercicio_actual,
                             angulo, angulo2, warnings)
            cv2.imshow('IA COACH', canvas)

cap.release()
cv2.destroyAllWindows()