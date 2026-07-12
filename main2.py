import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
from PIL import Image, ImageDraw, ImageFont, ImageFilter

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ─────────────────────────────────────────────
# CONEXIONES SOLO DEL CUERPO (sin cara)
# ─────────────────────────────────────────────
BODY_CONNECTIONS = frozenset([
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
])
BODY_LANDMARKS = set(range(11, 33))

# ─────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────
C = {
    'bg':      (8,   10,  20),
    'surface': (18,  22,  32),
    'border':  (40,  46,  66),
    'white':   (255, 255, 255),
    'muted':   (160, 170, 200),
    'accent':  (99,  179, 237),
    'green':   (72,  199, 142),
    'red':     (252, 100, 100),
    'yellow':  (250, 202,  80),
}

def bgr(rgb): return (rgb[2], rgb[1], rgb[0])

# ─────────────────────────────────────────────
# UTILIDADES PIL
# ─────────────────────────────────────────────

def cv2pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")

def pil2cv(pil):
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)

def get_font(size, bold=False):
    paths = [
        f"C:/Windows/Fonts/{'arialbd' if bold else 'arial'}.ttf",
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except:
            continue
    return ImageFont.load_default()

def rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    pil = cv2pil(img)
    draw = ImageDraw.Draw(pil)
    draw.rounded_rectangle([x1, y1, x2, y2], radius=r,
                           fill=color if thickness == -1 else None,
                           outline=color if thickness > 0 else None,
                           width=thickness if thickness > 0 else 0)
    img[:] = pil2cv(pil)

def draw_gradient_bar(img, x1, y1, x2, y2, pct, color_rgb):
    rounded_rect(img, x1, y1, x2, y2, 4, (30, 35, 50))
    fill_w = int((x2 - x1) * pct)
    if fill_w > 8:
        rounded_rect(img, x1, y1, x1 + fill_w, y2, 4, color_rgb)

# ─────────────────────────────────────────────
# VOZ
# ─────────────────────────────────────────────

class VoiceCoach:
    def __init__(self):
        self.lock        = threading.Lock()
        self.is_speaking = False
        self.last_said   = {}
        self.cooldown    = 5.0
        self._spanish_voice = self._find_spanish_voice()

    def _find_spanish_voice(self):
        try:
            e = pyttsx3.init()
            voices = e.getProperty('voices')
            for v in voices:
                if 'spanish' in v.name.lower() or 'es_' in v.id.lower():
                    e.stop()
                    return v.id
            e.stop()
        except:
            pass
        return None

    def say(self, messages):
        if self.is_speaking:
            return
        now    = time.time()
        to_say = [m for m in messages
                  if now - self.last_said.get(m, 0) >= self.cooldown]
        if not to_say:
            return
        for m in to_say:
            self.last_said[m] = now
        texto = ". ".join(to_say)

        def speak():
            with self.lock:
                self.is_speaking = True
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 148)
                    engine.setProperty('volume', 1.0)
                    if self._spanish_voice:
                        engine.setProperty('voice', self._spanish_voice)
                    engine.say(texto)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    print(f"[voz] error: {e}")
                finally:
                    self.is_speaking = False

        threading.Thread(target=speak, daemon=True).start()

# ─────────────────────────────────────────────
# ÁNGULOS
# ─────────────────────────────────────────────

def calc_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    ang = np.degrees(np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0]))
    ang = abs(ang)
    return 360 - ang if ang > 180 else ang

def get_coords(landmarks, punto):
    lm = landmarks[mp_pose.PoseLandmark[punto].value]
    return [lm.x, lm.y]

def get_visibility(landmarks, punto):
    return landmarks[mp_pose.PoseLandmark[punto].value].visibility

def en_zona_arriba(a, angulo_up,   m=15): return a > angulo_up   - m
def en_zona_abajo (a, angulo_down, m=15): return a < angulo_down + m

# ─────────────────────────────────────────────
# CLASE BASE EJERCICIO
# ─────────────────────────────────────────────

class Ejercicio:
    def __init__(self, nombre, codigo, descripcion, angulo_up, angulo_down, color_rgb):
        self.nombre      = nombre
        self.codigo      = codigo
        self.descripcion = descripcion
        self.angulo_up   = angulo_up
        self.angulo_down = angulo_down
        self.color_rgb   = color_rgb
        self.color_bgr   = bgr(color_rgb)
        self.reps        = 0
        self.state       = 'arriba'
        self.rep_flash   = 0

    def get_angulo_principal(self, landmarks): raise NotImplementedError
    def get_angulo_secundario(self, landmarks): return None
    def check_warnings(self, a, a2, state, arr, abj): return []

    def update(self, landmarks):
        a  = self.get_angulo_principal(landmarks)
        a2 = self.get_angulo_secundario(landmarks)
        if a > self.angulo_up:
            if self.state == 'abajo':
                self.reps += 1
                self.rep_flash = 18
            self.state = 'arriba'
        elif a < self.angulo_down:
            self.state = 'abajo'
        arr = en_zona_arriba(a, self.angulo_up)
        abj = en_zona_abajo (a, self.angulo_down)
        w   = self.check_warnings(a, a2, self.state, arr, abj)
        if self.rep_flash > 0:
            self.rep_flash -= 1
        return a, a2, w

    def reset(self):
        self.reps = 0
        self.state = 'arriba'
        self.rep_flash = 0

# ─────────────────────────────────────────────
# EJERCICIOS
# ─────────────────────────────────────────────

class Squat(Ejercicio):
    def __init__(self):
        super().__init__("Sentadilla","SQ","Rodilla hasta 90°",160,90,(99,179,237))
    def get_angulo_principal(self, lm):
        return calc_angulo(get_coords(lm,'LEFT_HIP'),get_coords(lm,'LEFT_KNEE'),get_coords(lm,'LEFT_ANKLE'))
    def get_angulo_secundario(self, lm):
        return calc_angulo(get_coords(lm,'LEFT_SHOULDER'),get_coords(lm,'LEFT_HIP'),get_coords(lm,'LEFT_KNEE'))
    def check_warnings(self, a, a2, state, arr, abj):
        w = []
        if abj and a > self.angulo_down + 10: w.append("Baja más!")
        if abj and a2 and a2 < 55:            w.append("Espalda muy inclinada!")
        return w

class Deadlift(Ejercicio):
    def __init__(self):
        super().__init__("Peso Muerto","DL","Bisagra de cadera, espalda recta",155,65,(250,202,80))
    def get_angulo_principal(self, lm):
        return calc_angulo(get_coords(lm,'LEFT_SHOULDER'),get_coords(lm,'LEFT_HIP'),get_coords(lm,'LEFT_KNEE'))
    def get_angulo_secundario(self, lm):
        return calc_angulo(get_coords(lm,'LEFT_HIP'),get_coords(lm,'LEFT_KNEE'),get_coords(lm,'LEFT_ANKLE'))
    def check_warnings(self, a, a2, state, arr, abj):
        w = []
        if abj and a2 and a2 < 120: w.append("No dobles tanto las rodillas!")
        if arr and a < 150:          w.append("Extiende bien la cadera!")
        return w

class Lunge(Ejercicio):
    def __init__(self):
        super().__init__("Zancada","LU","Rodilla delantera a 90°",160,90,(72,199,142))
    def get_angulo_principal(self, lm):
        return calc_angulo(get_coords(lm,'LEFT_HIP'),get_coords(lm,'LEFT_KNEE'),get_coords(lm,'LEFT_ANKLE'))
    def get_angulo_secundario(self, lm):
        return calc_angulo(get_coords(lm,'LEFT_SHOULDER'),get_coords(lm,'LEFT_HIP'),get_coords(lm,'LEFT_KNEE'))
    def check_warnings(self, a, a2, state, arr, abj):
        w = []
        if abj and a > self.angulo_down + 15: w.append("Baja más la rodilla!")
        if abj and a2 and a2 < 65:            w.append("Mantén el torso erguido!")
        return w

class PushUp(Ejercicio):
    def __init__(self):
        super().__init__("Flexión","PU","Vista frontal — cuerpo recto",155,85,(252,100,100))
    def _lado(self, lm):
        return 'LEFT' if get_visibility(lm,'LEFT_ELBOW') >= get_visibility(lm,'RIGHT_ELBOW') else 'RIGHT'
    def get_angulo_principal(self, lm):
        l = self._lado(lm)
        return calc_angulo(get_coords(lm,f'{l}_SHOULDER'),get_coords(lm,f'{l}_ELBOW'),get_coords(lm,f'{l}_WRIST'))
    def get_angulo_secundario(self, lm):
        ls = get_coords(lm,'LEFT_SHOULDER');  rs = get_coords(lm,'RIGHT_SHOULDER')
        lh = get_coords(lm,'LEFT_HIP');       rh = get_coords(lm,'RIGHT_HIP')
        return abs((ls[1]+rs[1])/2 - (lh[1]+rh[1])/2) * 100
    def check_warnings(self, a, a2, state, arr, abj):
        w = []
        if abj and a > self.angulo_down + 15:        w.append("Baja más el pecho!")
        if state == 'abajo' and a2 and a2 > 45:      w.append("Cuerpo recto!")
        return w

class BicepCurl(Ejercicio):
    def __init__(self):
        super().__init__("Curl de Bícep","BC","Codo dobla hasta 40°",155,50,(192,132,252))
    def get_angulo_principal(self, lm):
        return calc_angulo(get_coords(lm,'LEFT_SHOULDER'),get_coords(lm,'LEFT_ELBOW'),get_coords(lm,'LEFT_WRIST'))
    def check_warnings(self, a, a2, state, arr, abj):
        w = []
        if arr and a < 145: w.append("Extiende el brazo completamente!")
        if abj and a > 65:  w.append("Sube más el peso!")
        return w

class ShoulderPress(Ejercicio):
    def __init__(self):
        super().__init__("Press Hombro","SP","Brazos arriba, espalda recta",160,85,(251,146,60))
    def get_angulo_principal(self, lm):
        return calc_angulo(get_coords(lm,'LEFT_SHOULDER'),get_coords(lm,'LEFT_ELBOW'),get_coords(lm,'LEFT_WRIST'))
    def get_angulo_secundario(self, lm):
        return calc_angulo(get_coords(lm,'LEFT_SHOULDER'),get_coords(lm,'LEFT_HIP'),get_coords(lm,'RIGHT_HIP'))
    def check_warnings(self, a, a2, state, arr, abj):
        w = []
        if arr and a < 155:  w.append("Extiende bien los brazos!")
        if a2 and a2 > 120:  w.append("No arquees la espalda!")
        return w

# ─────────────────────────────────────────────
# GLASSMORPHISM — FONDO
# ─────────────────────────────────────────────

def build_background(w, h):
    bg      = Image.new("RGB", (w, h), (8, 10, 20))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    blobs   = [
        ((w * 0.15, h * 0.20), 180, (60,  80, 180, 55)),
        ((w * 0.85, h * 0.15), 160, (80,  40, 160, 45)),
        ((w * 0.75, h * 0.80), 200, (30,  90, 180, 50)),
        ((w * 0.20, h * 0.75), 150, (60, 150, 120, 40)),
        ((w * 0.50, h * 0.45), 220, (40,  60, 140, 30)),
    ]
    for (cx, cy), r, color in blobs:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=90))
    bg.paste(overlay, mask=overlay.split()[3])
    draw2 = ImageDraw.Draw(bg)
    draw2.rectangle([0, 0, w, 2], fill=(99, 179, 237))
    return np.array(bg)[:, :, ::-1].copy()

# ─────────────────────────────────────────────
# GLASSMORPHISM — CARD
# ─────────────────────────────────────────────

def build_card_glass(card_w, card_h, color_rgb, hv):
    img  = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    glass_alpha  = int(22 + hv * 20)
    draw.rounded_rectangle([0, 0, card_w - 1, card_h - 1],
                           radius=16, fill=(255, 255, 255, glass_alpha))
    border_alpha = int(60 + hv * 130)
    border_color = (
        int(color_rgb[0] * hv + 255 * (1 - hv)),
        int(color_rgb[1] * hv + 255 * (1 - hv)),
        int(color_rgb[2] * hv + 255 * (1 - hv)),
        border_alpha,
    )
    draw.rounded_rectangle([0, 0, card_w - 1, card_h - 1],
                           radius=16, outline=border_color,
                           width=int(1 + hv * 1))
    draw.rounded_rectangle([1, 1, card_w - 2, card_h // 3],
                           radius=14,
                           fill=(255, 255, 255, int(8 + hv * 18)))
    bar_alpha = int(180 + hv * 75)
    draw.rounded_rectangle([2, 2, card_w - 3, 7],
                           radius=4, fill=(*color_rgb, bar_alpha))
    return img

def blend_card(canvas_bgr, card_pil_rgba, x, y):
    cw, ch   = card_pil_rgba.size
    h_c, w_c = canvas_bgr.shape[:2]
    x2 = min(x + cw, w_c);  y2 = min(y + ch, h_c)
    cw_c = x2 - x;          ch_c = y2 - y
    if cw_c <= 0 or ch_c <= 0:
        return
    roi      = canvas_bgr[y:y2, x:x2].copy()
    roi_blur = cv2.GaussianBlur(roi, (21, 21), 0)
    roi_pil  = Image.fromarray(cv2.cvtColor(roi_blur, cv2.COLOR_BGR2RGB)).convert("RGBA")
    card_clip = card_pil_rgba.crop((0, 0, cw_c, ch_c))
    base = Image.new("RGBA", (cw_c, ch_c), (0, 0, 0, 0))
    base.paste(roi_pil, (0, 0))
    base.paste(card_clip, (0, 0), mask=card_clip.split()[3])
    canvas_bgr[y:y2, x:x2] = cv2.cvtColor(np.array(base.convert("RGB")), cv2.COLOR_RGB2BGR)

# ─────────────────────────────────────────────
# ANIMACIÓN HOVER
# ─────────────────────────────────────────────

class HoverAnim:
    def __init__(self, n):
        self.vals  = [0.0] * n
        self.speed = 0.14

    def update(self, hover_idx):
        for i in range(len(self.vals)):
            target       = 1.0 if i == hover_idx else 0.0
            self.vals[i] += (target - self.vals[i]) * self.speed
        return self.vals

# ─────────────────────────────────────────────
# DRAW MENÚ GLASSMORPHISM
# ─────────────────────────────────────────────

def draw_menu_glass(canvas, bg_cache, ejercicios, hover_vals):
    h, w = canvas.shape[:2]
    canvas[:] = bg_cache

    # Título
    pil  = cv2pil(canvas)
    draw = ImageDraw.Draw(pil)
    f_big  = get_font(58, bold=True)
    f_sub  = get_font(16)
    f_hint = get_font(13)

    draw.text((w // 2 + 2, 50), "KINEXIS",
              fill=(0, 0, 0, 100), font=f_big, anchor="mm")
    draw.text((w // 2,     48), "KINEXIS",
              fill=(255, 255, 255, 255), font=f_big, anchor="mm")
    draw.text((w // 2, 90),
              "Selecciona un ejercicio para comenzar",
              fill=(160, 170, 200, 200), font=f_sub, anchor="mm")

    # Línea separadora
    for xi in range(80, w - 80):
        alpha = int(80 * np.sin(np.pi * (xi - 80) / (w - 160)))
        draw.point((xi, 110), fill=(99, 179, 237, alpha))

    canvas[:] = pil2cv(pil)

    # Grid
    cols, card_w, card_h = 3, 270, 168
    gap_x, gap_y         = 48, 28
    total_w = cols * card_w + (cols - 1) * gap_x
    sx = (w - total_w) // 2
    sy = 126

    rects = []
    for i, ej in enumerate(ejercicios):
        row, col = i // cols, i % cols
        x1 = sx + col * (card_w + gap_x)
        y1 = sy + row * (card_h + gap_y)
        rects.append((x1, y1, x1 + card_w, y1 + card_h))

        hv    = hover_vals[i]
        color = ej.color_rgb

        # Blur + glass
        blend_card(canvas, build_card_glass(card_w, card_h, color, hv), x1, y1)

        # Glow exterior
        if hv > 0.05:
            glow = canvas.copy()
            cv2.rectangle(glow,
                          (x1 - 4, y1 - 4),
                          (x1 + card_w + 4, y1 + card_h + 4),
                          bgr(color), 3)
            cv2.GaussianBlur(glow, (0, 0), 8, dst=glow)
            cv2.addWeighted(glow, hv * 0.35, canvas, 1 - hv * 0.35, 0, canvas)

        # Texto
        pil2 = cv2pil(canvas)
        d2   = ImageDraw.Draw(pil2)
        f_code = get_font(int(38 + hv * 5), bold=True)
        f_name = get_font(int(15 + hv * 2), bold=True)
        f_desc = get_font(12)
        f_cta  = get_font(12)

        bright     = int(hv * 40)
        code_color = tuple(min(255, c + bright) for c in color)
        d2.text((x1 + 20, y1 + 52), ej.codigo,
                fill=(*code_color, 255), font=f_code)
        d2.text((x1 + 20, y1 + 102), ej.nombre,
                fill=(255, 255, 255, 255), font=f_name)
        d2.text((x1 + 20, y1 + 124), ej.descripcion,
                fill=(160, 170, 200, 200), font=f_desc)
        if hv > 0.25:
            arrow_x = int(x1 + 20 + hv * 8)
            d2.text((arrow_x, y1 + 146), "Empezar →",
                    fill=(*color, int(hv * 255)), font=f_cta)
        canvas[:] = pil2cv(pil2)

    # Pie
    pil3 = cv2pil(canvas)
    d3   = ImageDraw.Draw(pil3)
    d3.text((w // 2, h - 20), "Q — salir",
            fill=(80, 90, 120, 180), font=f_hint, anchor="mm")
    canvas[:] = pil2cv(pil3)

    return rects

# ─────────────────────────────────────────────
# DRAW VISTA EJERCICIO
# ─────────────────────────────────────────────

def draw_exercise_ui(canvas, cam_frame, ejercicio, angulo, angulo2, warnings):
    h_c, w_c = canvas.shape[:2]
    h_f, w_f = cam_frame.shape[:2]
    canvas[:] = bgr(C['bg'])

    scale  = min(w_c / w_f, h_c / h_f)
    new_w  = int(w_f * scale)
    new_h  = int(h_f * scale)
    off_x  = (w_c - new_w) // 2
    off_y  = (h_c - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = cv2.resize(cam_frame, (new_w, new_h))

    color_rgb = ejercicio.color_rgb
    color_bgr = ejercicio.color_bgr

    # Header semitransparente
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w_c, 56), bgr(C['bg']), -1)
    cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
    cv2.rectangle(canvas, (0, 53), (w_c, 56), color_bgr, -1)

    pil  = cv2pil(canvas)
    draw = ImageDraw.Draw(pil)
    f_title = get_font(22, bold=True)
    f_hint  = get_font(12)
    f_label = get_font(11)
    f_value = get_font(42, bold=True)
    f_state = get_font(18, bold=True)
    f_warn  = get_font(15, bold=True)
    f_pct   = get_font(11)

    draw.text((18, 16), ejercicio.nombre.upper(),
              fill=color_rgb, font=f_title)
    draw.text((w_c - 18, 20), "R reiniciar   B menú   Q salir",
              fill=C['muted'], font=f_hint, anchor="rm")
    canvas[:] = pil2cv(pil)

    # Panel REPS
    flash    = ejercicio.rep_flash > 0
    reps_bg  = color_rgb if flash else C['surface']
    reps_txt = C['bg']   if flash else C['white']
    rounded_rect(canvas, 0, 60, 148, 165, 12, reps_bg)
    pil = cv2pil(canvas); d = ImageDraw.Draw(pil)
    d.text((14, 72), "REPS",
           fill=C['bg'] if flash else C['muted'], font=f_label)
    d.text((14, 130), str(ejercicio.reps),
           fill=reps_txt, font=f_value, anchor="lm")
    canvas[:] = pil2cv(pil)

    # Panel ESTADO
    s_color = C['green'] if ejercicio.state == 'arriba' else color_rgb
    rounded_rect(canvas, 156, 60, 340, 165, 12, C['surface'])
    pil = cv2pil(canvas); d = ImageDraw.Draw(pil)
    d.text((170, 72), "ESTADO", fill=C['muted'], font=f_label)
    d.text((170, 128), ejercicio.state.upper(),
           fill=s_color, font=f_state, anchor="lm")
    canvas[:] = pil2cv(pil)

    # Panel ÁNGULO
    rounded_rect(canvas, 348, 60, 532, 165, 12, C['surface'])
    pil = cv2pil(canvas); d = ImageDraw.Draw(pil)
    d.text((362, 72), "ÁNGULO", fill=C['muted'], font=f_label)
    d.text((362, 128), f"{int(angulo)}°",
           fill=C['yellow'], font=f_state, anchor="lm")
    if angulo2 is not None:
        d.text((430, 128), f"/ {int(angulo2)}°",
               fill=C['muted'], font=f_label, anchor="lm")
    canvas[:] = pil2cv(pil)

    # Barra rango
    bar_y = h_c - 32
    pct   = float(np.clip(
        (angulo - ejercicio.angulo_down) /
        max(ejercicio.angulo_up - ejercicio.angulo_down, 1), 0, 1))
    draw_gradient_bar(canvas, 0, bar_y, w_c, bar_y + 20, pct, color_rgb)
    pil = cv2pil(canvas); d = ImageDraw.Draw(pil)
    d.text((8, h_c - 8), f"Rango de movimiento: {int(pct*100)}%",
           fill=C['muted'], font=f_pct)
    canvas[:] = pil2cv(pil)

    # Advertencias
    y_w = 185
    for msg in warnings:
        overlay = canvas.copy()
        rounded_rect(overlay, off_x, y_w - 26, off_x + 400, y_w + 16, 8, (40, 10, 10))
        cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
        cv2.circle(canvas, (off_x + 16, y_w - 4), 5, bgr(C['red']), -1)
        pil = cv2pil(canvas); d = ImageDraw.Draw(pil)
        d.text((off_x + 30, y_w - 10), msg, fill=C['white'], font=f_warn)
        canvas[:] = pil2cv(pil)
        y_w += 52

# ─────────────────────────────────────────────
# SKELETON SIN CARA
# ─────────────────────────────────────────────

def draw_body_landmarks(image, results):
    if not results.pose_landmarks:
        return
    lms = results.pose_landmarks.landmark
    h, w = image.shape[:2]
    for (a, b) in BODY_CONNECTIONS:
        la, lb = lms[a], lms[b]
        if la.visibility < 0.4 or lb.visibility < 0.4:
            continue
        pa = (int(la.x * w), int(la.y * h))
        pb = (int(lb.x * w), int(lb.y * h))
        cv2.line(image, pa, pb, bgr(C['accent']), 2, cv2.LINE_AA)
    for idx in BODY_LANDMARKS:
        lm = lms[idx]
        if lm.visibility < 0.4:
            continue
        px = (int(lm.x * w), int(lm.y * h))
        cv2.circle(image, px, 5, bgr(C['white']),  -1, cv2.LINE_AA)
        cv2.circle(image, px, 5, bgr(C['accent']),  1, cv2.LINE_AA)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

VENTANA_W, VENTANA_H = 1280, 720

EJERCICIOS = [
    Squat(), Deadlift(), Lunge(),
    PushUp(), BicepCurl(), ShoulderPress(),
]

screen           = 'menu'
ejercicio_actual = None
hover_idx        = -1
card_rects       = []

voice      = VoiceCoach()
canvas     = np.zeros((VENTANA_H, VENTANA_W, 3), dtype=np.uint8)
bg_cache   = build_background(VENTANA_W, VENTANA_H)
hover_anim = HoverAnim(len(EJERCICIOS))

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

cv2.namedWindow('KINEXIS', cv2.WINDOW_NORMAL)
cv2.resizeWindow('KINEXIS', VENTANA_W, VENTANA_H)
cv2.setMouseCallback('KINEXIS', on_mouse)

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

        if screen == 'menu':
            hv         = hover_anim.update(hover_idx)
            card_rects = draw_menu_glass(canvas, bg_cache, EJERCICIOS, hv)
            cv2.imshow('KINEXIS', canvas)

        elif screen == 'ejercicio':
            if key == ord('b'):
                screen = 'menu'; hover_idx = -1; continue
            if key == ord('r'):
                ejercicio_actual.reset()

            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
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
                draw_body_landmarks(img_bgr, results)

            draw_exercise_ui(canvas, img_bgr, ejercicio_actual,
                             angulo, angulo2, warnings)
            cv2.imshow('KINEXIS', canvas)

cap.release()
cv2.destroyAllWindows()