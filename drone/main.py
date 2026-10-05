import cv2
import time
import threading
import socket
import json
from ultralytics import RTDETR
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
model_path = ROOT / "models" / "fire_rtdetr.engine"

# --- CONFIGURAÇÕES DO WILDBRIDGE ---
# Substitua pelo IP real do seu controle/dispositivo onde o WildBridge roda
WILDBRIDGE_IP = "192.168.1.100" 
RTSP_PORT = "8554"
TELEMETRY_PORT = 8081  # Porta padrão do socket de telemetria do WildBridge

# URL do RTSP do MediaMTX integrado ao WildBridge
WILDBRIDGE_RTSP_URL = f"rtsp://{WILDBRIDGE_IP}:{RTSP_PORT}/streaming/live/1"

CLASS_NAMES = ["fire", "smoke"]
FIRE_ID = 0
SMOKE_ID = 1

COLORS = {
    0: (0, 0, 255),    # Fogo (Vermelho)
    1: (255, 0, 0)     # Fumaça (Azul)
}

# Variáveis globais protegidas para telemetria
telemetry_data = {
    "altitude": 0.0,
    "speed": 0.0
}
telemetry_lock = threading.Lock()

def telemetry_socket_loop():
    """Thread em segundo plano que consome o socket TCP contínuo do WildBridge."""
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((WILDBRIDGE_IP, TELEMETRY_PORT))
            buffer = ""
            
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk.decode('utf-8', errors='ignore')
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            t = json.loads(line)
                            # Extrai altitude e velocidade (ajuste as chaves se necessário conforme o JSON do WildBridge)
                            alt = t.get('location', {}).get('altitude', 0.0)
                            spd = t.get('speed', 0.0) # Ou a chave correspondente à velocidade no JSON
                            
                            with telemetry_lock:
                                telemetry_data["altitude"] = float(alt)
                                telemetry_data["speed"] = float(spd)
                        except json.JSONDecodeError:
                            pass
        except Exception:
            # Se cair a conexão, tenta reconectar após 2 segundos
            time.sleep(2.0)

def filter_by_priority(boxes, scores, class_ids, fire_id=0, smoke_id=1):
    """Mantém apenas detecções de fogo se houverem, caso contrário mantém fumaça."""
    if len(class_ids) == 0:
        return boxes, scores, class_ids
    fire_mask = (class_ids == fire_id)
    if np.any(fire_mask):
        return boxes[fire_mask], scores[fire_mask], class_ids[fire_mask]
    smoke_mask = (class_ids == smoke_id)
    return boxes[smoke_mask], scores[smoke_mask], class_ids[smoke_mask]

def draw_detections(frame, boxes, scores, class_ids):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = [int(v) for v in box]
        cls_id = int(cls_id)
        
        color = COLORS.get(cls_id, (0, 255, 0))
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        label = f"{name}: {score:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def main():
    # 1. Inicia a thread de telemetria via Socket em background
    t_telemetry = threading.Thread(target=telemetry_socket_loop, daemon=True)
    t_telemetry.start()

    # 2. Carrega o modelo RT-DETR
    print(f"Carregando modelo de: {model_path}")
    model = RTDETR(str(model_path))
    
    # 3. Conecta ao stream RTSP do WildBridge
    print(f"Conectando ao stream RTSP em {WILDBRIDGE_RTSP_URL}...")
    cap = cv2.VideoCapture(WILDBRIDGE_RTSP_URL)
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir o stream RTSP do WildBridge.")
        return

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao receber o frame ou o stream terminou.")
            break
            
        # Cálculo de FPS real
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
        prev_time = current_time

        # Pega dados de telemetria atuais de forma segura
        with telemetry_lock:
            current_alt = telemetry_data["altitude"]
            current_spd = telemetry_data["speed"]

        # Inferência do RT-DETR
        results = model.predict(frame, conf=0.5, verbose=False)
        
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
        scores = results[0].boxes.conf.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
        class_ids = results[0].boxes.cls.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])

        # Aplica o filtro de prioridade (Fogo > Fumaça)
        if len(class_ids) > 0:
            boxes, scores, class_ids = filter_by_priority(boxes, scores, class_ids, FIRE_ID, SMOKE_ID)
            frame = draw_detections(frame, boxes, scores, class_ids)

        # --- EXIBIÇÃO DA TELEMETRIA NO EXTREMO SUPERIOR DIREITO (PROPORCIONAL) ---
        h, w, _ = frame.shape
        text_lines = [
            f"ALT: {current_alt:.1f}m",
            f"SPD: {current_spd:.1f}m/s",
            f"FPS: {fps:.1f}"
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, h / 1800.0) 
        thickness = max(1, int(font_scale * 2))
        color = (255, 255, 255)  # Branco
        margin_right = int(w * 0.015)
        start_y = int(h * 0.04)
        line_spacing = int(h * 0.035)

        for i, text in enumerate(text_lines):
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = w - text_w - margin_right
            y = start_y + (i * line_spacing)
            
            # Sombra preta para legibilidade e texto principal em branco
            cv2.putText(frame, text, (x + 1, y + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Mostra o resultado final
        cv2.imshow("Mavic3M - RT-DETR + WildBridge", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()