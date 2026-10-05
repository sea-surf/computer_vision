# Setup Guide: Instalação e Integração do WildBridge no RC + Mavic 3M + RT-DETR

Este guia oficial detalha como compilar e instalar o WildBridge no controle remoto (RC) do drone, além de configurar a estação de solo em Python com OpenCV, telemetria via Socket TCP e inferência do modelo RT-DETR com foco em detecção de incêndios.

---

## 🛠️ 1. Compilação e Instalação do WildBridge no RC (Remote Controller)

O **RC** (Remote Controller) é o controle remoto do drone (como o DJI RC Pro Enterprise ou um dispositivo Android conectado). Como o WildBridge precisa rodar embarcado nele, você deve compilar o APK no seu PC e transferi-lo para o RC.

### Pré-requisitos de Instalação:
1. Drone DJI + RC compatível, ponto de acesso Wi-Fi de 5 GHz, computador de estação de solo.
2. Android Studio Koala (ou versão recente).
3. Conta de desenvolvedor DJI + Chave de API obtida em [developer.dji.com](https://developer.dji.com).

### Passos para Instalar o App:
1. Clone o repositório do WildBridge no seu PC:
   ```bash
   git clone https://github.com/WildDrone/WildBridge.git
   ```
2. Abra a pasta `WildBridge/WildBridgeApp/android-sdk-v5-as` no Android Studio.
3. Adicione a sua chave de API da DJI no arquivo `local.properties`:
   ```properties
   AIRCRAFT_API_KEY="Your_App_Key"
   ```
4. Conecte o seu RC (ou telefone Android) ao PC via USB (com o **Modo de Desenvolvedor + Depuração USB** ativados) e compile/implante diretamente pelo Android Studio.
5. Se preferir compilar via linha de comando (CLI):
   ```bash
   cd WildBridge/WildBridgeApp/android-sdk-v5-as
   ./gradlew :sample:assembleDebug
   ```
   O APK gerado de depuração estará em:
   `WildBridgeApp/android-sdk-v5-sample/build/outputs/apk/debug/sample-debug.apk`
   Transfira e instale este APK no seu RC.

---

## 🚀 2. Iniciando o Servidor no RC
1. Abra o aplicativo **WildBridge** instalado no RC — os servidores (RTSP e Telemetria) iniciam automaticamente no layout padrão.
2. Anote o **Device IP** exibido no aplicativo (ou obtenha via endpoint `/config`).
3. Certifique-se de que o seu PC e o RC estão conectados na **mesma rede Wi-Fi / Hotspot**.

---

## 🏗️ 3. Arquitetura do Sistema na Estação de Solo (PC)

1. **Controle Remoto do Drone (RC):**
   * Transmite o vídeo via **RTSP** (`rtsp://<IP_DO_CONTROLE>:8554/streaming/live/1`).
   * Expõe a telemetria (altitude e velocidade) via **Socket TCP** na porta `8081`.
2. **Computador (Estação de Solo):**
   * Roda o script em Python unificado.
   * Conecta-se ao RTSP para o vídeo, consome o socket TCP em thread paralela para a telemetria, roda o RT-DETR e exibe os dados no canto superior direito.

---

## ⚙️ 4. Pré-requisitos e Dependências (`requirements.txt`)

Crie um arquivo chamado **`requirements.txt`** na raiz do projeto:
```text
opencv-python>=4.5.0
ultralytics>=8.0.0
numpy>=1.20.0
```
Instale com:
```bash
pip install -r requirements.txt
```

---

## 🌐 5. Configuração do IP do WildBridge (`WILDBRIDGE_IP`)
No código Python, atualize a variável **`WILDBRIDGE_IP`** com o endereço IP do RC anotado no aplicativo WildBridge (ex: `"192.168.1.100"`).

---

## 🐍 6. Código Python Completo (`main.py`)

```python
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
# ATENÇÃO: WILDBRIDGE_IP é o endereço IP do controle remoto do drone na rede local.
WILDBRIDGE_IP = "192.168.1.100"  
RTSP_PORT = "8554"
TELEMETRY_PORT = 8081  # Porta padrão do socket de telemetria do WildBridge

# URL do RTSP do MediaMTX integrado ao WildBridge
WILDBRIDGE_RTSP_URL = f"rtsp://{WILDBRIDGE_IP}:{RTSP_PORT}/streaming/live/1"

CLASS_NAMES = ["fire", "smoke"]
FIRE_ID = 0
SMOKE_ID = 1

COLORS = {
    0: (0, 0, 255),    # Fogo (Vermelho em BGR)
    1: (255, 0, 0)     # Fumaça (Azul em BGR)
}

# Variáveis globais protegidas para telemetria
telemetry_data = {
    "altitude": 0.0,
    "speed": 0.0
}
telemetry_lock = threading.Lock()

def telemetry_socket_loop():
    """Thread em segundo plano que consome o socket TCP contínuo de telemetria do WildBridge."""
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
                            alt = t.get('location', {}).get('altitude', 0.0)
                            spd = t.get('speed', 0.0)
                            
                            with telemetry_lock:
                                telemetry_data["altitude"] = float(alt)
                                telemetry_data["speed"] = float(spd)
                        except json.JSONDecodeError:
                            pass
        except Exception:
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
    # 1. Inicia a thread de telemetria via Socket TCP em background
    t_telemetry = threading.Thread(target=telemetry_socket_loop, daemon=True)
    t_telemetry.start()

    # 2. Carrega o modelo RT-DETR
    print(f"Carregando modelo de: {model_path}")
    model = RTDETR(str(model_path))
    
    # 3. Conecta ao stream RTSP do WildBridge
    print(f"Conectando ao stream RTSP em {WILDBRIDGE_RTSP_URL}...")
    cap = cv2.VideoCapture(WILDBRIDGE_RTSP_URL)
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir o stream RTSP do WildBridge. Verifique o IP do controle.")
        return

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao receber o frame ou o stream terminou.")
            break
            
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
        prev_time = current_time

        with telemetry_lock:
            current_alt = telemetry_data["altitude"]
            current_spd = telemetry_data["speed"]

        results = model.predict(frame, conf=0.5, verbose=False)
        
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
        scores = results[0].boxes.conf.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
        class_ids = results[0].boxes.cls.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])

        if len(class_ids) > 0:
            boxes, scores, class_ids = filter_by_priority(boxes, scores, class_ids, FIRE_ID, SMOKE_ID)
            frame = draw_detections(frame, boxes, scores, class_ids)

        # --- EXIBIÇÃO DA TELEMETRIA NO EXTREMO SUPERIOR DIREITO ---
        h, w, _ = frame.shape
        text_lines = [
            f"ALT: {current_alt:.1f}m",
            f"SPD: {current_spd:.1f}m/s",
            f"FPS: {fps:.1f}"
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, h / 1800.0) 
        thickness = max(1, int(font_scale * 2))
        color = (255, 255, 255)
        margin_right = int(w * 0.015)
        start_y = int(h * 0.04)
        line_spacing = int(h * 0.035)

        for i, text in enumerate(text_lines):
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = w - text_w - margin_right
            y = start_y + (i * line_spacing)
            
            cv2.putText(frame, text, (x + 1, y + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        cv2.imshow("Mavic3M - RT-DETR + WildBridge", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()