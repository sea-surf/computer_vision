"""
Inferência do modelo RTDETR com o vídeo transmitido do drone mavic3m a partir do aplicativo dji pilot 2 do drone
Rodar o mediamtx no terminal
Conseguir o ip rtmp://ip:1935/live/drone com ipconfig no terminal, colocar esse link no dji2 para transmitir em rtmp
"""

import cv2
import time
from ultralytics import YOLO
from ultralytics import RTDETR
from pathlib import Path

ROOT = Path(__file__).parent.parent

model_path = ROOT / "models" / "fire_rtdetr.pt"
predict_dir = ROOT / "predict" / "fire_rtdetr"

def main():
    # Load model
    model = RTDETR(str(model_path))
    
    # URL RTMP do MediaMTX (mesmo configuradao no DJI Pilot 2)
    rtmp_url = "rtmp://localhost:1935/live/drone"
    
    print(f"Conectando ao stream RTMP em {rtmp_url}...")
    cap = cv2.VideoCapture(rtmp_url)
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir o stream de vídeo.")
        print("Verifique se o MediaMTX está rodando e se o drone está transmitindo para o IP correto.")
        return

    print("Stream conectado! Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao receber o frame ou o stream terminou. Tentando reconectar...")
            break
            
        # Roda a inferência do YOLO no frame
        results = model(frame, stream=True)
        
        # Extrai o frame com as labels
        for r in results:
            annotated_frame = r.plot()
            # Mostra o frame na tela
            cv2.imshow("Mavic3m Real-time", annotated_frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
