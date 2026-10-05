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
    
    # URL do MediaMTX (use rtmp://10.149.171.97:1935/drone ou rtsp://10.149.171.97:8554/drone)
    rtmp_url = "rtmp://10.149.171.97:1935/drone"
    
    print(f"Conectando ao stream em {rtmp_url}...")
    cap = cv2.VideoCapture(rtmp_url)
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir o stream de vídeo.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao receber o frame ou o stream terminou.")
            break
            
        results = model(frame, stream=True)

        for r in results:
            annotated_frame = r.plot()
            cv2.imshow("Mavic3m Real-time", annotated_frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()