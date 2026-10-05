import os
from pathlib import Path
import threading
import cv2
from ultralytics import RTDETR

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|max_delay;0"
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "fire_rtdetr.engine"


class FreshFrameReader:
  """Thread leve que consome o buffer do OpenCV e mantém apenas o último frame."""

  def __init__(self, src):
    self.cap = cv2.VideoCapture(src)
    self.ret, self.frame = False, None
    self.running = True
    self.t = threading.Thread(target=self._reader, daemon=True)
    self.t.start()

  def _reader(self):
    while self.running:
      ret, frame = self.cap.read()
      if not ret:
        break
      self.ret, self.frame = ret, frame

  def read(self):
    return self.ret, self.frame

  def release(self):
    self.running = False
    self.cap.release()


def main():
  model = RTDETR(str(MODEL_PATH))
  stream = FreshFrameReader("rtmp://localhost:1935/live/drone")

  print("Processando em tempo real na Jetson...")

  while True:
    ret, frame = stream.read()
    if not ret or frame is None:
      continue

    # A Jetson roda na velocidade máxima dela (9 FPS)
    # sempre pegando o frame mais recente gerado pelo drone
    results = model(frame, imgsz=640, conf=0.4, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("Mavic 3M - Jetson Real-Time", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  stream.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()