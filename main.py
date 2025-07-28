import os
import cv2
from ultralytics import YOLO

# Crear carpeta de resultados si no existe
os.makedirs("resultados", exist_ok=True)

# Cargar el modelo YOLOv8
model = YOLO("modelos/yolov8n.pt")

# Ruta del vídeo de entrada
video_path = "videos/Chickens_eating.mp4"

# Abrir el vídeo
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Crear el archivo de salida
output_path = "resultados/video_anotado_pollos.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Procesar cada fotograma
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame)
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

# Liberar recursos
cap.release()
out.release()

print(f"✅ Vídeo procesado y guardado en: {output_path}")
