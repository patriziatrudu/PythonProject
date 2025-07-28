import streamlit as st
import cv2
from ultralytics import YOLO
import easyocr
import numpy as np

# Título de la app
st.title("Detección de Matrículas con YOLOv8 + EasyOCR")

# Cargar imagen
imagen_path = "imagenes/Seat-Ateca-FR-Frontal.jpg"
imagen = cv2.imread(imagen_path)

# Mostrar imagen original
st.subheader("Imagen original")
st.image(imagen[..., ::-1], channels="RGB", use_container_width=True)

# Cargar modelo y OCR
model = YOLO("modelos/license_plate_detector.pt")
reader = easyocr.Reader(['es'])

# Ejecutar detección
resultados = model.predict(source=imagen, conf=0.5)

# Procesar resultados
matricula_detectada = ""
for r in resultados:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        placa_crop = imagen[y1:y2, x1:x2]
        texto = reader.readtext(placa_crop)
        if texto:
            matricula_detectada = texto[0][1]
        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Mostrar imagen anotada
st.subheader("Resultado con detección")
st.image(imagen[..., ::-1], channels="RGB", use_column_width=True)

# Mostrar matrícula detectada
if matricula_detectada:
    st.success(f"Matrícula detectada: **{matricula_detectada}**")
else:
    st.warning("No se detectó ninguna matrícula.")
