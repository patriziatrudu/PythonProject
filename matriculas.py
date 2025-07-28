from ultralytics import YOLO
import cv2
import easyocr

# Cargar el modelo YOLOv8 entrenado para matrículas
model = YOLO("modelos/license_plate_detector.pt")

# Cargar la imagen
imagen = cv2.imread("imagenes/Seat-Ateca-FR-Frontal.jpg")

# Ejecutar la detección
resultados = model.predict(source=imagen, conf=0.5)

# Inicializar el lector OCR
reader = easyocr.Reader(['es'])

# Procesar cada detección
for r in resultados:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        placa_crop = imagen[y1:y2, x1:x2]

        # Aplicar OCR
        texto = reader.readtext(placa_crop)
        if texto:
            print("Matrícula detectada:", texto[0][1])

        # Dibujar la caja en la imagen
        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Guardar la imagen con anotaciones
cv2.imwrite("resultados/salida_matricula.jpg", imagen)
print("Imagen guardada como salida_matricula.jpg")
