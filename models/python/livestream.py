import cv2
import torch
import argparse
import os
from datetime import datetime

def grüne_msg(msg):
    print('\x1b[6;30;42m' + msg + '\x1b[0m')

def gelbe_msg(msg):
    print('\x1b[6;30;43m' + msg + '\x1b[0m')

def verarbeite_livestream(args):
    cap = cv2.VideoCapture(0)  # Index anpassen, wenn eine andere Kamera verwendet werden soll
    model = torch.hub.load("yolov5", 'custom', path="real_world_pytorch_model.pt", source='local')
    model.conf = args.threshold
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Verzeichnis für gespeicherte Bilder erstellen
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(args.save_dir, f"output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Bildvorbereitung
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(frame_rgb).to(device)
        img = img.float()
        img /= 255.0
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)

        # Objekterkennung
        results = model(img)[0]

        # Bounding Boxes zeichnen
        for detection in results:
            xmin, ymin, xmax, ymax, confidence, class_id = detection[:6].tolist()
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            label = f"Klasse: {class_id}, Vertrauen: {confidence:.2f}"
            print(label)
            cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Livestream anzeigen
        cv2.imshow("Livestream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' drücken, um den Livestream zu beenden
            break

        # Bilder speichern
        image_name = f"frame_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
        image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(image_path, frame)

    # Aufräumen
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UFO-AI-Beispiel-Skript')
    parser.add_argument('--save_dir', type=str, default='output', help='Verzeichnis zum Speichern der Bilder')
    parser.add_argument('--threshold', type=float, default=0.3, help='Schwellenwert für die Erkennung')

    args = parser.parse_args()

    verarbeite_livestream(args)

