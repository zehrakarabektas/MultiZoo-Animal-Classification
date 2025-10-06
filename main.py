from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QPushButton, QFileDialog,
    QVBoxLayout, QFrame, QHBoxLayout, QGraphicsDropShadowEffect
)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QBrush, QColor, QIcon
from PyQt5.QtCore import Qt, QSize
import sys
import torch
from torchvision import transforms
from PIL import Image
from model import load_model

with open("zoo_class_names.txt", "r") as f:
    zoo_names = [line.strip() for line in f.readlines()]

model = load_model("son_best_beit_model.pth")
model.eval()

resize_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class UploadImage(QFrame):
    def __init__(self, icon_path):
        super().__init__()
        self.setStyleSheet("background-color: rgba(255, 255, 255, 180); border-radius: 20px;")
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.icon_path = icon_path
        self.uploadIcon()

    def uploadIcon(self):
        pixmap = QPixmap(self.icon_path)
        self.label.setPixmap(pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.label.setFixedSize(120, 120)
        self.setFixedSize(260, 260)
        self.label.move((self.width() - 120) // 2, (self.height() - 120) // 2)

    def resetImage(self):
        self.uploadIcon()

class ResultBox(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: rgba(255, 255, 255, 200); border-radius: 20px;")
        self.setFixedSize(360, 200)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(12)

        self.icon_guess = QLabel()
        self.icon_guess.setFixedSize(32, 32)
        self.icon_guess.setPixmap(QPixmap("search.png").scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.icon_guess.setStyleSheet("background: transparent;")

        self.text_guess = QLabel()
        self.text_guess.setFont(QFont("Arial", 14))
        self.text_guess.setStyleSheet("color: #333; background: transparent;")

        guess_layout = QHBoxLayout()
        guess_layout.setSpacing(10)
        guess_layout.addWidget(self.icon_guess)
        guess_layout.addWidget(self.text_guess)

        self.icon_score = QLabel()
        self.icon_score.setFixedSize(32, 32)
        self.icon_score.setPixmap(QPixmap("target.png").scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.icon_score.setStyleSheet("background: transparent;")

        self.text_score = QLabel()
        self.text_score.setFont(QFont("Arial", 14))
        self.text_score.setStyleSheet("color: #333; background: transparent;")

        score_layout = QHBoxLayout()
        score_layout.setSpacing(10)
        score_layout.addWidget(self.icon_score)
        score_layout.addWidget(self.text_score)

        self.text_alternatives = QLabel()
        self.text_alternatives.setFont(QFont("Arial", 12))
        self.text_alternatives.setStyleSheet("color: #666; background: transparent;")
        self.text_alternatives.setWordWrap(True)

        layout.addLayout(guess_layout)
        layout.addLayout(score_layout)
        layout.addWidget(self.text_alternatives)

        self.setLayout(layout)
        self.hide()

    def set_result(self, guess, score, alt_preds):
        self.text_guess.setText(f"<b>Tahmin:</b> {guess}")
        self.text_score.setText(f"<b>Güven:</b> {score:.2f}%")
        alt_text = "<b>Alternatifler:</b><br>• " + "<br>• ".join([f"{name} (%{conf:.1f})" for name, conf in alt_preds])
        self.text_alternatives.setText(alt_text)
        self.show()

    def reset(self):
        self.text_guess.setText("")
        self.text_score.setText("")
        self.text_alternatives.setText("")
        self.hide()

class AppMain(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🐾 Hayvan Türü Sınıflandırma App Uygulaması")
        self.setFixedSize(800, 700)

        background = QPalette()
        bg = QPixmap("background.png").scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        background.setBrush(QPalette.Window, QBrush(bg))
        self.setPalette(background)

        self.title = QLabel("Hayvan Türü\nSınıflandırması", self)
        self.title.setFont(QFont("Arial", 28, QFont.Bold))
        self.title.setStyleSheet("color: #2b2b2b;")
        self.title.setAlignment(Qt.AlignCenter)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setXOffset(2)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 160))
        self.title.setGraphicsEffect(shadow)

        self.upload = UploadImage("upload_icon.png")
        self.result = ResultBox()

        self.button = QPushButton(" Görüntü Yükle", self)
        self.button.setIcon(QIcon("add-photo.png"))
        self.button.setIconSize(QSize(28, 28))
        self.button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 20px;
                padding: 10px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.button.clicked.connect(self.load_image)

        self.reset_button = QPushButton(" Sıfırla", self)
        self.reset_button.setIcon(QIcon("switch-camera.png"))
        self.reset_button.setIconSize(QSize(28, 28))
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 20px;
                padding: 10px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.reset_button.clicked.connect(self.reset_view)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.button)
        button_layout.addWidget(self.reset_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 20, 10, 20)
        layout.addWidget(self.title, alignment=Qt.AlignHCenter)
        layout.addSpacing(40)
        layout.addWidget(self.upload, alignment=Qt.AlignHCenter | Qt.AlignTop)
        layout.addSpacing(10)
        layout.addWidget(self.result, alignment=Qt.AlignHCenter)
        layout.addStretch()
        layout.addLayout(button_layout)
        layout.addSpacing(10)

        self.setLayout(layout)

    def guess_image(self, path):
        img = Image.open(path).convert("RGB")
        img_tensor = resize_image(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top3_scores, top3_indices = torch.topk(probs, 3)
            top3 = [(zoo_names[i], top3_scores[0][j].item() * 100) for j, i in enumerate(top3_indices[0])]
            print("🔍 Top-3 Tahminler:")
            for cls, scr in top3:
                print(f"• {cls}: %{scr:.2f}")
        return top3

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Bir görsel seçin", "", "Image Files (*.png *.jpg *.jpeg)")
        if path:
            pixmap = QPixmap(path)
            if pixmap.isNull():
                print("⚠️ Görsel yüklenemedi!")
                return

            scaled_pixmap = pixmap.scaled(320, 210, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.upload.label.setPixmap(scaled_pixmap)
            self.upload.label.setFixedSize(scaled_pixmap.size())
            self.upload.setFixedSize(scaled_pixmap.width() + 20, scaled_pixmap.height() + 20)
            self.upload.label.move(10, 10)

            top3 = self.guess_image(path)
            main_guess, main_conf = top3[0]
            alternatives = top3[1:]
            self.result.set_result(main_guess, main_conf, alternatives)

    def reset_view(self):
        self.upload.resetImage()
        self.upload.label.setFixedSize(120, 120)
        self.upload.setFixedSize(260, 260)
        self.upload.label.move((260 - 120) // 2, (260 - 120) // 2)
        self.result.reset()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AppMain()
    window.show()
    sys.exit(app.exec_())
