import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QSpinBox,
    QPushButton, QVBoxLayout, QGridLayout, QFrame, QHBoxLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


# =========================
# MODEL TRAINING PART
# =========================

df = pd.read_csv("data/diabetes_data_upload.csv")

df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

binary_columns = [
    "Polyuria", "Polydipsia", "sudden weight loss", "weakness",
    "Polyphagia", "Genital thrush", "visual blurring", "Itching",
    "Irritability", "delayed healing", "partial paresis",
    "muscle stiffness", "Alopecia", "Obesity"
]

for col in binary_columns:
    df[col] = df[col].map({"Yes": 1, "No": 0})

df["class"] = df["class"].map({"Positive": 1, "Negative": 0})

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = SVC(kernel="linear")
model.fit(X_train_scaled, y_train)


# =========================
# UI PART
# =========================

class DiabetesUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Diabetes Prediction System")
        self.resize(1500, 850)

        self.setStyleSheet("""
            QWidget {
                background-color: #f1f5f9;
                font-family: Nirmala UI;
                color: #0f172a;
            }

            QLabel {
                background: transparent;
                color: #0f172a;
                font-size: 13px;
                font-weight: 600;
            }

            QFrame#mainCard {
                background-color: #ffffff;
                border-radius: 24px;
                border: 1px solid #dbeafe;
            }

            QFrame#leftPanel {
                background-color: #2563eb;
                border-radius: 22px;
            }

            QFrame#rightPanel {
                background-color: #f8fbff;
                border-radius: 18px;
                border: 1px solid #dbeafe;
            }

            QComboBox, QSpinBox {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #bfdbfe;
                border-radius: 9px;
                padding: 5px 8px;
                min-height: 34px;
                font-size: 13px;
            }

            QComboBox:hover, QSpinBox:hover {
                border: 1px solid #2563eb;
            }

            QPushButton {
                background-color: #16a34a;
                color: white;
                border: none;
                border-radius: 14px;
                padding: 12px;
                font-size: 17px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #15803d;
            }
        """)

        self.inputs = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(25, 18, 25, 25)

        title = QLabel("Diabetes Prediction System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            color: #2563eb;
            font-size: 34px;
            font-weight: bold;
            margin-bottom: 8px;
        """)
        outer.addWidget(title)

        main_card = QFrame()
        main_card.setObjectName("mainCard")

        main_layout = QHBoxLayout(main_card)
        main_layout.setContentsMargins(26, 26, 26, 26)
        main_layout.setSpacing(26)

        # =========================
        # LEFT PANEL
        # =========================

        left_panel = QFrame()
        left_panel.setObjectName("leftPanel")
        left_panel.setFixedWidth(390)

        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(28, 28, 28, 28)

        icon = QLabel("🩺")
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet("color: white; font-size: 48px;")
        left_layout.addWidget(icon)

        left_title = QLabel("AI Health Assistant")
        left_title.setAlignment(Qt.AlignCenter)
        left_title.setStyleSheet("""
            color: white;
            font-size: 26px;
            font-weight: bold;
            margin-top: 6px;
        """)
        left_layout.addWidget(left_title)

        left_text = QLabel(
            "Enter patient symptoms and the SVM machine learning model "
            "will predict diabetes risk.\n\n"
            "বাংলা ও English symptom input support করা হয়েছে।"
        )
        left_text.setWordWrap(True)
        left_text.setAlignment(Qt.AlignCenter)
        left_text.setMinimumHeight(125)
        left_text.setStyleSheet("""
            color: #ffffff;
            font-size: 15px;
            font-weight: 600;
            line-height: 1.5;
            margin-top: 12px;
        """)
        left_layout.addWidget(left_text)

        self.result_label = QLabel("🔍 Waiting for Prediction\n\nফলাফল দেখার জন্য তথ্য দিন")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setMinimumHeight(140)
        self.result_label.setStyleSheet("""
            background-color: #ffffff;
            color: #2563eb;
            font-size: 20px;
            font-weight: bold;
            padding: 22px;
            border-radius: 18px;
            margin-top: 22px;
        """)
        left_layout.addWidget(self.result_label)

        note = QLabel("Educational ML prediction system. Not a replacement for medical diagnosis.")
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignCenter)
        note.setStyleSheet("""
            color: #dbeafe;
            font-size: 12px;
            margin-top: 14px;
        """)
        left_layout.addWidget(note)
        left_layout.addStretch()

        main_layout.addWidget(left_panel)

        # =========================
        # RIGHT PANEL
        # =========================

        right_panel = QFrame()
        right_panel.setObjectName("rightPanel")

        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(24, 24, 24, 20)

        form_title = QLabel("Patient Symptom Information")
        form_title.setStyleSheet("""
            color: #1d4ed8;
            font-size: 23px;
            font-weight: bold;
            margin-bottom: 10px;
        """)
        right_layout.addWidget(form_title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)

        features = [
            ("Age", "বয়স", "age"),
            ("Gender", "লিঙ্গ", "gender"),
            ("Polyuria", "বারবার প্রসাব", "Polyuria"),
            ("Polydipsia", "বেশি পিপাসা", "Polydipsia"),
            ("Sudden Weight Loss", "হঠাৎ ওজন কমা", "sudden weight loss"),
            ("Weakness", "দুর্বলতা", "weakness"),
            ("Polyphagia", "বেশি ক্ষুধা", "Polyphagia"),
            ("Genital Thrush", "ইনফেকশন", "Genital thrush"),
            ("Visual Blurring", "চোখ ঝাপসা", "visual blurring"),
            ("Itching", "চুলকানি", "Itching"),
            ("Irritability", "খিটখিটে মেজাজ", "Irritability"),
            ("Delayed Healing", "ক্ষত শুকাতে দেরি", "delayed healing"),
            ("Partial Paresis", "আংশিক দুর্বলতা", "partial paresis"),
            ("Muscle Stiffness", "পেশী শক্ত", "muscle stiffness"),
            ("Alopecia", "চুল পড়া", "Alopecia"),
            ("Obesity", "স্থূলতা", "Obesity")
        ]

        for i, (eng, bn, key) in enumerate(features):
            row = i % 8
            col = (i // 8) * 2

            label = QLabel(f"{eng}\n{bn}")
            label.setWordWrap(True)
            label.setMinimumHeight(48)
            label.setStyleSheet("""
                background-color: #eaf1fb;
                color: #0f172a;
                padding: 6px 10px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
            """)

            if key == "age":
                widget = QSpinBox()
                widget.setRange(1, 120)
                widget.setValue(40)
            elif key == "gender":
                widget = QComboBox()
                widget.addItems(["Male | পুরুষ", "Female | মহিলা"])
            else:
                widget = QComboBox()
                widget.addItems(["No | না", "Yes | হ্যাঁ"])

            widget.setFixedHeight(40)
            self.inputs[key] = widget

            grid.addWidget(label, row, col)
            grid.addWidget(widget, row, col + 1)

        right_layout.addLayout(grid)

        predict_btn = QPushButton("Predict Diabetes Risk | ফলাফল দেখুন")
        predict_btn.setFixedHeight(56)
        predict_btn.clicked.connect(self.predict_diabetes)
        right_layout.addWidget(predict_btn)

        footer = QLabel("Powered by SVM Machine Learning")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("""
            color: #64748b;
            font-size: 12px;
            margin-top: 5px;
        """)
        right_layout.addWidget(footer)

        main_layout.addWidget(right_panel)

        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 3)

        outer.addWidget(main_card)

    def predict_diabetes(self):
        data = []

        data.append(self.inputs["age"].value())

        gender_text = self.inputs["gender"].currentText()
        gender = 1 if "Male" in gender_text else 0
        data.append(gender)

        feature_order = [
            "Polyuria", "Polydipsia", "sudden weight loss", "weakness",
            "Polyphagia", "Genital thrush", "visual blurring", "Itching",
            "Irritability", "delayed healing", "partial paresis",
            "muscle stiffness", "Alopecia", "Obesity"
        ]

        for feature in feature_order:
            value = 1 if "Yes" in self.inputs[feature].currentText() else 0
            data.append(value)

        sample_df = pd.DataFrame([data], columns=X.columns)
        sample_scaled = scaler.transform(sample_df)
        prediction = model.predict(sample_scaled)

        if prediction[0] == 1:
            self.result_label.setText("⚠️ Diabetes Positive\n\nডায়াবেটিস ঝুঁকি সনাক্ত হয়েছে")
            self.result_label.setStyleSheet("""
                background-color: #fee2e2;
                color: #dc2626;
                font-size: 21px;
                font-weight: bold;
                padding: 22px;
                border-radius: 18px;
                margin-top: 22px;
            """)
        else:
            self.result_label.setText("✅ Diabetes Negative\n\nডায়াবেটিস ঝুঁকি পাওয়া যায়নি")
            self.result_label.setStyleSheet("""
                background-color: #dcfce7;
                color: #15803d;
                font-size: 21px;
                font-weight: bold;
                padding: 22px;
                border-radius: 18px;
                margin-top: 22px;
            """)


if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setFont(QFont("Nirmala UI", 10))

    window = DiabetesUI()
    window.showMaximized()
    app.exec()