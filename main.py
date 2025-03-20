import sys
import cv2
import sqlite3
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QDialog, QLabel, QLineEdit, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from mtcnn import MTCNN
import numpy as np
from facenet_pytorch import InceptionResnetV1

# probando las caracteristicas de esta funcion

class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sistema de Registro")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        self.button_iniciar_sesion = QPushButton("Iniciar Sesión")
        self.button_iniciar_sesion.clicked.connect(self.mostrar_ventana_inicio)
        layout.addWidget(self.button_iniciar_sesion)

        self.button_registro = QPushButton("Registro de Usuario")
        self.button_registro.clicked.connect(self.mostrar_ventana_registro)
        layout.addWidget(self.button_registro)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def mostrar_ventana_inicio(self):
        self.ventana_inicio = VentanaInicio()
        self.ventana_inicio.show()

    def mostrar_ventana_registro(self):
        # Crear una instancia de VentanaRegistro si no existe
        self.ventana_registro = VentanaRegistro()
        self.ventana_registro.show()

#-------------------------------------------------------------#
#-------------------  INICIO DE SESION -----------------------#
#-------------------------------------------------------------#

class VentanaInicio(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Inicio de Sesión")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        self.label_camara = QLabel()
        layout.addWidget(self.label_camara)

        self.label_nombre_usuario = QLabel("Nombre de Usuario:")
        layout.addWidget(self.label_nombre_usuario)

        self.line_edit_nombre_usuario = QLineEdit()
        layout.addWidget(self.line_edit_nombre_usuario)

        self.button_iniciar_verificacion = QPushButton("Iniciar Verificación")
        self.button_iniciar_verificacion.clicked.connect(self.iniciar_verificacion)
        layout.addWidget(self.button_iniciar_verificacion)

        self.setLayout(layout)

        # Inicializar el atributo 'cap' para la captura de video
        self.cap = cv2.VideoCapture("http://192.168.1.19:4747/video")
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_video)
        self.timer.start(30)

    def actualizar_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Mostrar la imagen en el QLabel
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame_rgb, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.label_camara.setPixmap(pixmap)

    def iniciar_verificacion(self):
        nombre_usuario = self.line_edit_nombre_usuario.text()
        # Capturar la imagen del usuario
        ret, frame = self.cap.read()
        if ret:
            # Recortar el rostro usando MTCNN
            detector = MTCNN()
            resultados = detector.detect_faces(frame)
            if resultados:
                resultado = resultados[0]
                x, y, w, h = resultado['box']
                cara_reg = frame[y:y+h, x:x+w]
                # Comparar el nombre del usuario con la base de datos
                if self.verificar_usuario(nombre_usuario):
                    # Verificar la similitud facial
                    if self.verificar_similitud_facial(cara_reg, nombre_usuario):
                        QMessageBox.information(self, "Inicio de Sesión", "¡Bienvenido!")
                    else:
                        QMessageBox.warning(self, "Inicio de Sesión", "¡La verificación facial ha fallado!")
                else:
                    QMessageBox.warning(self, "Inicio de Sesión", "¡El usuario no está registrado!")
            else:
                QMessageBox.warning(self, "Inicio de Sesión", "¡No se detectaron rostros en la imagen!")
        else:
            QMessageBox.warning(self, "Inicio de Sesión", "¡No se pudo capturar la imagen!")

    def verificar_usuario(self, nombre_usuario):
        # Comprobar si el usuario existe en la base de datos
        conn = sqlite3.connect('usuarios.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM usuarios WHERE nombre_usuario = ?", (nombre_usuario,))
        resultado = cursor.fetchone()
        conn.close()
        return resultado is not None

    def verificar_similitud_facial(self, imagen, nombre_usuario):
        # Obtener la imagen almacenada del usuario
        conn = sqlite3.connect('usuarios.db')
        cursor = conn.cursor()
        cursor.execute("SELECT foto FROM usuarios WHERE nombre_usuario = ?", (nombre_usuario,))
        resultado = cursor.fetchone()
        conn.close()
        if resultado:
            # Convertir la imagen almacenada a un formato compatible
            imagen_almacenada = self.convertir_a_numpy(resultado[0])
            # Realizar el reconocimiento facial y comparar características
            return self.reconocer_rostro(imagen_almacenada, imagen)
        return False

    def reconocer_rostro(self, imagen_almacenada, imagen_capturada):
        # Utilizar FaceNet para comparar las imágenes
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        imagen_almacenada_embedding = resnet(imagen_almacenada)
        imagen_capturada_embedding = resnet(imagen_capturada)
        # Calcular la distancia euclidiana entre los embeddings
        distancia = np.linalg.norm(imagen_almacenada_embedding.detach().numpy() - imagen_capturada_embedding.detach().numpy())
        # Establecer un umbral de distancia aceptable
        umbral = 1.0  # Este valor debe ajustarse según tus necesidades
        # Retornar True si la distancia es menor que el umbral, lo que indica que las imágenes son del mismo rostro
        return distancia < umbral

    def convertir_a_numpy(self, blob):
        # Convertir la imagen almacenada como blob a formato numpy
        nparr = np.frombuffer(blob, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
#-------------------------------------------------------------#
#-----------------  REGISTRO DE USUARIO ----------------------#
#-------------------------------------------------------------#

class VentanaRegistro(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Registro de Usuario")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        self.label_camara = QLabel()
        layout.addWidget(self.label_camara)

        self.button_guardar_usuario = QPushButton("Guardar Usuario")
        self.button_guardar_usuario.clicked.connect(self.guardar_usuario)
        layout.addWidget(self.button_guardar_usuario)

        self.line_edit_nombre_usuario = QLineEdit()
        self.line_edit_nombre_usuario.setPlaceholderText("Nombre de usuario")
        layout.addWidget(self.line_edit_nombre_usuario)

        self.setLayout(layout)

        # Configurar la captura de video
        self.cap = cv2.VideoCapture("http://192.168.1.19:4747/video")
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_video)
        self.timer.start(30)

    def actualizar_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Mostrar la imagen en el QLabel
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame_rgb, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.label_camara.setPixmap(pixmap)

    def guardar_usuario(self):
        ret, frame = self.cap.read()
        if ret:
            # Detectar rostros en la imagen
            detector = MTCNN()
            resultados = detector.detect_faces(frame)
            if resultados:
                # Tomar solo el primer rostro detectado
                resultado = resultados[0]
                x, y, w, h = resultado['box']
                cara_reg = frame[y:y+h, x:x+w]
                # Mostrar la imagen en una ventana emergente
                ventana_foto = VentanaFoto(cara_reg, self.line_edit_nombre_usuario.text())
                if ventana_foto.exec_() == QDialog.Accepted:
                    print("Usuario registrado")
                    # Guardar el usuario en la base de datos
                    self.guardar_en_bd(self.line_edit_nombre_usuario.text(), cara_reg)
                    self.close()
                else:
                    print("Registro cancelado")
            else:
                print("No se detectaron rostros en la imagen")

    def guardar_en_bd(self, nombre_usuario, imagen):
        try:
            conn = sqlite3.connect('usuarios.db')
            cursor = conn.cursor()
            # Crear la tabla 'usuarios' si no existe
            cursor.execute('''CREATE TABLE IF NOT EXISTS usuarios (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                nombre_usuario TEXT NOT NULL,
                                foto BLOB NOT NULL
                            )''')
            # Convertir la imagen a formato blob
            foto_blob = self.convertir_a_blob(imagen)
            cursor.execute("INSERT INTO usuarios (nombre_usuario, foto) VALUES (?, ?)", (nombre_usuario, foto_blob))
            conn.commit()
            conn.close()
        except Exception as e:
            print("Error al guardar en la base de datos:", e)

    def convertir_a_blob(self, imagen):
        # Convertir la imagen a formato numpy
        imagen_np = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        # Convertir la imagen numpy a formato blob
        return sqlite3.Binary(cv2.imencode('.jpg', imagen_np)[1])

class VentanaFoto(QDialog):
    def __init__(self, foto, nombre_usuario):
        super().__init__()

        self.setWindowTitle("Vista previa de la foto")
        self.setGeometry(100, 100, 400, 400)

        layout = QVBoxLayout()

        self.label_foto = QLabel()
        self.label_foto.setPixmap(QPixmap.fromImage(self.convertir_imagen(foto)))
        layout.addWidget(self.label_foto)

        botones_layout = QHBoxLayout()

        self.button_aceptar = QPushButton("Aceptar")
        self.button_aceptar.clicked.connect(self.accept)
        botones_layout.addWidget(self.button_aceptar)

        self.button_cancelar = QPushButton("Cancelar")
        self.button_cancelar.clicked.connect(self.reject)
        botones_layout.addWidget(self.button_cancelar)

        layout.addLayout(botones_layout)

        self.setLayout(layout)

        self.nombre_usuario = nombre_usuario

    def convertir_imagen(self, frame):
        # Convertir de BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convertir de formato de matriz OpenCV a QImage
        image = QImage(frame_rgb, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
        return image

#-------------------------------------------------------------#
#-------------------  INICIO DE LA APP -----------------------#
#-------------------------------------------------------------#
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana_principal = VentanaPrincipal()
    ventana_principal.show()
    sys.exit(app.exec_())
