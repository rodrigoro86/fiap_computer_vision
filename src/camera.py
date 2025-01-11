from tkinter import Label
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import numpy as np
import face_recognition
import pickle

# Inicialize o MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class CameraOpenCV:
    def __init__(self, window, top_label,image_width=640, image_height=480):
        self.window = window

        self.top_label = top_label 
        # Configurar layout para centralização
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(0, weight=1)

        # Label para exibição do vídeo
        self.label = Label(window)
        self.label.grid(row=0, column=0, sticky="nsew")  # Centraliza no frame

        # Captura de vídeo
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Dimensões da imagem redimensionada
        self.image_width = image_width
        self.image_height = image_height

        # Inicializar o FaceMesh do MediaPipe
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Estado inicial da função
        self.func = 'default'
        self.comando = ''
        self.count_frame = 0
        self.flag_face_reconhecida = False
        self.flag_mensagem = False
        self.name = ''


        with open("arq/encodings.pkl", "rb") as f:
            encodings_data = pickle.load(f)

        self.encodings_data = encodings_data

    def distance_to_ellipse(self, point, center, axes):
            """
            Calcula a distância aproximada de um ponto até a borda da elipse.
            """
            px, py = point
            cx, cy = center
            a, b = axes

            # Coordenadas normalizadas em relação ao centro da elipse
            normalized_x = (px - cx) / a
            normalized_y = (py - cy) / b

            # Calcula a magnitude do vetor normalizado
            magnitude = np.sqrt(normalized_x ** 2 + normalized_y ** 2)

            # Ponto projetado na elipse
            projected_x = cx + (normalized_x / magnitude) * a
            projected_y = cy + (normalized_y / magnitude) * b

            # Distância do ponto à projeção na elipse
            distance = np.sqrt((px - projected_x) ** 2 + (py - projected_y) ** 2)
            return distance

    def face_mesh_process(self, cv2image, frame):
        results = self.face_mesh.process(cv2image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Desenhar os landmarks e conexões
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                # Desenhar contornos faciais
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

    def face_mesh_vivacidade(self, cv2image, frame):
        # IDs para landmarks dos olhos e da boca
        EAR_THRESHOLD = 0.2  # Limite para detectar piscadas
        MAR_THRESHOLD = 0.5   # Limite para detectar boca aberta
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        MOUTH = [13, 14, 78, 308]

        # Função para calcular a Razão de Abertura do Olho (EAR)
        def eye_aspect_ratio(eye_landmarks, landmarks):
            # Distâncias verticais
            A = dist.euclidean(landmarks[eye_landmarks[1]], landmarks[eye_landmarks[5]])
            B = dist.euclidean(landmarks[eye_landmarks[2]], landmarks[eye_landmarks[4]])
            # Distância horizontal
            C = dist.euclidean(landmarks[eye_landmarks[0]], landmarks[eye_landmarks[3]])
            # EAR
            return (A + B) / (2.0 * C)

        # Função para calcular a Razão de Abertura da Boca (MAR)
        def mouth_aspect_ratio(mouth_landmarks, landmarks):
            # Distância vertical
            A = dist.euclidean(landmarks[mouth_landmarks[0]], landmarks[mouth_landmarks[1]])
            # Distância horizontal
            B = dist.euclidean(landmarks[mouth_landmarks[2]], landmarks[mouth_landmarks[3]])
            # MAR
            return A / B
        
        results = self.face_mesh.process(cv2image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extraia coordenadas dos landmarks
                h, w, _ = frame.shape
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

                # Calcule EAR para ambos os olhos
                left_ear = eye_aspect_ratio(LEFT_EYE, landmarks)
                right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks)

                # Calcule MAR para a boca
                mar = mouth_aspect_ratio(MOUTH, landmarks)

                # Verifique os comandos
                if left_ear < EAR_THRESHOLD:
                    self.comando = "Piscou o olho esquerdo!"
                    #cv2.putText(frame, "Piscou o olho esquerdo!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if right_ear < EAR_THRESHOLD:
                    self.comando = "Piscou o olho direito!"
                    #cv2.putText(frame, "Piscou o olho direito!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if mar > MAR_THRESHOLD:
                    self.comando = "Boca aberta!"
                    #cv2.putText(frame, "Boca aberta!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def face_mesh_vivacidade_janela(self, cv2image, frame):
        
        face_border_points = [10, 152, 234, 454, 323, 93]
        ellipse_axes = (150, 200)    # Tamanhos dos eixos (largura, altura)
        ellipse_color = (0, 255, 0)  # Cor da elipse
        ellipse_thickness = 2        # Espessura da elipse
        distance_threshold = 70

        h, w, _ = frame.shape
        # Atualize dinamicamente o centro da elipse com base no tamanho do frame
        ellipse_center = (w // 2, h // 2)

        cv2.ellipse(frame, ellipse_center, ellipse_axes, 0, 0, 360, ellipse_color, ellipse_thickness)
        # Processa o frame e obtém os landmarks faciais

        results = self.face_mesh.process(cv2image)
        # Verifique se há landmarks faciais detectados

        # Verifique se há algum rosto detectado
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                all_within_threshold = True  # Assume que todos os pontos estão dentro do limite

                # Calcular distância dos pontos da borda do rosto até a elipse
                for idx in face_border_points:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)

                    # Calcule a distância até a elipse
                    distance = self.distance_to_ellipse((x, y), ellipse_center, ellipse_axes)

                    # Verifique se o ponto excede o limite de distância
                    if distance > distance_threshold:
                        all_within_threshold = False

                    # Desenhe o ponto e a distância
                    color = (0, 255, 0) if distance <= distance_threshold else (0, 0, 255)
                    cv2.circle(frame, (x, y), 5, color, -1)
                    cv2.putText(frame, f"{distance:.1f}px", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Alterne a cor da elipse dependendo da posição do rosto
                if all_within_threshold:
                    ellipse_color = (0, 255, 0)  # Verde: Todos os pontos estão dentro do limite
                    cv2.putText(frame, "Rosto enquadrado!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    ellipse_color = (0, 0, 255)  # Vermelho: Algum ponto excede o limite
                    cv2.putText(frame, "Ajuste o rosto!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def face_recognition(self, cv2image, frame):
        # Carregar encodings salvos
        face_locations = face_recognition.face_locations(cv2image)
        face_encodings = face_recognition.face_encodings(cv2image, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.encodings_data["encodings"], face_encoding)
            name = "Desconhecido"

            face_distances = face_recognition.face_distance(self.encodings_data["encodings"], face_encoding)
            best_match_index = face_distances.argmin() if len(face_distances) > 0 else None

            if best_match_index is not None and matches[best_match_index]:
                name = self.encodings_data["names"][best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def indentifica_rosto_dentro_janela(self, face_landmarks, ellipse_center, h, w):
        distance_threshold = 70
        face_border_points = [10, 152, 234, 454, 323, 93]
        ellipse_axes = (150, 200)    # Tamanhos dos eixos (largura, altura)
        # Calcular distância dos pontos da borda do rosto até a elipse
        all_within_threshold = True  # Assume que todos os pontos estão dentro do limite
        for idx in face_border_points:
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            # Calcule a distância até a elipse
            distance = self.distance_to_ellipse((x, y), ellipse_center, ellipse_axes)

            # Verifique se o ponto excede o limite de distância
            if distance > distance_threshold:
                all_within_threshold = False
        
        return all_within_threshold

    def reconhece_rosto(self, cv2image):
        # Carregar encodings salvos
        face_locations = face_recognition.face_locations(cv2image)
        face_encodings = face_recognition.face_encodings(cv2image, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.encodings_data["encodings"], face_encoding)

            face_distances = face_recognition.face_distance(self.encodings_data["encodings"], face_encoding)
            best_match_index = face_distances.argmin() if len(face_distances) > 0 else None

            if best_match_index is not None and matches[best_match_index]:
                return self.encodings_data["names"][best_match_index]
            else:
                return "Desconhecido"

    def teste_vivacidade(self, face_landmarks, h, w):
        # IDs para landmarks dos olhos e da boca
        EAR_THRESHOLD = 0.2  # Limite para detectar piscadas
        MAR_THRESHOLD = 0.5   # Limite para detectar boca aberta
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        MOUTH = [13, 14, 78, 308]

        # Função para calcular a Razão de Abertura da Boca (MAR)
        def mouth_aspect_ratio(mouth_landmarks, landmarks):
            # Distância vertical
            A = dist.euclidean(landmarks[mouth_landmarks[0]], landmarks[mouth_landmarks[1]])
            # Distância horizontal
            B = dist.euclidean(landmarks[mouth_landmarks[2]], landmarks[mouth_landmarks[3]])
            # MAR
            return A / B

        landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
        # Calcule MAR para a boca
        mar = mouth_aspect_ratio(MOUTH, landmarks)
        if mar > MAR_THRESHOLD:
            return True
        else:
            return False

    def pipeline_vivacidade_recognition(self, cv2image, frame):
        self.count_frame += 1
        if self.count_frame >= 60:
            self.count_frame = 0

        ellipse_axes = (150, 200)    # Tamanhos dos eixos (largura, altura)
        ellipse_color = (0, 255, 0)  # Cor da elipse
        ellipse_thickness = 2        # Espessura da elipse

        h, w, _ = frame.shape
        # Atualize dinamicamente o centro da elipse com base no tamanho do frame
        ellipse_center = (w // 2, h // 2)

        cv2.ellipse(frame, ellipse_center, ellipse_axes, 0, 0, 360, ellipse_color, ellipse_thickness)
        # Processa o frame e obtém os landmarks faciais

        results = self.face_mesh.process(cv2image)
        # Verifique se há landmarks faciais detectados

        # Verifique se há algum rosto detectado
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                all_within_threshold = self.indentifica_rosto_dentro_janela(
                    face_landmarks, ellipse_center, h, w
                    )
                if not self.flag_mensagem:
                    self.comando = "Ajuste o rosto dentro da Elipse!"
                    self.flag_mensagem = True

                # Alterne a cor da elipse dependendo da posição do rosto
                if all_within_threshold:
                    self.flag_mensagem = False
                    ellipse_color = (0, 255, 0)  # Verde: Todos os pontos estão dentro do limite
                    if not self.flag_face_reconhecida:
                        name = self.reconhece_rosto(cv2image)
                        if not self.flag_mensagem:
                            self.comando = "Reconhecendo rosto..."
                            self.flag_mensagem = True

                        if name != "Desconhecido":
                            self.flag_face_reconhecida = True
                            self.comando = name
                            self.name = name
                    else:
                        if not self.flag_mensagem:
                            self.comando = "Abra a Boca !"
                            self.flag_mensagem = True

                        flag_boca_aberta = self.teste_vivacidade(face_landmarks, h, w)
                        if flag_boca_aberta:
                            self.comando = f"Teste de Vivacidade Concluído! {self.name}"   
                else:
                    self.comando = "Ajuste o rosto dentro da Elipse!" 

    def reset_flags(self):
        self.flag_face_reconhecida = False
        self.flag_mensagem = False
        self.comando = ''

    def show_frames(self):
        ret, frame = self.cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Verificar qual função está ativa
            if self.func == 'face_mesh':
                self.face_mesh_process(cv2image, frame)
            elif self.func == 'face_mesh_vivacidade':
                self.face_mesh_vivacidade(cv2image, frame)
            elif self.func == 'face_mesh_vivacidade_janela':
                self.face_mesh_vivacidade_janela(cv2image, frame)
            elif self.func == 'face_recognition':
                self.face_recognition(cv2image, frame)
            elif self.func == 'pipeline':
                self.pipeline_vivacidade_recognition(cv2image, frame)
                
        
            # Redimensionar a imagem para o tamanho especificado
            annotated_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(annotated_image)
            img = img.resize((self.image_width, self.image_height), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)

            # Atualizar o label com a imagem
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        # Reexecutar a função após 20ms
        
        self.top_label.config(text=self.comando)
        self.label.after(20, self.show_frames)

        

    def set_function(self, func):
        self.reset_flags()
        self.func = func

    def set_label(self, label):
        self.top_label = label

    def start(self):
        self.show_frames()
        

    def stop(self):
        self.cap.release()
        self.face_mesh.close()
        cv2.destroyAllWindows()
