import tkinter as tk
from src.camera import CameraOpenCV

# Função para atualizar o texto na linha superior
def update_label_text(text):
    top_label.config(text=text)

# Criar a janela principal
root = tk.Tk()
root.title("Projeto Integado - Visão Computacional")  # Título da janela
root.geometry("1000x600")  # Dimensões da janela

# Configurar pesos das colunas para 80% e 20%
root.grid_columnconfigure(0, weight=4)  # Coluna 0 (frame_80)
root.grid_columnconfigure(1, weight=1)  # Coluna 1 (frame_20)

# Configurar pesos das linhas para frame_80 (10% e 90%)
root.grid_rowconfigure(0, weight=1)  # Linha superior para o texto
root.grid_rowconfigure(1, weight=9)  # Linha inferior para a câmera

# Frame para a coluna esquerda (80%)
frame_80 = tk.Frame(root)
frame_80.grid(row=0, column=0, rowspan=2, sticky="nsew")
frame_80.grid_rowconfigure(0, weight=1)  # 10%
frame_80.grid_rowconfigure(1, weight=9)  # 90%
frame_80.grid_columnconfigure(0, weight=1)

# Label na linha superior para exibir o botão pressionado
top_label = tk.Label(frame_80, text="", bg="lightblue", font=("Arial", 16))
top_label.grid(row=0, column=0, sticky="nsew")

# Frame para exibir a câmera na linha inferior
camera_frame = tk.Frame(frame_80, bg="black")
camera_frame.grid(row=1, column=0, sticky="nsew")

# Inicializar a câmera no frame inferior
camera = CameraOpenCV(camera_frame, top_label, image_width=800, image_height=500)

# Frame para a coluna direita (20%)
frame_20 = tk.Frame(root)
frame_20.grid(row=0, column=1, rowspan=2, sticky="nsew")

# Adicionar botões na coluna direita
btn1 = tk.Button(frame_20, text="Default", 
    command=lambda: camera.set_function('default'))
btn2 = tk.Button(frame_20, text="Face Mesh", 
    command=lambda: camera.set_function('face_mesh'))
btn3 = tk.Button(frame_20, text="Face Mesh Vivacidade", 
    command=lambda: camera.set_function('face_mesh_vivacidade'))
btn4 = tk.Button(frame_20, text="Face Mesh Janela", 
    command=lambda: camera.set_function('face_mesh_vivacidade_janela'))
btn5 = tk.Button(frame_20, text="Face Recognition", 
    command=lambda: camera.set_function('face_recognition'))
btn6 = tk.Button(frame_20, text="Pipeline", 
    command=lambda: camera.set_function('pipeline'))

# Posicionar os botões
btn1.pack(pady=10, padx=10, fill="x")
btn2.pack(pady=10, padx=10, fill="x")
btn3.pack(pady=10, padx=10, fill="x")
btn4.pack(pady=10, padx=10, fill="x")
btn5.pack(pady=10, padx=10, fill="x")
btn6.pack(pady=10, padx=10, fill="x")


# Iniciar a câmera
camera.start()

# Iniciar o loop do Tkinter
root.mainloop()
