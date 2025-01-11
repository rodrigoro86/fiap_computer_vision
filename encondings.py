import os
import face_recognition
import pickle

# Diretório contendo as imagens
dataset_path = "arq/imagens"
encodings_file = "arq/encodings.pkl"

# Dicionário para armazenar encodings e nomes
encodings_data = {"encodings": [], "names": []}

# Iterar sobre cada pasta (uma pasta por pessoa)
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):  # Ignorar arquivos que não são diretórios
        continue

    # Iterar sobre cada imagem na pasta
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        try:
            # Carregar a imagem
            image = face_recognition.load_image_file(image_path)

            # Detectar e codificar rostos
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:  # Verificar se há um rosto na imagem
                encodings_data["encodings"].append(encodings[0])
                encodings_data["names"].append(person_name)
                print(f"Processado: {image_path}")
            else:
                print(f"Nenhum rosto encontrado em: {image_path}")
        except Exception as e:
            print(f"Erro ao processar {image_path}: {e}")

# Salvar encodings no arquivo encodings.pkl
with open(encodings_file, "wb") as f:
    pickle.dump(encodings_data, f)

print(f"Encodings salvos com sucesso em '{encodings_file}'!")
