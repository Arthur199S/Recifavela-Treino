import os
import shutil

origem = r"data/PET"
destino = r"data/NOT_PET"

os.makedirs(destino, exist_ok=True)

movidos = 0

for arquivo in os.listdir(origem):

    nome = arquivo.lower()

    if nome.endswith("_1.jpg") and not nome.endswith("_1_1.jpg"):
        src = os.path.join(origem, arquivo)
        dst = os.path.join(destino, arquivo)

        shutil.move(src, dst)
        movidos += 1

print(f"{movidos} imagens movidas para NOT_PET.")