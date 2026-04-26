import os
import shutil

origem = r"C:\Users\arthu\Downloads\archive\VN_trash_classification\Test\Alu"
destino = r"../data/NOT_PET"

os.makedirs(destino, exist_ok=True)

movidos = 0

for arquivo in os.listdir(origem):

    src = os.path.join(origem, arquivo)

    if os.path.isfile(src):
        nome, ext = os.path.splitext(arquivo)

        novo_nome = f"{nome}_1{ext}"  # acrescenta +1 no final
        dst = os.path.join(destino, novo_nome)

        shutil.move(src, dst)
        movidos += 1

print(f"De {origem}:{movidos} imagens movidas para {destino}.")