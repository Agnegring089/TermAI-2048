import json
import random

def carregar_palavras(caminho="words.json"):
    with open(caminho, "r") as f:
        palavras = json.load(f)
    # Filtra ou garante que todas tenham 5 letras
    palavras = [p for p in palavras if len(p) == 5]
    return palavras

def avaliar_palpite(alvo, palpite):
    """Dado um palpite e uma palavra alvo, retornar lista de feedback:
    'G' para letra correta e posição correta (Green),
    'Y' para letra correta posição errada (Yellow),
    'B' para letra inexistente (Black/Gray).
    """
    resultado = [""] * 5
    alvo_restante = list(alvo)

    # Primeiro, marca as exatas (G)
    for i in range(5):
        if palpite[i] == alvo[i]:
            resultado[i] = "G"
            alvo_restante[i] = None

    # Depois, marca as letras corretas na posição errada (Y)
    for i in range(5):
        if resultado[i] == "":
            if palpite[i] in alvo_restante:
                resultado[i] = "Y"
                # Remove a letra da lista para não ser usada mais de uma vez
                idx = alvo_restante.index(palpite[i])
                alvo_restante[idx] = None
            else:
                resultado[i] = "B"

    return resultado

def escolher_palavra_aleatoria(palavras):
    return random.choice(palavras)

def validar_palavra(palavra, palavras):
    """Checa se a palavra dada é válida (existe no conjunto de palavras).
    Pode ser usada para garantir que o palpite está na lista."""
    return palavra in palavras

if __name__ == "__main__":
    # Exemplo de uso
    lista_palavras = carregar_palavras("words.json")
    alvo = escolher_palavra_aleatoria(lista_palavras)
    print("Palavra alvo:", alvo)

    # Palpite de teste
    palpite = "casa"
    if validar_palavra(palpite, lista_palavras):
        feedback = avaliar_palpite(alvo, palpite)
        print(f"Palpite: {palpite}, Feedback: {feedback}")
    else:
        print(f"A palavra {palpite} não é válida!")
