import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from termo import carregar_palavras, avaliar_palpite, escolher_palavra_aleatoria


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class AgenteReforco:
    def __init__(self, palavras, lr=0.001, gamma=0.99):
        self.palavras = palavras
        self.gamma = gamma
        self.action_size = len(palavras)
        self.state_size = self.action_size

        self.qnetwork = QNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def state_to_tensor(self, state):
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    def escolher_acao(self, state, epsilon=0.1):
        valid_indices = np.where(state == 1)[0]
        if len(valid_indices) == 0:
            return random.randint(0, self.action_size - 1)

        if random.random() < epsilon:
            return np.random.choice(valid_indices)
        else:
            with torch.no_grad():
                s = self.state_to_tensor(state)
                q_values = self.qnetwork(s).numpy()[0]
                valid_q = q_values[valid_indices]
                best_idx = valid_indices[np.argmax(valid_q)]
            return best_idx

    def treinar(self, state, action, reward, next_state, done):
        s = self.state_to_tensor(state)
        if not done:
            ns = self.state_to_tensor(next_state)

        with torch.no_grad():
            target_q = reward
            if not done:
                next_q = self.qnetwork(ns).max().item()
                target_q = reward + self.gamma * next_q

        q_values = self.qnetwork(s)
        pred_q = q_values[0, action]

        loss = self.criterion(pred_q, torch.tensor(target_q, dtype=torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def palavra_consistente(cand, palpite, feedback):
    cand_count = {}
    pal_count = {}
    for ch in cand:
        cand_count[ch] = cand_count.get(ch, 0) + 1
    for ch in palpite:
        pal_count[ch] = pal_count.get(ch, 0) + 1

    for i, f in enumerate(feedback):
        if f == 'G' and cand[i] != palpite[i]:
            return False

    for i, f in enumerate(feedback):
        if f == 'G':
            ch = palpite[i]
            cand_count[ch] -= 1
            pal_count[ch] -= 1

    for i, f in enumerate(feedback):
        if f == 'Y':
            ch = palpite[i]
            if ch not in cand_count or cand_count[ch] == 0:
                return False
            else:
                cand_count[ch] -= 1

    for i, f in enumerate(feedback):
        if f == 'B':
            ch = palpite[i]
            if ch in cand_count and cand_count[ch] > 0:
                return False

    return True


def aplicar_feedback(palpite, feedback, palavras, estado_atual):
    new_state = estado_atual.copy()
    for i, w in enumerate(palavras):
        if new_state[i] == 1:
            if not palavra_consistente(w, palpite, feedback):
                new_state[i] = 0
    return new_state


def contar_novas_green_letras(palpite, feedback):
    g_count = sum(1 for f in feedback if f == 'G')
    return 0.1 * g_count


def jogar_episodio(agente, palavras_alvo, palavras_tentativa, epsilon=0.1):
    alvo = escolher_palavra_aleatoria(palavras_alvo)
    state = np.ones(len(palavras_tentativa), dtype=int)
    max_tentativas = 4
    historico_tentativas = []
    palavras_usadas = set()

    for tentativa in range(max_tentativas):
        # Filtra o estado removendo palavras já usadas (coloca 0 onde já foi usada)
        state_filtrado = state.copy()
        for i, w in enumerate(palavras_tentativa):
            if w in palavras_usadas:
                state_filtrado[i] = 0  # não deixa esta palavra disponível

        # Se não houver nenhuma palavra disponível que não foi usada, então o agente terá que repetir (situação extrema)
        # Mas isso será raro se o conjunto de palavras é grande.
        action = agente.escolher_acao(state_filtrado, epsilon)
        palpite = palavras_tentativa[action]

        feedback = avaliar_palpite(alvo, palpite)
        historico_tentativas.append((palpite, feedback))

        partial_reward = contar_novas_green_letras(palpite, feedback)
        partial_reward -= 0.01

        if palpite not in palavras_usadas:
            partial_reward += 0.05
        palavras_usadas.add(palpite)

        next_state = aplicar_feedback(palpite, feedback, palavras_tentativa, state)

        if all(f == 'G' for f in feedback):
            reward = 1.0 + partial_reward
            agente.treinar(state, action, reward, next_state, done=True)
            return 1, tentativa + 1, alvo, historico_tentativas
        else:
            if tentativa == max_tentativas - 1:
                reward = -1.0 + partial_reward
                agente.treinar(state, action, reward, next_state, done=True)
                return -1, max_tentativas, alvo, historico_tentativas
            else:
                reward = 0.0 + partial_reward
                agente.treinar(state, action, reward, next_state, done=False)
                state = next_state

    return -1, max_tentativas, alvo, historico_tentativas


if __name__ == "__main__":
    palavras = carregar_palavras("words.json")
    palavras_alvo = palavras
    palavras_tentativa = palavras

    agente = AgenteReforco(palavras, lr=0.001, gamma=0.99)

    episodios = 5000
    for e in range(episodios):
        resultado, tentativas, alvo, hist = jogar_episodio(agente, palavras_alvo, palavras_tentativa, epsilon=0.1)
        if (e + 1) % 100 == 0:
            print(f"Episódio {e + 1}, resultado: {resultado}")

    # Teste após treinamento sem exploração
    teste_episodios = 100
    acertos = []
    erros = []
    for _ in range(teste_episodios):
        res, tent, alvo, hist = jogar_episodio(agente, palavras_alvo, palavras_tentativa, epsilon=0.0)
        if res == 1:
            acertos.append((alvo, tent, hist))
        else:
            erros.append((alvo, tent, hist))

    exemplos_acertos = random.sample(acertos, min(len(acertos), 5)) if acertos else []
    exemplos_erros = random.sample(erros, min(len(erros), 5)) if erros else []


    def imprimir_tabela(dados, titulo):
        if not dados:
            print(f"\n{titulo}: Nenhum dado disponível.")
            return
        print(f"\n{titulo}:")
        print("-" * 50)
        print(f"{'Palavra':<15}{'Tentativas':<10}")
        print("-" * 50)
        for palavra, tentativas, hist in dados:
            print(f"{palavra:<15}{tentativas:<10}")
            print("  Tentativas:")
            for i, (palp, feed) in enumerate(hist, start=1):
                feedback_str = ''.join(feed)
                print(f"    {i}. {palp} -> {feedback_str}")
            print("-" * 50)


    imprimir_tabela(exemplos_acertos, "Exemplos de Palavras Acertadas")
    imprimir_tabela(exemplos_erros, "Exemplos de Palavras Erradas")

    # Imprime quantas palavras acertou e quantas errou no teste
    total_acertos = len(acertos)
    total_erros = len(erros)
    print(f"\nNo teste de {teste_episodios} episódios:")
    print(f"Acertos: {total_acertos}")
    print(f"Erros: {total_erros}")
