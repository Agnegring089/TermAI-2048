# Jogo de Termo com IA por Reforço

Este projeto utiliza técnicas de Aprendizado por Reforço (Reinforcement Learning, RL) para treinar um agente a jogar uma versão do jogo estilo "Termo". O objetivo do agente é adivinhar uma palavra de 5 letras a partir de tentativas sucessivas, recebendo feedback parcial sobre cada palpite.

## Motivação

A escolha pelo Aprendizado por Reforço se deve ao caráter interativo e sequencial do problema. Ao invés de um modelo puramente supervisionado, o RL permite ao agente aprender por tentativa e erro qual estratégia de palpites maximiza a chance de acertar rapidamente a palavra alvo. Isso torna o agente mais flexível e capaz de lidar com diferentes recompensas parciais e heurísticas de exploração.

## Dataset (Conjunto de Palavras)

O dataset é um conjunto de centenas de palavras de 5 letras, contidas no arquivo `words.json`.  
- **Origem:** Lista customizada compilada a partir de dicionários de palavras em português, filtrada para ter apenas letras minúsculas, sem acentos.  
- **Uso:**  
  - Palavras-alvo: escolhidas aleatoriamente para serem a solução de cada rodada.  
  - Palavras-tentativa: usadas pelo agente como ações (palpites).

## Modelo de IA (RL)

- **Abordagem:** Q-Learning com uma rede neural (Q-Network) para aproximar a função de valor.  
- **Estado:** Representado por um vetor binário indicando quais palavras ainda são plausíveis, conforme feedbacks anteriores.  
- **Ação:** Escolher uma palavra dentre as disponíveis (ainda plausíveis).  
- **Recompensa:**  
  - +1 ao acertar a palavra.  
  - -1 se não acertar após um número limite de tentativas (ex: 10).  
  - Recompensas parciais: +0.1 por letra verde encontrada, +0.05 ao tentar uma palavra nova (incentivo à diversidade), -0.01 por tentativa (para encorajar acertos rápidos).

- **Hiperparâmetros:**  
  - Otimizador: Adam, LR=0.001  
  - Gamma (desconto): 0.99  
  - Política epsilon-greedy para balancear exploração e exploração.

## Treinamento

Durante o treinamento, o agente jogou milhares de partidas, ajustando sua política pelo feedback recebido. Houve um processo de "fine-tuning" no desenho das recompensas e na forma de filtrar repetições de palavras, a fim de reduzir comportamentos indesejados, como insistir na mesma palavra.

## Teste e Métricas

Após o treinamento, o agente foi avaliado em um conjunto de partidas sem exploração (epsilon=0). Coletamos:
- **Taxa de acerto:** Percentual de palavras descobertas no limite de tentativas.
- **Média de tentativas por acerto.**
- **Casos específicos:** Inspeção de exemplos de acertos e erros, analisando o histórico de tentativas.

## Análise Crítica

Apesar dos ajustes e incentivos, o agente ainda apresentava, em certos cenários, o comportamento de repetir palavras com feedback parcialmente positivo. Para contornar isso, foram tomadas medidas como:
- Remover palavras já usadas do conjunto de ações disponíveis no episódio.
- Ajustar as recompensas para incentivar ainda mais a diversidade de tentativas.

Tais mudanças melhoraram o desempenho, mas não eliminaram completamente o desafio. Futuros trabalhos podem incluir representações de estado mais ricas, uso de técnicas de RL mais avançadas (como Replay Buffer, DQN aprimorado) ou combinação com heurísticas linguísticas.

## Conclusão

O projeto demonstra a aplicabilidade do RL a um jogo de raciocínio de palavras, destacando as dificuldades de engenharia de recompensas, controle de exploração e ajustes finos no modelo. Apesar dos desafios, o agente aprendeu a melhorar suas estratégias, oferecendo um ponto de partida para estudos futuros nessa direção.

---
