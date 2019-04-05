from random import random
import csv
from math import log2, exp

TAM_CAMADA_ENTRADA = 784
TAM_CAMADA_INTERMEDIARIA = 111
TAM_CAMADA_SAIDA = 10

alpha = 0.2

# respostas = [
#     [1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, 1, -1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, 1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, 1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, 1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1, 1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1],
#     [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1]
# ]


respostas = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
]


x = []
z = [0 for _ in range(TAM_CAMADA_INTERMEDIARIA)]
y = [0 for _ in range(TAM_CAMADA_SAIDA)]

v = [[random() * 2 - 1 for _ in range(TAM_CAMADA_INTERMEDIARIA)] for _ in range(TAM_CAMADA_ENTRADA)]
w = [[random() * 2 - 1 for _ in range(TAM_CAMADA_SAIDA)] for _ in range(TAM_CAMADA_INTERMEDIARIA)]
delta_v = [[0 for _ in range(TAM_CAMADA_INTERMEDIARIA)] for _ in range(TAM_CAMADA_ENTRADA)]
delta_w = [[0 for _ in range(TAM_CAMADA_SAIDA)] for _ in range(TAM_CAMADA_INTERMEDIARIA)]

erro = [0 for _ in range(TAM_CAMADA_SAIDA)]

acertos = []
erros = []
iteracao = 0


def treinamento():
    with open('mnist_train.csv', newline='') as arq:
        testes = csv.reader(arq, delimiter=',')
        carrega_pesos()
        i = 1
        for linha in testes:
            if i > 70000:
                break
            treinar(linha)
            verifica_acerto(int(linha[0]))
            print_info(i, linha)
            i += 1
        salva_pesos()
        print_acertos_erros()


def testar():
    with open('mnist_train.csv', newline='') as arq:
        testes = csv.reader(arq, delimiter=',')
        carrega_pesos()
        i = 1
        for linha in testes:
            if i > 999999:
                break
            analisar(linha)
            verifica_acerto(int(linha[0]))
            print_info(i, linha)
            i += 1
        print_acertos_erros()


def analisar(linha):
    global x
    x = [log2(int(n)) if n != '0' else 0 for n in linha[1:]]
    feedforward(x, z, v, TAM_CAMADA_INTERMEDIARIA)
    feedforward(z, y, w, TAM_CAMADA_SAIDA)


def verifica_acerto(index):
    global acertos, erros
    if respostas[index] == [round(n) for n in y]:
        acertos[iteracao] += 1
    else:
        erros[iteracao] += 1


def treinar(linha):
    resposta = respostas[int(linha[0])]
    analisar(linha)
    calcula_erro(resposta)
    calcula_delta()
    corrije_pesos(w, delta_w)
    corrije_pesos(v, delta_v)


def feedforward(entrada, saida, pesos, t):
    for i in range(t):
        soma = somatorio(entrada, pesos, i)
        saida[i] = funcao_ativacao(soma)


def somatorio(entrada, pesos, j):
    soma = 0
    for i in range(len(entrada)):
        soma += entrada[i] * pesos[i][j]
    return soma


def funcao_ativacao(n):
    return 1 / (1 + (exp(-n)))
    # return 2 / (1 + exp(-n)) - 1


def derivada(n):
    return n * (1 - n)
    # return (1 / 2) * (1 + n) * (1 - n)


def calcula_erro(resposta):
    for k in range(TAM_CAMADA_SAIDA):
        erro[k] = ((resposta[k] - y[k]) * derivada(y[k]))
        for j in range(TAM_CAMADA_INTERMEDIARIA):
            delta_w[j][k] = alpha * erro[k] * z[j]


def calcula_delta():
    for j in range(len(z)):
        soma = 0
        for k in range(len(w[j])):
            soma += erro[k] * w[j][k]
        e = soma * derivada(z[j])
        for i in range(TAM_CAMADA_ENTRADA):
            delta_v[i][j] = alpha * e * x[i]


def corrije_pesos(pesos, deltas):
    for i in range(len(pesos)):
        for j in range(len(pesos[i])):
            pesos[i][j] += deltas[i][j]


def print_info(i, linha):
    if i % 100 == 0:
        print('-------------------------')
        print(i, acertos, ', ', erros)
        print('target: ', [float(r) for r in respostas[int(linha[0])]])
        print('...res: ', [round(i, 1) for i in y])
        print('..erro: ', [round(e, 5) for e in erro])


def print_acertos_erros():
        print('------------------')
        print(iteracao)
        print('Arcertos: %s' % acertos)
        print('Erros:    %s' % erros)


def salva_pesos():
    tw = open('w.txt', 'w')
    tv = open('v.txt', 'w')

    for i in range(len(v)):
        linha = ','.join(str(n) for n in v[i])
        tv.write(linha)
        if i < (len(v) - 1):
            tv.write('\n')

    for i in range(len(w)):
        linha = ','.join(str(a) for a in w[i])
        tw.write(linha)
        if i < (len(w) - 1):
            tw.write('\n')

    tw.close()
    tv.close()


def carrega_pesos():
    tw = open('w.txt', 'r')
    tv = open('v.txt', 'r')

    global v, w
    v.clear()
    w.clear()
    for l in tv.readlines():
        v.append([float(n) for n in l.split(',')])
    for l in tw.readlines():
        w.append([float(n) for n in l.split(',')])

    tw.close()
    tv.close()


if __name__ == '__main__':
    for _ in range(10):
        acertos.append(0)
        erros.append(0)
        treinamento()
        # testar()
        iteracao += 1
