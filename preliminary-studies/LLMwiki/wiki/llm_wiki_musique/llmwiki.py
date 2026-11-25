
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from slugify import slugify as slugify
import json
import unicodedata


# FEITO = PRE PROCESSAMENTO PARA CNSTRUCAO DA WIKI, FLUXO DE COSTRUIR ESTA CORRETO. PARTE DE PERGUNTAR, VERIFICAR RESPOSTA TUDO OK
# AJUSTAR = PESQUISAR AUTOVERIFICACAO DA CONSTRUCAO DA WIKI, CONEXAO ENTRE WIKIS, AJUSTAR PARTE DE BUSCA PRA TRAZER
#   COLOCAR/AJUSTAR A FUNCAO DE ATUALIZAR WIKI APPOS RESPOSTAS INCORRETAS
#   AJUSTAR PROMPTS PARA INGLES



""""Carregando o GEMMA12B"""
# define como os pesos vao ser comprimidos na memoria
config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,            # guarda os pesos em 4 bits em vez de 16/32
    bnb_4bit_quant_type="nf4",     # tipo de quantizacao otimizado pra pesos de rede neural
    bnb_4bit_compute_dtype=torch.bfloat16,  # na hora de calcular, descomprime pra 16 bits
)

model2 = "/media/work/pedrogoularte/models/gemma-3-12b-it-bnb-4bit"
tokenizer2 = AutoTokenizer.from_pretrained(model2)
# torch_dtype bfloat16: sem isso o 4 bits carrega o resto em float16 e o gemma 3 gera resposta vazia
model2 = AutoModelForCausalLM.from_pretrained( model2, device_map="auto",quantization_config=config_4bit, torch_dtype=torch.bfloat16)


"""Notícias e Caminho da Pasta da Wiki"""

PASTA_BASE = "data/musique_ans_v1.0_dev.jsonl"


PASTA_WIKI = "data/wiki"

os.makedirs(PASTA_WIKI, exist_ok=True)

""""Funcao auxiliar para formatar titulo do arquivo de cada wiki"""
# transforma um titulo em nome sem espaco ou acento p 
def slugify_function(titulo):
  return slugify(titulo)

# print (slugify_function("Neymar - carreira de 25"))
"""Funcao auxiliar para tokenizar trechos"""
def tokenizar(texto):
    texto = texto.lower()  # maiuscula vira minuscula
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()  # tira acento
    return re.findall(r"\w+", texto)  # quebra em palavras e descarta pontuacao

""""Funcao do gemma para retornar texto"""
def gerar_gemma_12b(prompt, max_tokens=350):

    inputs = tokenizer2(prompt, return_tensors="pt").to(model2.device)  # tokeniza o prompt e manda pro device do modelo pesado

    outputs = model2.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer2.eos_token_id)  # gera a resposta de forma determinística

    tamanho_entrada = inputs["input_ids"].shape[1]
    resposta_ids = outputs[0][tamanho_entrada:]  # pega só os tokens novos, sem repetir o prompt

    resposta = tokenizer2.decode(resposta_ids, skip_special_tokens=True).strip()

    return resposta

"""# Funções Ingest"""
# le todos os arquivos da wiki e pega o titulo
def listar_paginas(pasta = PASTA_WIKI):
    if not os.path.isdir(pasta):
        return []

    paginas = []

    for nome_arquivo in sorted(os.listdir(pasta)):

        if not nome_arquivo.endswith(".md"):
          continue
        caminho_arquivo = os.path.join(pasta, nome_arquivo)

        # le a primeira linha (que é o titulo e adciona na lista de paginas)
        with open(caminho_arquivo, "r", encoding="utf-8") as arquivo:
            linha = arquivo.readline().strip()

        titulo = linha.replace("#", "").strip()
        paginas.append((titulo, caminho_arquivo))

    return paginas

PASTA_CACHE = "data/llm_wiki_cache"  # salva a wiki completa em apenas um arquivo .md, para passar como contexto caso necessario
os.makedirs(PASTA_CACHE, exist_ok=True)

def carregar_wiki_completa(pasta=PASTA_WIKI):
    # pega todas as paginas, le, lista e salva na pasta
    textos = []
    paginas = listar_paginas(pasta)
    for titulo, caminho in paginas:
        with open(caminho, "r", encoding="utf-8") as arq:
            textos.append(arq.read())

    conteudo_completo = "\n\n".join(textos)

    caminho_saida = PASTA_CACHE + "/" + "wiki_completa.md"
    with open(caminho_saida, "w", encoding="utf-8") as saida:
        saida.write(conteudo_completo)

SCHEMA_WIKI = """Regras da wiki:
- Cada pagina tem um titulo curto (2 a 5 palavras) representando um unico topico
- O conteudo e uma lista de fatos em bullets, um fato por linha, sem repeticao
- Nao invente informacoes que nao estejam no texto fornecido
- Escreva sempre em portugues do Brasil
- Se o trecho so menciona um nome/topico de passagem, sem fatos substanciais sobre ele,
  NAO crie uma pagina nova para isso - ignore a mencao.
"""

def buscar_pagina_relacionada(trecho, pasta=PASTA_WIKI):
    paginas = listar_paginas(pasta)
    if not paginas:
        return None

    temp = []
    for titulo, caminho in paginas:
        temp.append(titulo)
    lista_titulos = "\n".join(temp)

    prompt = f"""Você recebe uma lista de tópicos existentes e um trecho novo de texto.
      Responda APENAS com o nome exato de um tópico da lista se o trecho for sobre o mesmo
      assunto dele, ou responda "NENHUM" se não houver relação clara.

      Tópicos existentes: {lista_titulos}

      Trecho novo: {trecho}

      Tópico relacionado:"""

    resposta = gerar_gemma_12b(prompt, max_tokens=20)

    for titulo, caminho in paginas:
        if titulo.lower() in resposta.lower():
            return (titulo, caminho)
    return None

def buscar_paginas_pergunta(trecho, pasta=PASTA_WIKI):
    paginas = listar_paginas(pasta)
    if not paginas:
        return []

    temp = []
    for titulo, caminho in paginas:
        temp.append(titulo)
    lista_titulos = "\n".join(temp)

    prompt = f"""Você recebe uma lista de tópicos existentes e uma pergunta.
      Responda APENAS com o nome exato de TODOS tópicos da lista se a pergunta for sobre o mesmo
      assunto de algum trecho, ou responda "NENHUM" se não houver relação clara.

      Tópicos existentes: {lista_titulos}

      Pergunta: {trecho}

      Tópico relacionado:"""

    resposta = gerar_gemma_12b(prompt, max_tokens=100)
    relacionadas = []

    for titulo, caminho in paginas:
        if titulo.lower() in resposta.lower():
            relacionadas.append((titulo, caminho))
    if relacionadas:
        # print(len(relacionadas))
        return relacionadas
    return []

"""# Gerar Título e Gerar conteúdos da Wiki"""

def buscar_titulo(trecho):
    prompt = f"""{SCHEMA_WIKI}
Leia o trecho abaixo e responda apenas com um titulo curto (2 a 5 palavras) e ESPECIFICO
para uma pagina de wiki sobre ele.

Trecho: {trecho}

Titulo:"""

    return gerar_gemma_12b(prompt, max_tokens=15).strip()

def buscar_conteudo(trecho):
    prompt = f"{SCHEMA_WIKI}\nLeia o trecho abaixo e liste os fatos dele em bullets, um fato por linha. Trecho: {trecho} Conteudo:"
    return gerar_gemma_12b(prompt, max_tokens=200).strip()

"""# Criação da Wiki"""

def criar_pagina(trecho, pasta=PASTA_WIKI):
    if not os.path.isdir(pasta):
        os.makedirs(pasta)

    # gera o titulo e os bullets points da noticia com a gemma baseado no texto
    titulo = buscar_titulo(trecho)
    conteudo = buscar_conteudo(trecho)

    nome_arquivo = slugify_function(titulo) + ".md"
    caminho_arquivo = pasta + "/" + nome_arquivo

    with open(caminho_arquivo, "w", encoding="utf-8") as arquivo:
        arquivo.write(f"# {titulo}\n\n")
        arquivo.write(conteudo)

    return (titulo, caminho_arquivo)

def atualizar_pagina(caminho, trecho_novo):
    # pega a pagina a ser atualziada, e pede pro gemma listar fatos novos que nao estao na pagina e estao no trecho passado
    with open(caminho, "r", encoding="utf-8") as f:
        conteudo_atual = f.read()

    prompt = f"""{SCHEMA_WIKI}
Pagina atual: {conteudo_atual}
Informacao nova: {trecho_novo}
Liste APENAS os fatos da informacao nova que AINDA NAO estao na pagina atual, em bullets.
Se nao houver fato novo, responda apenas: NENHUM."""

    resposta = gerar_gemma_12b(prompt, max_tokens=150).strip()

    # filtra em Python: so aceita linhas que sao bullet (comecam com "-")
    linhas_novas = []
    for linha in resposta.split("\n"):
        linha = linha.strip()
        if not linha.startswith("-"):
            continue
        if linha not in conteudo_atual:
            linhas_novas.append(linha)

    if linhas_novas:
        with open(caminho, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(linhas_novas))

    return caminho

def processar_trecho(trecho, pasta=PASTA_WIKI):
    relacionada = buscar_pagina_relacionada(trecho)
    if relacionada:
        return atualizar_pagina(relacionada[1], trecho)
    else:
        return criar_pagina(trecho, pasta)

def processar_base(base):
    with open(base, "r", encoding="utf-8") as arquivo:
        texto = arquivo.read()
        for linha in texto.splitlines()[:60]:
            linha_dicionario = json.loads(linha)
            for paragrafo in linha_dicionario["paragraphs"]:
                # passa o o titulo e o paragrafo para construir a wiki
                titulo = paragrafo["title"]
                conteudo = paragrafo["paragraph_text"]
                trecho = titulo + "\n" + conteudo
                processar_trecho(trecho)


"""# Funções de Pergunta e Resposta"""


def resposta_tem_fato_novo(pergunta, resposta):
    caminho_cache = PASTA_CACHE + "/" + "wiki_completa.md"

    with open(caminho_cache, "r", encoding="utf-8") as f:
      wiki = f.read()


    # verificar se a resposta ou parte dele aparece na wiki
    if resposta.strip() in wiki:
        return False

    return True

def atualizar_wiki_com_resposta(pergunta, resposta):
    # checa se essa resposta ja esta registrada em alguma pagina
    if resposta_tem_fato_novo(pergunta, resposta):

        # se for fato novo, monta o trecho e manda pro processar_trecho arquivar
        trecho_para_arquivar = f"Pergunta: {pergunta} Resposta: {resposta}"
        print("não esta na wiki, sera arquivada")
        processar_trecho(trecho_para_arquivar)
    else:
        print("essa informacao ja esta na wiki, nada a fazer")


def perguntar_wiki(pergunta, max_tokens=50):
    caminho_cache = PASTA_CACHE + "/" + "wiki_completa.md"
    limite_contexto=3000

    relacionadas = buscar_paginas_pergunta(pergunta)

    if relacionadas:
        contexto = ""
        for titulo, caminho in relacionadas:
            with open(caminho, "r", encoding="utf-8") as f:
                contexto += f.read()
    else:
        with open(caminho_cache, "r", encoding="utf-8") as f:
            contexto = f.read()
        contexto = contexto[:limite_contexto]  # corta o contexto pra nao estourar memoria

    prompt_final = f"""Voce e um assistente que responde perguntas usando apenas o contexto abaixo.
Nao invente informacoes que nao estejam no contexto.
Contexto: {contexto} Pergunta: {pergunta} Resposta:"""

    if relacionadas:
        resposta = gerar_gemma_12b(prompt_final, max_tokens=max_tokens)
    else:
        print("Quantidade de paginas relacionadas: 0")
        resposta = gerar_gemma_12b(prompt_final, max_tokens=max_tokens)

    return resposta,relacionadas

def verificar_resposta(resposta_modelo, resposta_certa):
    # cria uma lista para tirar os artigos da resposta
    resposta_sem_artigo = []

    # tokeniza a resposta com a funcao do bm25
    resposta_tokenizada = tokenizar(resposta_modelo)
    # remove os artigos
    for  palavra in resposta_tokenizada:
      if palavra not in ["a", "the" , "an"]:
        resposta_sem_artigo.append(palavra)

    # acertou fica true se a resposta for igual a alguma resposta certa
    acertou = False

    # em cada variacao da resposta certa, tokeniza, tira o artigo e compara com a resposta do modelo
    for certa in resposta_certa:
       certa_sem_artigo = []
       certa_tokenizada = tokenizar(certa)
       for palavra in certa_tokenizada:
             if palavra not in ["a", "the" , "an"]:
               certa_sem_artigo.append(palavra)

       # se as palavras forem iguais acertou
       if resposta_sem_artigo == certa_sem_artigo:
          acertou = True

    return acertou
def processar_perguntas_e_respostas(PASTA_BASE):
    perguntas = []
    respostas = []
    with open(PASTA_BASE, "r", encoding= "utf-8") as arquivo:
        trecho = arquivo.read()
        for linha in trecho.splitlines()[:60]:
            linha_dicionario = json.loads(linha)
            perguntas.append(linha_dicionario["question"])
            respostas.append([linha_dicionario["answer"]] + linha_dicionario["answer_aliases"])
    return perguntas, respostas


"""Funcao que roda as perguntaas, pega as perguntas e respostas das listas, roda a pergunta e confere a resposta"""
def rodar_perguntas():
    numPerguntas = 0
    numAcertos = 0
    perguntas, respostas = processar_perguntas_e_respostas(PASTA_BASE)
    for pergunta, resposta in zip (perguntas, respostas):
        resposta_modelo, relacionadas = perguntar_wiki(pergunta)
        print(f"\n Pergunta: {pergunta}")
        resultado = verificar_resposta(resposta_modelo, resposta)
        print(f"\nResposta Gerada: {resposta_modelo}")
        print(f"\n Resposta: {resposta}")
        print (f"\nAcertou: {resultado}")

        print(f"Quantidade de paginas relacionadas: {len(relacionadas)}\n\nPaginas relacionadas: {relacionadas}")
        numPerguntas += 1
        if (resultado):
            numAcertos +=1

    print ("\n===== RELATORIO FINAL=====")
    print(f"\nNumero de acertos = {numAcertos} / {numPerguntas}")
    acuracia = round((numAcertos / numPerguntas) * 100, 1)
    print (f"\nAcuracia: {acuracia} % ")


# processar_base(PASTA_BASE)
# print(listar_paginas())
#carregar_wiki_completa()