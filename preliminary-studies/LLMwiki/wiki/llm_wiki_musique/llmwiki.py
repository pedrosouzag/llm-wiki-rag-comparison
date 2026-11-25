
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

SCHEMA_WIKI = """Wiki rules:
- Each page has a short title (2 to 5 words) about a single topic
- The content is a list of facts in bullets, one fact per line, with no repetition
- Do not make up information that is not in the given text
- Always write in English and keep names exactly as they appear in the text
- If the text only mentions a name/topic in passing, without real facts about it,
  do NOT create a new page for it - ignore the mention.
"""

def buscar_pagina_relacionada(trecho, pasta=PASTA_WIKI):
    paginas = listar_paginas(pasta)
    if not paginas:
        return None

    temp = []
    for titulo, caminho in paginas:
        temp.append(titulo)
    lista_titulos = "\n".join(temp)

    prompt = f"""
        You receive a list of existing topics and a new piece of text.
        Answer ONLY with the exact name of one topic from the list if the text is about the same
        subject, or answer "NONE" if there is no clear relation.

        Existing topics: {lista_titulos}

        New text: {trecho}

        Related topic:"""

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

    prompt = f"""
        You receive a list of existing topics and a question.
        Answer ONLY with the exact names of ALL topics from the list that are related to the question,
        or answer "NONE" if there is no clear relation.

        Existing topics: {lista_titulos}

        Question: {trecho}

        Related topics:"""

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
    prompt = f"""
        {SCHEMA_WIKI}
        Read the text below and answer only with a short (2 to 5 words) and SPECIFIC title
        for a wiki page about it.

        Text: {trecho}

        Title:"""

    return gerar_gemma_12b(prompt, max_tokens=15).strip()

def buscar_conteudo(trecho):
    prompt = f"""
        {SCHEMA_WIKI}
        Read the text below and list its facts in bullets, one fact per line.

        Text: {trecho}

        Content:"""

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

    prompt = f"""
        {SCHEMA_WIKI}
        Current page: {conteudo_atual}

        New information: {trecho_novo}

        List ONLY the facts from the new information that are NOT YET in the current page, in bullets.
        If there is no new fact, answer only: NONE."""

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


"""Funcao que guarda a resposta na wiki, usando as paginas que a pergunta usou"""
def atualizar_wiki_com_resposta(pergunta, resposta, relacionadas):
    # nao guarda nada quando o modelo nao achou a resposta
    if "not found" in resposta.lower():
        #print("resposta nao encontrada, nada a guardar")
        return

    # monta o trecho com a pergunta e a resposta
    trecho = f"Question: {pergunta} Answer: {resposta}"

    if relacionadas:
        # passa nas paginas que a wiki usou pra responder
        # pegamos apenas o caminho da tupla
        for titulo, caminho in relacionadas:
            atualizar_pagina(caminho, trecho)
    else:
        # nenhuma pagina foi usada na pergunta
        # o processar_trecho procura pagina pro trecho com a resposta, se achar atualiza, se nao cria uma nova
        processar_trecho(trecho)

    # refaz a wiki completa pra ela ficar igual as paginas
    carregar_wiki_completa()

def o_que_falta(pergunta, contexto):
    # pergunta pro gemma se os trechos ja bastam ou o que ainda falta buscar
    prompt = f"""You are helping to answer a question that may need more than one step.
        Read the question and the passages below.
        If the passages already have all the information needed to answer the question, write only: DONE
        If something is missing, write only the next search to find it.
        The search must be short and use the names you already found in the passages.
        Example: if you found that the film Blue River was directed by John Carter but nothing about where he was born, write: John Carter birthplace

        Passages:
        {contexto}

        Question: {pergunta}"""

    # resposta curta, so uma busca ou DONE
    texto = gerar_gemma_12b(prompt, max_tokens=30)

    # se o gemma disse que ja tem tudo devolve vazio
    if "done" in texto.lower():
        return ""

    # quebra o texto em linhas e pega so a primeira
    linhas = texto.split("\n")
    primeira_linha = linhas[0]

    # tira os espacos do comeco e do fim
    sem_espaco = primeira_linha.strip()

    # tira as aspas do comeco e do fim
    proxima_busca = sem_espaco.strip('"') # as veze ele devolve respostas com aspas

    return proxima_busca

def busca_profundidade(pergunta):
    caminho_cache = PASTA_CACHE + "/" + "wiki_completa.md"
    limite_contexto=3000
    contexto = ""
    # busca as paginas relacionadas
    relacionadas = buscar_paginas_pergunta(pergunta)

    if relacionadas:
        # le cada pagina e junta no contexto
        for titulo, caminho in relacionadas:
            with open(caminho, "r", encoding="utf-8") as f:
                contexto += f.read() + "\n\n"
    else:
        # nenhuma pagina achada, usa o comeco da wiki completa
        with open(caminho_cache, "r", encoding="utf-8") as f:
            contexto = f.read()
        contexto = contexto[:limite_contexto]  # corta o contexto pra nao estourar memoria

    # guarda as buscas ja feitas pra nao repetir
    buscas_feitas = []
    # roda a profundidade
    for i in range(2):
      falta = o_que_falta(pergunta, contexto)
      # se o gemma disse que ja tem tudo para de buscar
      if falta == "":
        print("sem nova busca")
        break
      # se pediu uma busca que ja foi feita para de buscar
      if falta.lower() in buscas_feitas:
        print("busca repetida, para")
        break
      buscas_feitas.append(falta.lower())
      # mostra o que o gemma pediu pra buscar
      print("nova busca", i + 1, ":", falta)
      # busca as paginas da busca nova
      novas_paginas = buscar_paginas_pergunta(falta)
      # verifica se a pagina ja esta nas relacionadas, compara pelo caminho que e unico
      for titulo, caminho in novas_paginas:
         repetido = False
         for titulo_existente, caminho_existente in relacionadas:
            if caminho == caminho_existente:
               repetido = True
               break
         # se a pagina e nova adiciona na lista e le ela pro contexto
         if not repetido:
            relacionadas.append((titulo, caminho))
            with open(caminho, "r", encoding="utf-8") as f:
               contexto += f.read() + "\n\n"

    # mostra quantas paginas juntou no total
    print("TOTAL DE PAGINAS:", len(relacionadas))
    return contexto, relacionadas

def perguntar_wiki(pergunta, contexto,  max_tokens=50):

    prompt_final = f"""
        Answer the question using only the wiki pages below. Do not make anything up.
        The question may need more than one step: first find the first fact, then use it to find the answer.
        Write at most 3 short lines of reasoning, then the last line like this:
        Final answer: <only the name or the value>
        If the pages do not have the answer, write: Final answer: not found

        Wiki pages:
        {contexto}

        Question: {pergunta}"""

    texto = gerar_gemma_12b(prompt_final, max_tokens=max_tokens)

    # mostra o raciocinio pra ver o erro
    print("RACIOCINIO:", texto)

    # pega so o que vem depois de final answer:
    resposta = texto
    for linha in texto.split("\n"):
      if "final answer:" in linha.lower():
        posicao = linha.lower().find("final answer:")
        resposta = linha[posicao + len("final answer:"):].strip()

    return resposta

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
        print(f"\n Pergunta: {pergunta}")
        # busca as paginas da pergunta e vai descendo nas buscas novas, devolve o contexto e as paginas usadas
        contexto, relacionadas = busca_profundidade(pergunta)

        # responde a pergunta com o contexto montado
        resposta_modelo = perguntar_wiki(pergunta, contexto)

        # verifica o resultado
        resultado = verificar_resposta(resposta_modelo, resposta)

        if resultado:
            atualizar_wiki_com_resposta(pergunta, resposta_modelo, relacionadas)
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