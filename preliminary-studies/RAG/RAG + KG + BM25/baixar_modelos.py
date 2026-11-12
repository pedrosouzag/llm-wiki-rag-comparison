# baixa os modelos direto pra pasta do lab, sem usar o cache do hugging face
# (o cache trava no disco de rede do lab por causa dos arquivos .lock)

import os
import fnmatch
import requests
from huggingface_hub import HfApi, hf_hub_url, get_token

PASTA_MODELOS = "/media/work/pedrogoularte/models"

# repositorio no hugging face e nome da pasta onde vai salvar
modelos = [
    ("unsloth/gemma-3-12b-it-bnb-4bit", "gemma-3-12b-it-bnb-4bit"),
    ("intfloat/multilingual-e5-base", "multilingual-e5-base"),
]

# so os arquivos que precisa (pesos, configs e tokenizer)
padroes = ["*.json", "*.safetensors", "*.model", "*.jinja", "1_Pooling/*"]

api = HfApi()
token = get_token()
cabecalho = {}
if token:
    cabecalho = {"Authorization": "Bearer " + token}

for repo, pasta in modelos:
    destino = os.path.join(PASTA_MODELOS, pasta)

    for arquivo in api.list_repo_files(repo):
        # pula o que nao precisa
        quer = False
        for padrao in padroes:
            if fnmatch.fnmatch(arquivo, padrao):
                quer = True
        if not quer:
            continue

        caminho = os.path.join(destino, arquivo)
        os.makedirs(os.path.dirname(caminho), exist_ok=True)

        # baixa em pedacos de 1mb direto pro arquivo final
        print("baixando", repo, arquivo)
        with requests.get(hf_hub_url(repo, arquivo), headers=cabecalho, stream=True) as resposta:
            resposta.raise_for_status()
            with open(caminho, "wb") as saida:
                for pedaco in resposta.iter_content(1024 * 1024):
                    saida.write(pedaco)

    print("OK", pasta)
