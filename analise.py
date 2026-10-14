# -*- coding: utf-8 -*-
"""
analise.py — Análise das questões do ENEM via Gemini API.

Caderno: 2013-2-6 CINZA (Reaplicação/PPL)
Aluno:   João Vitor de Jesus Oliveira — CCM Darcy José Costa — 3B
"""

import os
import re
import json
import glob
import time
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

# =========================================================
# CONFIGURAÇÕES GERAIS
# =========================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "Defina GEMINI_API_KEY no arquivo .env (copie de .env.example)."
    )

genai.configure(api_key=GEMINI_API_KEY)

MODELO = "gemini-2.0-flash"   # troque por "gemini-1.5-pro" se quiser mais qualidade

PASTA_TEXTOS   = "textos_extraidos"
PASTA_IMAGENS  = "imagens"
PASTA_SAIDA    = "resultados"
ARQ_ESTRUTURA  = "estrutura.json"
ARQ_MATRIZ     = "matriz_referencia_enem.json"
ARQ_GABARITO   = "gabarito.txt"

IDENTIFICACAO_CADERNO = {
    "ano": 2013,
    "dia": 2,
    "numero": 6,
    "cor": "CINZA",
    "aplicacao": "REAPLICAÇÃO_PPL",
}

ESTUDANTE = {
    "nome_completo": "João Vitor de Jesus Oliveira",
    "colegio": "CCM Darcy José Costa",
    "turma": "3B",
}

os.makedirs(PASTA_SAIDA, exist_ok=True)


# =========================================================
# LEITURA DE APOIO
# =========================================================
def carregar_gabarito(caminho: str) -> dict:
    """Lê gabarito.txt em formatos flexíveis: '1 - A', '1-A', '1 A', '1. A'."""
    gabarito = {}
    if not os.path.exists(caminho):
        print(f"[AVISO] Gabarito não encontrado em {caminho}. Seguindo sem gabarito.")
        return gabarito

    padrao = re.compile(r"^\s*(\d+)\s*[-\.\)]?\s*([A-Ea-e])\s*$")
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith("#"):
                continue
            m = padrao.match(linha)
            if m:
                gabarito[int(m.group(1))] = m.group(2).upper()

    print(f"[INFO] Gabarito carregado: {len(gabarito)} itens.")
    return gabarito


def carregar_json(caminho: str) -> dict:
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# PARSER DE NOME DE ARQUIVO
# =========================================================
def parse_nome_arquivo(nome_arquivo: str) -> dict:
    """
    Extrai metadados do nome do arquivo.
    Formatos aceitos:
      questao-100.txt          -> {"numero": 100, "ingles": False, "espanhol": False}
      questao-ingles-3.txt     -> {"numero": 3,   "ingles": True,  "espanhol": False}
      questao-espanhol-2.txt   -> {"numero": 2,   "ingles": False, "espanhol": True}
    """
    stem = Path(nome_arquivo).stem.lower()
    info = {"numero": None, "ingles": False, "espanhol": False}

    m = re.match(r"questao-(ingles|espanhol)-(\d+)", stem)
    if m:
        info["numero"] = int(m.group(2))
        info["ingles"] = m.group(1) == "ingles"
        info["espanhol"] = m.group(1) == "espanhol"
        return info

    m = re.match(r"questao-(\d+)", stem)
    if m:
        info["numero"] = int(m.group(1))
        return info

    # fallback: primeiro número do nome
    m = re.search(r"(\d+)", stem)
    if m:
        info["numero"] = int(m.group(1))

    return info


# =========================================================
# PROMPT
# =========================================================
SUGESTION_PROMPT = """
Você é um analista especialista em questões do ENEM e em classificação segundo a Matriz de Referência do ENEM.

FONTES E HIERARQUIA:
1. O TXT da questão é a fonte primária do conteúdo, comando e alternativas.
2. estrutura.json é a fonte normativa para área, competência, habilidades, eixo cognitivo e objetos/conteúdos da Matriz.
3. gabarito.json é a fonte oficial do gabarito quando houver correspondência.

REGRAS ABSOLUTAS:
- Não invente informações. Se não houver evidência, use null ou [].
- Nunca crie área, competência, habilidade ou eixo fora de estrutura.json.
- Ao selecionar item da Matriz, copie ID e texto EXATAMENTE como estão em estrutura.json.
- Nunca parafraseie, resuma ou corrija a redação da Matriz.
- A habilidade principal representa principalmente o que o estudante precisa fazer para resolver a questão, não apenas o assunto superficial.
- Considere comando + operação cognitiva + conteúdo + alternativas + correspondência com a Matriz.
- Habilidades relacionadas só entram quando houver evidência clara.
- Gabarito oficial tem prioridade sobre qualquer inferência da LLM.
- Sem gabarito oficial, correta deve permanecer null.
- Analise todas as alternativas presentes e não crie alternativas ausentes.
- Justificativas devem ser sustentadas pela questão.
- Não use conhecimento externo para preencher lacunas.
- OCR pode conter erros; corrija mentalmente apenas erros evidentes necessários à compreensão, sem inventar texto.
- Analise cada questão independentemente.
- Confiança: alta = correspondência clara; media = alguma ambiguidade; baixa = evidência insuficiente/ambígua.
- Retorne somente o JSON exigido pelo schema.

A CLASSIFICAÇÃO DE CONTEÚDO é analítica e não deve ser apresentada como se fosse redação oficial da Matriz.
"""


def montar_prompt(questao_txt: str, matriz: dict, estrutura: dict,
                  gabarito_oficial: str | None,
                  meta: dict) -> str:
    matriz_str = json.dumps(matriz, ensure_ascii=False)
    estrutura_str = json.dumps(estrutura, ensure_ascii=False)

    idioma = "português"
    if meta.get("ingles"):
        idioma = "inglês (língua estrangeira moderna)"
    elif meta.get("espanhol"):
        idioma = "espanhol (língua estrangeira moderna)"

    return f"""{SUGESTION_PROMPT}

========================
MATRIZ DE REFERÊNCIA (fonte normativa, NÃO invente itens fora dela):
{matriz_str}

========================
SCHEMA DE SAÍDA (siga RIGOROSAMENTE esta estrutura):
{estrutura_str}

========================
METADADOS DA QUESTÃO:
- Número: {meta.get('numero')}
- Idioma: {idioma}
- Ano: {IDENTIFICACAO_CADERNO['ano']}
- Dia: {IDENTIFICACAO_CADERNO['dia']}
- Caderno: {IDENTIFICACAO_CADERNO['numero']} / Cor: {IDENTIFICACAO_CADERNO['cor']}
- Aplicação: {IDENTIFICACAO_CADERNO['aplicacao']}

GABARITO OFICIAL DESTA QUESTÃO: {gabarito_oficial if gabarito_oficial else "NÃO DISPONÍVEL"}

========================
TEXTO DA QUESTÃO (fonte primária, extraída por OCR):
\"\"\"
{questao_txt}
\"\"\"

Retorne APENAS o JSON no schema acima, preenchendo todos os campos.
Preencha "questao.identificacao_caderno", "questao.identificacao_questao",
"questao.conteudo_questao.gabarito" e "estudante" com base nos metadados acima.
Não escreva comentários fora do JSON.
"""


# =========================================================
# CHAMADA À API
# =========================================================
def analisar_questao(model, questao_txt: str, matriz: dict, estrutura: dict,
                     gabarito_oficial, meta: dict):
    prompt = montar_prompt(questao_txt, matriz, estrutura, gabarito_oficial, meta)

    for tentativa in range(3):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                },
            )
            texto = resp.text.strip()
            texto = re.sub(r"^```(?:json)?|```$", "", texto, flags=re.MULTILINE).strip()
            return json.loads(texto)
        except Exception as e:
            print(f"    [ERRO] tentativa {tentativa+1}/3: {e}")
            time.sleep(2)
    return None


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 70)
    print("ANÁLISE LLM — ENEM 2013-2-6 CINZA (Reaplicação PPL)")
    print("=" * 70)

    matriz    = carregar_json(ARQ_MATRIZ)
    estrutura = carregar_json(ARQ_ESTRUTURA)
    gabarito  = carregar_gabarito(ARQ_GABARITO)

    model = genai.GenerativeModel(MODELO)

    arquivos = sorted(glob.glob(os.path.join(PASTA_TEXTOS, "*.txt")))
    # ignora READMEs
    arquivos = [a for a in arquivos if Path(a).stem.lower() != "readme"]

    if not arquivos:
        print(f"[AVISO] Nenhum arquivo .txt em {PASTA_TEXTOS}/")
        return

    print(f"[INFO] {len(arquivos)} arquivos para analisar.\n")

    consolidado = []

    for arq in tqdm(arquivos, desc="Analisando"):
        with open(arq, "r", encoding="utf-8") as f:
            questao_txt = f.read().strip()

        if not questao_txt:
            print(f"[SKIP] {arq} vazio.")
            continue

        meta = parse_nome_arquivo(arq)
        num  = meta["numero"]
        gab_oficial = gabarito.get(num) if num is not None else None

        print(f"\n>>> {arq}  (nº {num}, ing={meta['ingles']}, esp={meta['espanhol']})  gab={gab_oficial}")

        resultado = analisar_questao(model, questao_txt, matriz, estrutura, gab_oficial, meta)
        if resultado is None:
            print(f"    [FALHA] não foi possível analisar {arq}")
            continue

        # ------- preenchimentos automáticos de identificação -------
        q = resultado.setdefault("questao", {})
        ic = q.setdefault("identificacao_caderno", {})
        ic.update(IDENTIFICACAO_CADERNO)

        iq = q.setdefault("identificacao_questao", {})
        if num is not None:
            iq["numero"] = num
        iq["ingles"]   = bool(meta.get("ingles"))
        iq["espanhol"] = bool(meta.get("espanhol"))

        cq = q.setdefault("conteudo_questao", {})
        cq["caminho_texto"] = arq.replace("\\", "/")

        # imagem correspondente
        nome_base = Path(arq).stem
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            cand = os.path.join(PASTA_IMAGENS, nome_base + ext)
            if os.path.exists(cand):
                cq["link_imagem"] = cand.replace("\\", "/")
                break

        if gab_oficial:
            cq["gabarito"] = gab_oficial

        # ------- auditoria -------
        aud = resultado.setdefault("auditoria_da_ia", {})
        aud["modelo"]   = MODELO
        aud["versao"]   = "1.0"
        aud["datetime"] = datetime.utcnow().isoformat() + "Z"
        aud.setdefault("confianca", None)
        aud.setdefault("evidencias", [])
        prov = aud.setdefault("proveniencia", {})
        prov["arquivo_questao"] = arq.replace("\\", "/")
        prov["arquivo_matriz"]  = ARQ_MATRIZ
        vh = aud.setdefault("validacao_humana", {})
        vh.setdefault("validado", None)
        vh.setdefault("data_validacao", None)

        # ------- estudante -------
        resultado["estudante"] = ESTUDANTE

        # ------- salvar individual -------
        saida_ind = os.path.join(PASTA_SAIDA, f"{nome_base}.json")
        with open(saida_ind, "w", encoding="utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False, indent=2)

        consolidado.append(resultado)

    # ------- salvar consolidado -------
    saida_cons = os.path.join(PASTA_SAIDA, "analise_consolidada.json")
    with open(saida_cons, "w", encoding="utf-8") as f:
        json.dump(consolidado, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"Total analisado : {len(consolidado)}")
    print(f"Saída individual: {PASTA_SAIDA}/<questao>.json")
    print(f"Saída consolidada: {saida_cons}")
    print("=" * 70)


if __name__ == "__main__":
    main()