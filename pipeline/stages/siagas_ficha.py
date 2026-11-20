"""siagas_ficha.py — Download e parsing da ficha técnica SIAGAS.

Adaptado do siagas_scraper.py do SIAGAS Explorer (GeoInovações).
Mudanças em relação ao original:
  - Cache movido para /data/raw/cache_siagas (volume Docker persistente);
  - Sanitização de caracteres de controle antes da carga no banco
    (mesmo problema dos caracteres ilegais do Excel, resolvido na origem);
  - Extração tolerante dos campos do teste de bombeamento (rótulos do
    SIAGAS variam entre fichas — busca por palavra-chave).
"""
import re
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

CACHE_DIR = Path("/data/raw/cache_siagas")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

URL_BASE = "https://siagasweb.sgb.gov.br/layout/pdf/exportar_pdf.php"

# Caracteres de controle que corrompem texto no banco/exportações
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def limpar(valor):
    """Remove caracteres de controle de strings (sanitização na origem)."""
    if isinstance(valor, str):
        return _CTRL_RE.sub("", valor).strip()
    return valor


def buscar_html(codigo: str, force_refresh: bool = False,
                timeout: int = 60) -> Optional[str]:
    """Baixa o HTML da ficha SIAGAS com cache em /data/raw/cache_siagas."""
    codigo = str(codigo).strip()
    cache_file = CACHE_DIR / f"{codigo}.html"

    if cache_file.exists() and not force_refresh:
        return cache_file.read_text(encoding="iso-8859-1")

    url = f"{URL_BASE}?ponto={codigo}"
    try:
        resp = requests.get(url, timeout=timeout,
                            headers={"User-Agent": "Mozilla/5.0 GroundSight-s01"})
        if resp.status_code != 200 or len(resp.content) < 1000:
            return None
        cache_file.write_bytes(resp.content)
        return resp.content.decode("iso-8859-1")
    except Exception:
        return None


def _txt(td) -> str:
    return td.get_text(strip=True) if td else ""


def _num(s):
    if s is None:
        return None
    s = str(s).replace(",", ".").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parsear_html(html: str) -> dict:
    """Extrai dados estruturados das tabelas da ficha SIAGAS.

    Idêntico em lógica ao SIAGAS Explorer; retorna filtros, revestimento,
    litologia, formações, aquífero e teste de bombeamento.
    """
    soup = BeautifulSoup(html, "html.parser")
    dados = {
        "codigo": None, "uf": None, "municipio": None, "localidade": None,
        "filtros": [], "revestimento": [], "litologia": [], "formacoes": [],
        "entradas_dagua": [],
        "aquifero": {}, "teste_bombeamento": {}, "outras_secoes": {},
    }

    for td in soup.find_all("td"):
        t = td.get_text(strip=True)
        m = re.match(r"Po[çc]o:\s*(\d+)", t)
        if m: dados["codigo"] = m.group(1)
        m = re.match(r"UF:\s*(\w+)", t)
        if m: dados["uf"] = m.group(1)
        m = re.match(r"Munic[ií]pio:\s*(.+)", t)
        if m: dados["municipio"] = limpar(m.group(1))
        m = re.match(r"Localidade:\s*(.+)", t)
        if m: dados["localidade"] = limpar(m.group(1))

    for tabela in soup.find_all("table"):
        header_td = tabela.find("td", class_="subsection-header")
        if not header_td:
            continue
        secao = header_td.get_text(strip=True).rstrip(":").strip().lower()

        rows = tabela.find_all("tr")
        registros = []
        labels_atuais = None
        for r in rows:
            tds = r.find_all("td")
            classes = [td.get("class") or [] for td in tds]
            if any("label-cell" in cl for cl in classes):
                labels_atuais = [_txt(td).rstrip(":").strip()
                                 for td, cl in zip(tds, classes)
                                 if "label-cell" in cl]
            elif any("data-cell" in cl for cl in classes) and labels_atuais:
                valores = [limpar(_txt(td)) for td, cl in zip(tds, classes)
                           if "data-cell" in cl]
                if any(valores):
                    registros.append(dict(zip(labels_atuais, valores)))

        if not registros:
            continue

        if secao.startswith("filtro"):
            for r in registros:
                t, b = _num(r.get("De (m)")), _num(r.get("Até (m)"))
                if t is not None and b is not None and b > t:
                    dados["filtros"].append({"topo": t, "base": b})
        elif "revestimento" in secao:
            for r in registros:
                t, b = _num(r.get("De (m)")), _num(r.get("Até (m)"))
                if t is not None and b is not None and b > t:
                    dados["revestimento"].append({"topo": t, "base": b})
        elif "entrada" in secao:
            for r in registros:
                p = _num_toler(r.get("Profundidade"))
                if p is not None and p > 0:
                    dados["entradas_dagua"].append(p)
        elif "litol" in secao:
            for r in registros:
                t, b = _num(r.get("De (m)")), _num(r.get("Até (m)"))
                if t is not None and b is not None and b > t:
                    dados["litologia"].append({
                        "topo": t, "base": b,
                        "descricao": " — ".join(
                            x for x in [r.get("Litologia", ""),
                                        r.get("Descrição Litológica", "")] if x),
                    })
        elif "formação" in secao or "formacao" in secao:
            for r in registros:
                t = _num(r.get("Profundidade Inicial (m)"))
                b = _num(r.get("Profundidade Final (m)"))
                if t is not None and b is not None and b > t:
                    dados["formacoes"].append({
                        "topo": t, "base": b,
                        "formacao": r.get("Tipo de Formação", "")})
        elif "aquífero" in secao or "aquifero" in secao:
            for r in registros:
                dados["aquifero"].update(r)
        elif "teste" in secao or "bombeamento" in secao:
            for r in registros:
                dados["teste_bombeamento"].update(r)
        else:
            dados["outras_secoes"][secao] = registros

    return dados


_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _num_toler(s) -> Optional[float]:
    """Extrai o primeiro número de um texto ('4,00 m3/h' -> 4.0)."""
    if s is None:
        return None
    m = _NUM_RE.search(str(s))
    if not m:
        return None
    return float(m.group(0).replace(",", "."))


def _valor_por_chave(d: dict, *palavras) -> Optional[float]:
    """Busca tolerante: retorna o primeiro valor numérico cuja chave
    contenha TODAS as palavras informadas (case/acento-insensível)."""
    def norm(s):
        s = s.lower()
        for a, b in [("á","a"),("à","a"),("ã","a"),("â","a"),("é","e"),
                     ("ê","e"),("í","i"),("ó","o"),("ô","o"),("õ","o"),
                     ("ú","u"),("ç","c")]:
            s = s.replace(a, b)
        return s
    alvo = [norm(p) for p in palavras]
    for k, v in d.items():
        kn = norm(str(k))
        if all(p in kn for p in alvo):
            n = _num_toler(v)
            if n is not None:
                return n
    return None


def extrair_teste(dados: dict) -> dict:
    """Extrai vazão, NE, ND e tempo do teste de bombeamento da ficha.

    Os rótulos variam entre fichas do SIAGAS; a busca é por palavra-chave.
    Retorna dict com chaves vazao_m3h, ne_m, nd_m, tempo_h (None se ausente).
    """
    tb = dados.get("teste_bombeamento", {}) or {}
    return {
        "vazao_m3h": (_valor_por_chave(tb, "vazao", "estab")
                      or _valor_por_chave(tb, "vazao", "livre")),
        "qs_ficha": _valor_por_chave(tb, "vazao", "espec"),
        "ne_m": (_valor_por_chave(tb, "estatico")
                 or _valor_por_chave(tb, "n.e")),
        "nd_m": (_valor_por_chave(tb, "dinamico")
                 or _valor_por_chave(tb, "n.d")),
        "tempo_h": (_valor_por_chave(tb, "duracao")
                    or _valor_por_chave(tb, "tempo", "bombeamento")),
    }
