"""s05c_inspecao.py — Inspeção dos dados de geofísica baixados.

Sem carregar nada pesado: lê apenas cabeçalhos e amostras.
  A) Para cada .XYZ em data/raw/geofisica/**: primeiras ~40 linhas
     (formato Geosoft: linhas de comentário '/', marcadores 'Line',
     nomes de canais) + contagem de colunas de uma linha de dados;
  B) O(s) .txt do 1039 (dicionário de canais) na íntegra (limitado);
  C) Cada .tif: bandas, tipo de dado, faixa de valores de uma janela —
     responde se é dado físico (float, 1 banda) ou imagem estilizada
     (8-bit, 3 bandas).

Relatório em tela + txt.

Execução:  docker compose run --rm pipeline python stages/s05c_inspecao.py
"""
import sys
from pathlib import Path

RAIZ = Path("/data/raw/geofisica")
RELATORIO = Path("/data/outputs/s05c_inspecao.txt")


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


def main():
    import numpy as np
    RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)

    print("===== INSPEÇÃO GEOFÍSICA =====")
    if not RAIZ.exists():
        print(f"Pasta ausente: {RAIZ}")
        return

    # ---------- A) XYZ ----------
    for xyz in sorted(RAIZ.rglob("*.xyz")) + sorted(RAIZ.rglob("*.XYZ")):
        mb = xyz.stat().st_size / 1e6
        print(f"\n########## {xyz.relative_to(RAIZ)} ({mb:.0f} MB) ##########")
        MAX_LINHAS = 60          # limite RÍGIDO de impressão por arquivo
        with open(xyz, "r", encoding="latin-1", errors="replace") as f:
            for i, linha in enumerate(f):
                if i >= MAX_LINHAS:
                    print(f"      ... [interrompido no limite de "
                          f"{MAX_LINHAS} linhas]")
                    break
                mostra = linha.rstrip()[:150]
                print(f"{i:4d}| {mostra} "
                      f"[{len(linha.split())} tokens]")

    # ---------- B) TXT (dicionário de canais) ----------
    for txt in sorted(RAIZ.rglob("*.txt")) + sorted(RAIZ.rglob("*.TXT")):
        print(f"\n########## {txt.relative_to(RAIZ)} ##########")
        conteudo = txt.read_text(encoding="latin-1", errors="replace")
        print(conteudo[:6000])
        if len(conteudo) > 6000:
            print(f"... [truncado — {len(conteudo)} caracteres]")

    # ---------- C) TIFs ----------
    try:
        import rasterio
        for tif in sorted(RAIZ.rglob("*.tif")) + sorted(RAIZ.rglob("*.TIF")):
            print(f"\n########## {tif.relative_to(RAIZ)} ##########")
            try:
                with rasterio.open(tif) as src:
                    print(f"bandas={src.count} dtype={src.dtypes[0]} "
                          f"crs={src.crs} shape={src.shape}")
                    jan = src.read(1, window=((0, min(200, src.height)),
                                              (0, min(200, src.width))))
                    print(f"amostra banda 1: min={np.nanmin(jan):.2f} "
                          f"max={np.nanmax(jan):.2f}")
                    veredito = ("DADO FÍSICO provável"
                                if src.count == 1 and "float" in src.dtypes[0]
                                else "IMAGEM ESTILIZADA provável "
                                     "(usar XYZ como fonte)")
                    print(f"-> {veredito}")
            except Exception as e:
                print(f"erro ao abrir: {e}")
    except ImportError:
        print("rasterio indisponível para checar TIFs")

    print(f"\nRelatório salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
