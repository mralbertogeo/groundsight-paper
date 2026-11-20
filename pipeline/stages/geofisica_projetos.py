"""geofisica_projetos.py — Quais levantamentos aerogeofísicos cobrem a AOI?

Consulta o catálogo de polígonos do serviço de Aerogeofísica do
GeoSGB (FeatureServer) com o envelope da AOI_BBOX e lista, por camada,
os projetos intersectantes com todos os seus atributos (nome, código,
ano, espaçamento de linhas etc.) — a lista de compras exata para o
download dos dados numéricos (XYZ/GeoTIFF) no GeoSGB.

Execução:  docker compose run --rm pipeline python stages/geofisica_projetos.py
"""
import json
import os
import sys
from pathlib import Path

import requests

BASE = ("https://geoportal.sgb.gov.br/server/rest/services/"
        "geofisica/aerogeofisica/MapServer")
RELATORIO = Path("/data/outputs/geofisica_projetos.txt")


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


def main():
    RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)

    bbox = [float(x) for x in os.environ["AOI_BBOX"].split(",")]
    geometria = json.dumps({
        "xmin": bbox[0], "ymin": bbox[1],
        "xmax": bbox[2], "ymax": bbox[3],
        "spatialReference": {"wkid": 4326},
    })
    sess = requests.Session()
    sess.headers.update({"User-Agent": "GroundSight-geofisica/1.0"})

    servico = sess.get(BASE, params={"f": "json"}, timeout=60).json()
    camadas = servico.get("layers", [])
    print(f"===== LEVANTAMENTOS AEROGEOFÍSICOS NA AOI =====")
    print(f"AOI: {bbox}")
    print(f"Camadas no serviço: {len(camadas)}\n")

    for cam in camadas:
        cid, nome = cam["id"], cam["name"]
        try:
            r = sess.get(f"{BASE}/{cid}/query", params={
                "geometry": geometria,
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "false",
                "f": "json",
            }, timeout=120).json()
        except Exception as e:
            print(f"[{cid}] {nome}: erro na consulta ({e})")
            continue
        feats = r.get("features", [])
        if not feats:
            continue
        print(f"----- [{cid}] {nome}: {len(feats)} projeto(s) na AOI -----")
        for f in feats[:20]:
            attrs = f.get("attributes", {})
            uteis = {k: v for k, v in attrs.items()
                     if v not in (None, "", " ")}
            for k, v in uteis.items():
                print(f"      {k}: {v}")
            print("      " + "-" * 50)
    print(f"\nRelatório salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
