"""estrutural_download.py — Baixa as estruturas geológicas (GIS Brasil
ao Milionésimo, 2004) da AOI direto do geoportal SGB e grava o
shapefile em data/raw/estrutural — pronto para o s05b.

Consulta sem parâmetros de paginação (o servidor da camada é antigo e
os rejeita); tenta GeoJSON e cai para Esri JSON se preciso.

Execução:  docker compose run --rm pipeline python stages/estrutural_download.py
"""
import json
import os
import sys
from pathlib import Path

import requests

CAMADA = ("https://geoportal.sgb.gov.br/server/rest/services/"
          "geologia/Estruturas_GIS_Brasil_2004/MapServer/0")
DESTINO = Path("/data/raw/estrutural")
RELATORIO = Path("/data/outputs/estrutural_download.txt")
BUFFER = 0.1


class Tee:
    def __init__(self, caminho):
        self.terminal = sys.stdout
        self.arquivo = open(caminho, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg); self.arquivo.write(msg)

    def flush(self):
        self.terminal.flush(); self.arquivo.flush()


def esri_para_geodf(dados):
    """Converte Esri JSON (polylines) em GeoDataFrame."""
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString

    wkid = (dados.get("spatialReference", {}) or {}).get("wkid", 4326)
    geoms, attrs = [], []
    for f in dados.get("features", []):
        paths = (f.get("geometry") or {}).get("paths", [])
        if not paths:
            continue
        linhas = [LineString(p) for p in paths if len(p) >= 2]
        if not linhas:
            continue
        geoms.append(linhas[0] if len(linhas) == 1
                     else MultiLineString(linhas))
        attrs.append(f.get("attributes", {}))
    epsg = 4674 if wkid in (4674,) else 4326
    return gpd.GeoDataFrame(attrs, geometry=geoms, crs=f"EPSG:{epsg}")


def main():
    import geopandas as gpd

    RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RELATORIO)

    bbox = [float(x) for x in os.environ["AOI_BBOX"].split(",")]
    envelope = json.dumps({
        "xmin": bbox[0] - BUFFER, "ymin": bbox[1] - BUFFER,
        "xmax": bbox[2] + BUFFER, "ymax": bbox[3] + BUFFER,
        "spatialReference": {"wkid": 4326}})
    sess = requests.Session()
    sess.headers.update({"User-Agent": "GroundSight-estrutural/1.0"})

    print("===== DOWNLOAD — Estruturas GIS Brasil 2004 =====")
    print(f"AOI (+buffer {BUFFER}°): {bbox}")

    base_params = {
        "geometry": envelope,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
    }

    g = None
    try:
        r = sess.get(f"{CAMADA}/query",
                     params={**base_params, "f": "geojson"},
                     timeout=180).json()
        if "features" in r and r.get("type") == "FeatureCollection":
            g = gpd.GeoDataFrame.from_features(r["features"],
                                               crs="EPSG:4326")
            print("Formato: GeoJSON")
    except Exception as e:
        print(f"GeoJSON indisponível ({e}) — tentando Esri JSON")

    if g is None or not len(g):
        r = sess.get(f"{CAMADA}/query",
                     params={**base_params, "f": "json"},
                     timeout=180).json()
        if "error" in r:
            raise RuntimeError(f"Servidor recusou a consulta: {r['error']}")
        g = esri_para_geodf(r)
        print("Formato: Esri JSON")
        if r.get("exceededTransferLimit"):
            print("AVISO: limite de transferência excedido — parte das "
                  "feições pode ter ficado de fora (reportar).")

    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    print(f"Feições estruturais na AOI: {len(g)}")
    if not len(g):
        print("Nenhuma feição — verifique AOI_BBOX.")
        sys.exit(1)

    campos = [c for c in g.columns if c != "geometry"]
    print(f"Atributos: {campos}")
    for c in campos:
        vals = g[c].dropna().unique()
        if 0 < len(vals) <= 12:
            print(f"   {c}: {sorted(map(str, vals))}")

    saida = DESTINO / "estruturas_gis_brasil_2004.shp"
    g.to_file(saida)
    print(f"\nShapefile gravado: {saida}")
    print("Pronto para o s05b (rota SGB).")
    print(f"Relatório salvo em: {RELATORIO}")
    sys.stdout.arquivo.close()
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    main()
