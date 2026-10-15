"""Grava ``recorte_cog_ndvi_20250407.tif``: a janela da área da seção 9 lida do COG original.

O notebook compara o mapa que o cliente WTSS devolve com a mesma janela lida direto da
imagem publicada pelo Brazil Data Cube. Ler o COG remoto exige rede e uma configuração
de certificado que nem todo ambiente tem, então o recorte é gravado aqui uma vez, num
GeoTIFF de poucos KB versionado junto do notebook, e o notebook lê o arquivo local.

A imagem de origem é o item STAC ``S2-16D_V2_020020_20250407`` (banda ``NDVI``), a data
de pico da área no ano agrícola 2024/25. Ela está na Albers do BDC; o recorte preserva
esse CRS, e é o notebook que reprojeta os centros de pixel do WTSS (EPSG:4326) para
sobrepô-los.

Sobre o certificado: o ``rasterio`` aponta ``GDAL_CURL_CA_BUNDLE`` para o ``cacert.pem``
do ``certifi``. No Windows, se esse caminho tiver acento (como ``Área de Trabalho``), o
schannel não consegue abri-lo e o GDAL falha com ``CURL error: schannel: invalid path
name for CA file``. A saída é copiar o ``cacert.pem`` para um caminho sem acentos e
apontar ``GDAL_CURL_CA_BUNDLE`` para lá — é o que ``preparar_ca_bundle()`` faz.

Uso, na pasta ``examples/caso1``::

    python preparar_recorte_cog.py
"""
import os
import pathlib
import shutil
import tempfile

AQUI = pathlib.Path(__file__).resolve().parent
DESTINO = AQUI / 'recorte_cog_ndvi_20250407.tif'

COG = ('/vsicurl/https://data.inpe.br/bdc/data/s2-16d/v2/020/020/2025/04/07/'
       'S2-16D_V2_020020_20250407_NDVI.tif')

# A mesma caixa da seção 9 do notebook.
LATITUDE, LONGITUDE, LADO = -13.55, -56.00, 0.004

# Folga, em pixels de 10 m, em cada lado da janela. A caixa lon/lat e a grade da Albers
# não se alinham, então a janela justa fica um pixel mais estreita que a grade que o WTSS
# devolve para a mesma caixa, e os centros de pixel da borda caem fora da imagem. Um
# pixel de folga de cada lado resolve, ao custo de alguns KB.
MARGEM = 1


def preparar_ca_bundle():
    """Garante um ``cacert.pem`` num caminho sem acentos antes de o GDAL abrir a URL.

    Precisa rodar *depois* de ``import rasterio``, que é quem define
    ``GDAL_CURL_CA_BUNDLE`` apontando para o ``certifi``.
    """
    atual = os.environ.get('GDAL_CURL_CA_BUNDLE', '')
    if atual and atual.isascii() and pathlib.Path(atual).exists():
        return atual
    import certifi
    copia = pathlib.Path(tempfile.gettempdir()) / 'cacert_bdc.pem'
    if not copia.exists():
        shutil.copy(certifi.where(), copia)
    os.environ['GDAL_CURL_CA_BUNDLE'] = str(copia)
    return str(copia)


def main():
    import rasterio
    from rasterio.windows import Window, from_bounds
    from pyproj import Transformer

    print(f'CA bundle: {preparar_ca_bundle()}')

    with rasterio.open(COG) as origem:
        print(f'COG  : {origem.width}x{origem.height}, {origem.crs.to_string()[:40]}...')

        para_albers = Transformer.from_crs('EPSG:4326', origem.crs, always_xy=True)
        cantos = [para_albers.transform(LONGITUDE + dx * LADO / 2, LATITUDE + dy * LADO / 2)
                  for dx in (-1, 1) for dy in (-1, 1)]
        xs = [x for x, _ in cantos]
        ys = [y for _, y in cantos]

        janela = from_bounds(min(xs), min(ys), max(xs), max(ys), origem.transform)
        janela = janela.round_offsets().round_lengths()
        janela = Window(janela.col_off - MARGEM, janela.row_off - MARGEM,
                        janela.width + 2 * MARGEM, janela.height + 2 * MARGEM)
        recorte = origem.read(1, window=janela)
        perfil = origem.profile | {
            'height': recorte.shape[0], 'width': recorte.shape[1],
            'transform': origem.window_transform(janela),
            'compress': 'deflate', 'tiled': False,
        }

    with rasterio.open(DESTINO, 'w', **perfil) as saida:
        saida.write(recorte, 1)
        saida.set_band_description(1, '2025-04-07 NDVI')

    print(f'{DESTINO.name}: {recorte.shape[1]}x{recorte.shape[0]} pixels, '
          f'{DESTINO.stat().st_size / 1024:.0f} KB')


if __name__ == '__main__':
    main()
