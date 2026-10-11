#!/usr/bin/env python3
"""
FloodIntel reference comparison - Trishuli / Bhote Koshi, August 2026 event.

Compares a FloodIntel prediction with two UNVALIDATED third-party reference
maps (IWM VAP02, TASA) on a common 10 m grid in WGS 84 / UTM 45N.

Outputs are "agreement with reference maps", NOT verified accuracy:
  * neither reference is officially validated;
  * both map the wider flood-affected corridor (water + debris + river bed),
    not open floodwater alone;
  * both include the normal river channel unless a permanent-water mask is given.

Requires only numpy and Pillow for vector inputs. A GeoTIFF prediction or mask
additionally needs rasterio (that code path was not tested in the sandbox).

Usage
  python floodintel_validate.py --refs-only            # reference statistics only
  python floodintel_validate.py --pred my_flood.geojson \
      [--valid valid_mask.tif|.geojson] [--perm-water gsw_mask.tif|.geojson] \
      [--res 10] [--buffer-m 500] [--tolerance-m 20] [--out results/]
"""
import argparse, hashlib, json, math, os, struct, sys, datetime
import numpy as np
from PIL import Image, ImageDraw

# ------------------------------------------------------------------ paths
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT = {
    'iwm':    os.path.join(ROOT, 'data', 'validation_only', 'iwm', 'IWM_2026_08_26_Nepal_Flood_VAP02.shp')
              if os.path.exists(os.path.join(ROOT, 'data', 'validation_only', 'iwm', 'IWM_2026_08_26_Nepal_Flood_VAP02.shp'))
              else os.path.join(HERE, 'reference', 'IWM_2026_08_26_Nepal_Flood_VAP02.shp'),
    'tasa':   os.path.join(ROOT, 'data', 'validation_only', 'tasa', 'Affected_0816_0828.shp')
              if os.path.exists(os.path.join(ROOT, 'data', 'validation_only', 'tasa', 'Affected_0816_0828.shp'))
              else os.path.join(HERE, 'reference', 'Affected_0816_0828.shp'),
    'hot':    os.path.join(ROOT, 'data', 'validation_only', 'hot_flood_npl', 'hot_flood_npl_flood_extent_2026-08-27.geojson')
              if os.path.exists(os.path.join(ROOT, 'data', 'validation_only', 'hot_flood_npl', 'hot_flood_npl_flood_extent_2026-08-27.geojson'))
              else os.path.join(HERE, 'reference', 'hot_flood_npl_flood_extent.geojson'),
    'nepal':  os.path.join(ROOT, 'data', 'validation_only', 'boundaries', 'geoBoundaries-NPL-ADM0.geojson')
              if os.path.exists(os.path.join(ROOT, 'data', 'validation_only', 'boundaries', 'geoBoundaries-NPL-ADM0.geojson'))
              else os.path.join(HERE, 'reference', 'geoBoundaries-NPL-ADM0.geojson'),
}

# ------------------------------------------------------------------ UTM 45N (WGS84)
A_ = 6378137.0; F_ = 1 / 298.257223563; E2 = F_ * (2 - F_); K0 = 0.9996
LON0 = math.radians(87.0); EP2 = E2 / (1 - E2)
E1 = (1 - math.sqrt(1 - E2)) / (1 + math.sqrt(1 - E2))

def ll_to_utm(lon, lat):
    p = math.radians(lat); l = math.radians(lon)
    N = A_ / math.sqrt(1 - E2 * math.sin(p) ** 2); T = math.tan(p) ** 2
    C = EP2 * math.cos(p) ** 2; A = (l - LON0) * math.cos(p)
    M = A_ * ((1 - E2 / 4 - 3 * E2 ** 2 / 64 - 5 * E2 ** 3 / 256) * p
              - (3 * E2 / 8 + 3 * E2 ** 2 / 32 + 45 * E2 ** 3 / 1024) * math.sin(2 * p)
              + (15 * E2 ** 2 / 256 + 45 * E2 ** 3 / 1024) * math.sin(4 * p)
              - (35 * E2 ** 3 / 3072) * math.sin(6 * p))
    x = K0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T ** 2 + 72 * C - 58 * EP2) * A ** 5 / 120) + 500000
    y = K0 * (M + N * math.tan(p) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                                      + (61 - 58 * T + T ** 2 + 600 * C - 330 * EP2) * A ** 6 / 720))
    return x, y

def utm_to_ll(x, y):
    x -= 500000.0
    mu = (y / K0) / (A_ * (1 - E2 / 4 - 3 * E2 ** 2 / 64 - 5 * E2 ** 3 / 256))
    p1 = (mu + (3 * E1 / 2 - 27 * E1 ** 3 / 32) * math.sin(2 * mu) + (21 * E1 ** 2 / 16 - 55 * E1 ** 4 / 32) * math.sin(4 * mu)
          + (151 * E1 ** 3 / 96) * math.sin(6 * mu) + (1097 * E1 ** 4 / 512) * math.sin(8 * mu))
    C1 = EP2 * math.cos(p1) ** 2; T1 = math.tan(p1) ** 2; N1 = A_ / math.sqrt(1 - E2 * math.sin(p1) ** 2)
    R1 = A_ * (1 - E2) / (1 - E2 * math.sin(p1) ** 2) ** 1.5; D = x / (N1 * K0)
    lat = p1 - (N1 * math.tan(p1) / R1) * (D ** 2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * EP2) * D ** 4 / 24
                                          + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * EP2 - 3 * C1 ** 2) * D ** 6 / 720)
    lon = LON0 + (D - (1 + 2 * T1 + C1) * D ** 3 / 6 + (5 - 2 * C1 + 28 * T1 + 24 * T1 ** 2 + 8 * EP2 + 3 * C1 ** 2) * D ** 5 / 120) / math.cos(p1)
    return math.degrees(lon), math.degrees(lat)

# ------------------------------------------------------------------ readers -> list of "polygons", each a list of rings in UTM 45N metres
def _crs_of_prj(prj_path):
    if not os.path.exists(prj_path):
        raise SystemExit(f'Missing .prj for {prj_path} - CRS unknown, refusing to guess.')
    t = open(prj_path).read()
    if 'UTM_Zone_45N' in t.replace(' ', '_') or 'UTM zone 45N' in t:
        return 'utm45'
    if t.startswith('GEOGCS') and 'WGS_1984' in t:
        return 'll'
    raise SystemExit(f'Unsupported CRS in {prj_path}: {t[:120]}... Reproject to EPSG:4326 or EPSG:32645 first.')

def read_shp(path):
    crs = _crs_of_prj(path[:-4] + '.prj')
    b = open(path, 'rb').read()
    stype = struct.unpack('<i', b[32:36])[0]
    if stype not in (5, 15, 25):
        raise SystemExit(f'{path}: shape type {stype} is not a polygon layer')
    flen = struct.unpack('>i', b[24:28])[0] * 2; pos = 100; polys = []
    while pos < flen:
        _, cl = struct.unpack('>ii', b[pos:pos + 8]); c = b[pos + 8:pos + 8 + cl * 2]; pos += 8 + cl * 2
        if struct.unpack('<i', c[:4])[0] == 0: continue
        n_parts, n_pts = struct.unpack('<ii', c[36:44])
        parts = list(struct.unpack(f'<{n_parts}i', c[44:44 + 4 * n_parts])); po = 44 + 4 * n_parts
        pts = [struct.unpack('<2d', c[po + 16 * i: po + 16 * i + 16]) for i in range(n_pts)]
        rings = [pts[parts[i]:(parts[i + 1] if i + 1 < n_parts else n_pts)] for i in range(n_parts)]
        if crs == 'll': rings = [[ll_to_utm(x, y) for x, y in r] for r in rings]
        polys.append(rings)          # one record = rings combined by even-odd rule
    return polys

def read_geojson(path):
    j = json.load(open(path))
    crs = (j.get('crs') or {}).get('properties', {}).get('name', 'CRS84')
    if '32645' in crs: conv = lambda r: [tuple(p[:2]) for p in r]
    elif any(k in crs for k in ('CRS84', '4326')): conv = lambda r: [ll_to_utm(p[0], p[1]) for p in r]
    else: raise SystemExit(f'{path}: unsupported CRS {crs}; use EPSG:4326 or EPSG:32645')
    feats = j['features'] if j.get('type') == 'FeatureCollection' else [j]
    polys = []
    for f in feats:
        g = f['geometry']
        if g is None: continue
        if g['type'] == 'Polygon': polys.append([conv(r) for r in g['coordinates']])
        elif g['type'] == 'MultiPolygon': polys += [[conv(r) for r in p] for p in g['coordinates']]
        else: raise SystemExit(f'{path}: geometry type {g["type"]} is not polygonal')
    return polys

def read_vector(path):
    if path.lower().endswith('.shp'): return read_shp(path)
    if path.lower().endswith(('.geojson', '.json')): return read_geojson(path)
    raise SystemExit(f'Unsupported vector format: {path}')

# ------------------------------------------------------------------ grid
class Grid:
    def __init__(self, x0, y0, x1, y1, res):
        self.res = res
        self.x0 = math.floor(x0 / res) * res; self.y1 = math.ceil(y1 / res) * res
        self.W = int(math.ceil((x1 - self.x0) / res)); self.H = int(math.ceil((self.y1 - y0) / res))
        self.cell_m2 = res * res
    def burn(self, polys):
        """Even-odd rasterisation per polygon, union across distinct polygons.
        Holes within a polygon are handled by even-odd (XOR) rule among its rings.
        Distinct polygon interiors are combined via logical OR."""
        out = np.zeros((self.H, self.W), bool)
        for rings in polys:
            if not rings:
                continue
            poly_xs = [(x - self.x0) / self.res for r in rings for x, _ in r]
            poly_ys = [(self.y1 - y) / self.res for r in rings for _, y in r]
            if not poly_xs or not poly_ys:
                continue
            c0 = max(int(math.floor(min(poly_xs))) - 1, 0)
            c1 = min(int(math.ceil(max(poly_xs))) + 1, self.W)
            r0 = max(int(math.floor(min(poly_ys))) - 1, 0)
            r1 = min(int(math.ceil(max(poly_ys))) + 1, self.H)
            if c1 <= c0 or r1 <= r0:
                continue
            poly_mask = np.zeros((r1 - r0, c1 - c0), bool)
            for r in rings:
                xs = [(x - self.x0) / self.res for x, _ in r]
                ys = [(self.y1 - y) / self.res for _, y in r]
                im = Image.new('1', (c1 - c0, r1 - r0), 0)
                ImageDraw.Draw(im).polygon([(x - c0 - .5, y - r0 - .5) for x, y in zip(xs, ys)], fill=1, outline=None)
                poly_mask ^= np.array(im, bool)
            out[r0:r1, c0:c1] |= poly_mask
        return out
    def area_km2(self, m): return float(m.sum()) * self.cell_m2 / 1e6

def read_raster_mask(path, grid, true_values, min_value=None):
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_origin
    except ImportError:
        raise SystemExit('A GeoTIFF input needs rasterio (pip install rasterio), or convert it to GeoJSON polygons first.')
    with rasterio.open(path) as src:
        dst = np.zeros((grid.H, grid.W), np.float32)
        reproject(source=rasterio.band(src, 1), destination=dst,
                  dst_transform=from_origin(grid.x0, grid.y1, grid.res, grid.res), dst_crs='EPSG:32645',
                  resampling=Resampling.nearest, src_nodata=src.nodata, dst_nodata=np.nan)
    if min_value is not None: return dst >= min_value
    return np.isin(dst, true_values) if true_values is not None else (dst > 0)

def load_mask(path, grid, true_values=None, min_value=None):
    if path.lower().endswith(('.tif', '.tiff')): return read_raster_mask(path, grid, true_values, min_value)
    return grid.burn(read_vector(path))

def dilate(m, px):
    """Square dilation by px cells (fast, separable)."""
    if px <= 0: return m.copy()
    out = m.copy()
    for _ in range(px):
        n = out.copy(); n[1:] |= out[:-1]; n[:-1] |= out[1:]; out = n
    tmp = out.copy()
    for _ in range(px):
        n = tmp.copy(); n[:, 1:] |= tmp[:, :-1]; n[:, :-1] |= tmp[:, 1:]; tmp = n
    return tmp

def scores(pred, ref, domain, g, tol_px=0):
    p, r = pred & domain, ref & domain
    tp, fp, fn = p & r, p & ~r, r & ~p
    TP, FP, FN = (g.area_km2(x) for x in (tp, fp, fn))
    out = dict(TP_km2=TP, FP_km2=FP, FN_km2=FN,
               IoU=TP / (TP + FP + FN) if TP + FP + FN else None,
               precision=TP / (TP + FP) if TP + FP else None,
               recall=TP / (TP + FN) if TP + FN else None,
               F1=2 * TP / (2 * TP + FP + FN) if TP + FP + FN else None)
    if tol_px:
        rd, pd = dilate(r, tol_px), dilate(p, tol_px)
        out['precision_tolerant'] = g.area_km2(p & rd) / g.area_km2(p) if p.any() else None
        out['recall_tolerant'] = g.area_km2(r & pd) / g.area_km2(r) if r.any() else None
    return out

def recall_only(pred, ref, domain, g):
    r = ref & domain
    return dict(ref_km2=g.area_km2(r), detected_km2=g.area_km2(pred & r),
                recall=g.area_km2(pred & r) / g.area_km2(r) if r.any() else None)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', help='FloodIntel flood mask: .geojson/.shp polygons or GeoTIFF (value>0 = flood)')
    ap.add_argument('--pred-values', type=float, nargs='*', help='GeoTIFF pixel values meaning flood (default: >0)')
    ap.add_argument('--valid', help='Mask of pixels FloodIntel could actually observe (cloud-free, not no-data)')
    ap.add_argument('--perm-water', help='Normal / permanent river water mask to remove from all layers')
    ap.add_argument('--perm-water-min', type=float, help='GeoTIFF threshold, e.g. 50 for JRC GSW occurrence >= 50%%')
    ap.add_argument('--iwm', default=DEFAULT['iwm']); ap.add_argument('--tasa', default=DEFAULT['tasa'])
    ap.add_argument('--hot', default=DEFAULT['hot']); ap.add_argument('--nepal', default=DEFAULT['nepal'])
    ap.add_argument('--res', type=float, default=10.0, help='grid cell size in metres (default 10)')
    ap.add_argument('--buffer-m', type=float, default=500.0, help='evaluation zone = reference corridor + this buffer')
    ap.add_argument('--tolerance-m', type=float, default=20.0, help='edge tolerance for tolerant precision/recall')
    ap.add_argument('--refs-only', action='store_true', help='report reference statistics without a prediction')
    ap.add_argument('--self-test', action='store_true', help='sanity check: IWM vs itself must give IoU = 1')
    ap.add_argument('--out', default='validation_results')
    a = ap.parse_args()
    if not (a.pred or a.refs_only or a.self_test): ap.error('give --pred, --refs-only or --self-test')
    os.makedirs(a.out, exist_ok=True)

    iwm, tasa = read_vector(a.iwm), read_vector(a.tasa)
    nepal = None if a.nepal.lower() == 'none' else read_vector(a.nepal)
    hot = read_vector(a.hot) if a.hot and os.path.exists(a.hot) else None
    xs = [x for p in iwm + tasa for r in p for x, _ in r]; ys = [y for p in iwm + tasa for r in p for _, y in r]
    pad = a.buffer_m + 1000
    g = Grid(min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad, a.res)
    print(f'grid: {g.W} x {g.H} cells of {a.res:g} m, UTM 45N, origin ({g.x0:.0f}, {g.y1:.0f})')

    NP = g.burn(nepal) if nepal else np.ones((g.H, g.W), bool); IWM = g.burn(iwm); TASA = g.burn(tasa)
    HOT = g.burn(hot) if hot else None
    PW = load_mask(a.perm_water, g, min_value=a.perm_water_min) if a.perm_water else None
    VALID = load_mask(a.valid, g) if a.valid else None
    notes = []
    if nepal is None: notes.append('Border clip OFF (--nepal none): areas on the China side of the border river are included.')
    if PW is None: notes.append('No permanent-water mask given: the normal river channel is counted as flood-affected in every layer.')
    if VALID is None: notes.append('No FloodIntel valid-observation mask given: cloudy / no-data pixels are treated as observed (results may be pessimistic).')

    keep = NP.copy()
    if PW is not None: keep &= ~PW
    refs = dict(IWM=IWM & keep, TASA=TASA & keep, CORE=IWM & TASA & keep)
    zone = dilate(IWM | TASA, int(round(a.buffer_m / a.res))) & keep
    if VALID is not None: zone &= VALID
    tol = int(round(a.tolerance_m / a.res))

    rep = dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               label='Agreement with unvalidated third-party reference maps - NOT verified accuracy',
               grid=dict(crs='EPSG:32645', res_m=a.res, W=g.W, H=g.H, x0=g.x0, y1=g.y1),
               inputs={k: dict(path=v, sha256=sha256(v)) for k, v in
                       dict(iwm=a.iwm, tasa=a.tasa, nepal=a.nepal if nepal else None, hot=a.hot if hot else None,
                            pred=a.pred, valid=a.valid, perm_water=a.perm_water).items() if v},
               notes=notes)
    # reference statistics (always)
    rep['reference_stats'] = dict(
        IWM_total_km2=g.area_km2(IWM), IWM_in_Nepal_km2=g.area_km2(IWM & NP),
        TASA_total_km2=g.area_km2(TASA), TASA_in_Nepal_km2=g.area_km2(TASA & NP),
        CORE_IWM_and_TASA_in_Nepal_km2=g.area_km2(IWM & TASA & NP),
        evaluation_zone_km2=g.area_km2(zone),
        TASA_vs_IWM_in_Nepal=scores(TASA & NP, IWM & NP, NP, g),
        HOT_vs_IWM_in_Nepal_same_window=scores(HOT & NP, IWM & NP, dilate(IWM, int(round(a.buffer_m / a.res))) & NP, g) if HOT is not None else None)

    pred = None
    if a.self_test:
        pred = IWM.copy(); rep['self_test'] = True
        notes.append('SELF-TEST: prediction = IWM itself. Expected IoU vs IWM = 1.0. These are NOT FloodIntel results.')
    elif a.pred:
        pred = load_mask(a.pred, g, a.pred_values)
    if pred is not None:
        pred = pred & keep
        rep['A_vs_IWM_full_metrics'] = scores(pred, refs['IWM'], zone, g, tol)
        rep['B_recall_vs_IWM_and_TASA_core'] = recall_only(pred, refs['CORE'], zone, g)
        rep['C_recall_vs_TASA_only'] = recall_only(pred, refs['TASA'], zone, g)
        if HOT is not None:
            rep['D_sensitivity_vs_HOT'] = scores(pred, HOT & keep, zone & dilate(HOT, int(round(a.buffer_m / a.res))), g, tol)
        rep['prediction_in_zone_km2'] = g.area_km2(pred & zone)
        rep['prediction_in_Nepal_outside_zone_km2'] = g.area_km2(pred & keep & ~zone)
        # colour map for visual checking: green both, red pred only, blue IWM only, grey zone
        img = np.full((g.H, g.W, 3), 255, np.uint8)
        img[zone] = (238, 238, 238); img[refs['IWM'] & zone & ~pred] = (70, 120, 255)
        img[pred & zone & ~refs['IWM']] = (230, 40, 30); img[pred & refs['IWM'] & zone] = (0, 150, 70)
        Image.fromarray(img).save(os.path.join(a.out, 'agreement_map.png'))

    with open(os.path.join(a.out, 'validation_results.json'), 'w') as f: json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))

if __name__ == '__main__':
    main()
