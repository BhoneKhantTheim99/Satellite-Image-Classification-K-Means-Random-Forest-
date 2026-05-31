"""
Satellite Image Classification App — Raw Sentinel-2 Pipeline
=============================================================
Pipeline:
  1. Upload raw 5-band GeoTIFF (B2, B3, B4, B8, B11) from GEE export
  2. Reproject to EPSG:32647 (UTM 47N) at 10 m; bilinear-resample B11 SWIR 20m→10m
  3. Divide by 10 000 → reflectance [0–1]
  4. Compute 9 spectral indices (NDVI, EVI, SAVI, NDWI, BSI, MBI, NDBI, CMI, FCI)
  5. Feature selection → StandardScaler
  6. K-Means (k=4, subsample 50k) → 3-Class RF → 4-Class Mine RF
  7. Multi-year change analysis
"""

import streamlit as st
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score as compute_silhouette,
                             confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import io, zipfile, tempfile, os, warnings, pickle
warnings.filterwarnings("ignore")

try:
    import geopandas as gpd
    from rasterio.features import rasterize
    from shapely.geometry import mapping
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CLASSES       = ['Forest', 'Sparse Vegetation', 'Bare Soil']
CLASS_COLORS_3 = {'Forest': [34,85,34], 'Sparse Vegetation': [255,191,0], 'Bare Soil': [139,69,19]}
CLASS_COLORS_4 = {'Forest': [0,100,0],  'Sparse Vegetation': [255,180,0], 'Bare Soil': [200,0,0], 'Mine': [0,80,200]}
CLASS_TO_INT_3 = {'Forest': 1, 'Sparse Vegetation': 2, 'Bare Soil': 3}
CLASS_TO_INT_4 = {'Forest': 1, 'Sparse Vegetation': 2, 'Bare Soil': 3, 'Mine': 4}
CLASSES_4      = list(CLASS_TO_INT_4.keys())

KNOWN_BANDS = ["B2_Blue","B3_Green","B4_Red","B8_NIR","B11_SWIR_10m","Other"]
SWIR_NAMES  = {"B11_SWIR_10m","B11_SWIR","B11"}
TARGET_CRS  = CRS.from_epsg(32647)
TARGET_RES  = 10.0   # metres

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  .main-header{font-size:2.5rem;font-weight:800;color:#00E676;margin-bottom:.2rem}
  .sub-header {font-size:1.1rem;color:#E0E0E0;margin-bottom:1.2rem}
  .step-badge {background:#2C3E50;border-left:6px solid #FF9800;padding:.6rem 1.2rem;
               border-radius:4px;font-weight:700;color:#fff;
               margin-top:1.5rem;margin-bottom:1.2rem;font-size:1.1rem}
  .stTabs [data-baseweb="tab-list"]{gap:8px}
  .stTabs [data-baseweb="tab"]{background-color:#2D3748;border-radius:6px 6px 0 0;
                                padding:8px 20px;color:#A0AEC0}
  .stTabs [data-baseweb="tab"][aria-selected="true"]{background-color:#3182CE !important;color:white !important}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
for k in ["bands_dict","band_names","profile","tif_bytes","indices_computed",
          "X_scaled","scaler","valid_mask","selected_features","pixel_area_m2","image_shape",
          "kmeans_result","kmeans_classified","mine_rf_result","mine_mask"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ─────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────
def safe_divide(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b != 0, a / b, np.nan).astype(np.float32)

def compute_indices(bands_dict):
    B2  = bands_dict.get("B2_Blue",      np.zeros((1,1), np.float32))
    B3  = bands_dict.get("B3_Green",     np.zeros((1,1), np.float32))
    B4  = bands_dict.get("B4_Red",       np.zeros((1,1), np.float32))
    B8  = bands_dict.get("B8_NIR",       np.zeros((1,1), np.float32))
    B11 = bands_dict.get("B11_SWIR_10m", np.zeros((1,1), np.float32))
    has_nir  = "B8_NIR"       in bands_dict
    has_swir = "B11_SWIR_10m" in bands_dict
    idx = {}
    if has_nir and "B4_Red" in bands_dict:
        idx["NDVI"] = safe_divide(B8-B4, B8+B4)
        idx["EVI"]  = np.clip(2.5*safe_divide(B8-B4, B8+6*B4-7.5*B2+1), -1, 1).astype(np.float32)
        idx["SAVI"] = safe_divide(1.5*(B8-B4), B8+B4+0.5)
    if has_nir and "B3_Green" in bands_dict:
        idx["NDWI"] = safe_divide(B3-B8, B3+B8)
    if has_swir and has_nir:
        idx["BSI"]  = safe_divide((B11+B4)-(B8+B2), (B11+B4)+(B8+B2))
        idx["MBI"]  = safe_divide(B11+B4-B8, B11+B4+B8)
        idx["NDBI"] = safe_divide(B11-B8, B11+B8)
        idx["CMI"]  = safe_divide(B11, B8)
        idx["FCI"]  = safe_divide(B11, B8)
    return idx

def make_rgb(bands_dict, H, W):
    r = bands_dict.get("B4_Red")
    g = bands_dict.get("B3_Green")
    b = bands_dict.get("B2_Blue")
    if r is None or g is None or b is None:
        return None
    def norm(a):
        p2,p98 = np.nanpercentile(a,2), np.nanpercentile(a,98)
        return np.clip((a-p2)/(p98-p2+1e-9), 0, 1)
    return np.dstack([norm(r), norm(g), norm(b)])

def build_feature_matrix(bands_dict, features):
    stack = np.stack([bands_dict[f] for f in features], axis=0)
    _, H, W = stack.shape
    flat  = stack.reshape(len(features), -1).T
    valid = np.all(np.isfinite(flat), axis=1)
    return flat, valid, H, W

def colors_for(names, cdict):
    return [np.array(cdict.get(n,[128,128,128]))/255.0 for n in names]

def build_class_rgb(img_int, color_map_int, H, W):
    rgb = np.zeros((H,W,3), dtype=np.uint8)
    for lbl,col in color_map_int.items():
        rgb[img_int==lbl] = col
    return rgb

def safe_sample(mask_2d, X_full, n, rng):
    idx = np.where(mask_2d.ravel())[0]
    idx = idx[~np.any(np.isnan(X_full[idx]), axis=1)]
    chosen = rng.choice(len(idx), min(n,len(idx)), replace=False)
    return X_full[idx[chosen]], idx[chosen]

def load_shapefile_from_upload(shp_upload, shp_zip_upload):
    tmp = tempfile.mkdtemp()
    if shp_zip_upload is not None:
        z = zipfile.ZipFile(io.BytesIO(shp_zip_upload.read()))
        z.extractall(tmp)
        shp_files = [os.path.join(r,f) for r,_,fs in os.walk(tmp) for f in fs if f.endswith(".shp")]
        if not shp_files:
            return None, "No .shp inside ZIP."
        return gpd.read_file(shp_files[0]), None
    elif shp_upload:
        path = os.path.join(tmp, shp_upload.name)
        with open(path,"wb") as f: f.write(shp_upload.read())
        try:    return gpd.read_file(path), None
        except Exception as e: return None, str(e)
    return None, "No file provided."

# ─────────────────────────────────────────────
# Preprocessing: reproject + resample + scale
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def preprocess_raw_sentinel2(raw_bytes, band_names_tuple, scale_factor):
    """
    1. Reproject all bands to EPSG:32647 (UTM 47N)
    2. Bilinear-resample B11 SWIR from 20 m → 10 m
    3. Divide by scale_factor → reflectance [0–1]
    Returns bands_dict, profile, pixel_area_m2, H, W
    """
    band_names = list(band_names_tuple)
    with rasterio.open(io.BytesIO(raw_bytes)) as src:
        src_crs  = src.crs
        src_t    = src.transform
        n_bands  = src.count
        nd       = src.nodata

        # Target grid at 10 m
        dst_t, dst_w, dst_h = calculate_default_transform(
            src_crs, TARGET_CRS, src.width, src.height,
            *src.bounds, resolution=TARGET_RES
        )

        bands_dict = {}
        for bi, bname in enumerate(band_names):
            if bname == "Other" or bi >= n_bands:
                continue
            band_raw = src.read(bi+1).astype(np.float32)
            if nd is not None:
                band_raw[band_raw == nd] = np.nan

            # Choose resampling: bilinear for B11 SWIR (20m→10m), nearest for rest
            resamp = Resampling.bilinear if bname in SWIR_NAMES else Resampling.nearest
            dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
            reproject(
                source=band_raw, destination=dst,
                src_transform=src_t, src_crs=src_crs,
                dst_transform=dst_t, dst_crs=TARGET_CRS,
                resampling=resamp,
                src_nodata=np.nan, dst_nodata=np.nan,
            )
            key = bname if bname not in bands_dict else f"{bname}_{bi}"
            bands_dict[key] = (dst / float(scale_factor)).astype(np.float32)

    px_area_m2 = TARGET_RES * TARGET_RES  # 100 m²
    profile = {
        "driver":"GTiff","dtype":"float32",
        "crs":TARGET_CRS,"transform":dst_t,
        "width":dst_w,"height":dst_h,
        "count":len(bands_dict),"nodata":np.nan,
    }
    return bands_dict, profile, px_area_m2, dst_h, dst_w

@st.cache_data(show_spinner=False)
def _cached_indices(bands_dict):
    return compute_indices(bands_dict)

@st.cache_data(show_spinner=False)
def _cached_rgb(bands_dict, H, W):
    return make_rgb(bands_dict, H, W)

@st.cache_data(show_spinner=False)
def _cached_feature_matrix(bands_dict, features_tuple):
    return build_feature_matrix(bands_dict, list(features_tuple))

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.markdown("""
<style>
.sidebar-nav-btn{display:block;width:100%;text-align:left;padding:.45rem .9rem;margin:3px 0;
  border:none;border-radius:6px;cursor:pointer;font-size:.93rem;font-weight:600;
  background:#2D3748;color:#CBD5E0;transition:background .15s,color .15s}
.sidebar-nav-btn:hover{background:#3182CE;color:white}
.sidebar-nav-btn.done{border-left:4px solid #38A169}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    sidebar_mode = st.radio("🔀 Mode",
        ["🛰️ Training Pipeline","🤖 Inference (Load Models)"],
        index=0, key="sidebar_mode")
    st.divider()

    if st.session_state.sidebar_mode == "🛰️ Training Pipeline":
        st.markdown("### 📋 Pipeline Progress")
        st.caption("Click a step to jump ↓")
        STEPS = [
            ("step_upload",   "1. 📁 Upload GeoTIFF",     st.session_state.bands_dict is not None),
            ("step_indices",  "2. 🔧 Compute Indices",    st.session_state.indices_computed is True),
            ("step_features", "3. ✅ Feature Selection",  st.session_state.X_scaled is not None),
            ("step_kmeans",   "4–5. 🔵 Classification",   st.session_state.kmeans_result is not None),
            ("step_ts",       "7. 📅 Multi-Year Analysis", st.session_state.get("yearly_results") not in (None,{})),
        ]
        for anchor, label, done in STEPS:
            st.markdown(
                f'<a href="#{anchor}" style="text-decoration:none;">'
                f'<div class="sidebar-nav-btn {"done" if done else ""}">'
                f'{"✅ " if done else "⬜ "}{label}</div></a>',
                unsafe_allow_html=True)

        st.divider()
        st.markdown("### 💾 Download Models")

        def _pkl(obj):
            buf = io.BytesIO(); pickle.dump(obj,buf); return buf.getvalue()

        if st.session_state.kmeans_result:
            st.download_button("⬇️ K-Means (.pkl)", _pkl(st.session_state.kmeans_result["model"]),
                               "kmeans_model.pkl", use_container_width=True)
        if st.session_state.mine_rf_result:
            st.download_button("⬇️ 4-Class Mine RF (.pkl)", _pkl(st.session_state.mine_rf_result["model"]),
                               "rf_4class_mine_model.pkl", use_container_width=True)
            if st.session_state.scaler:
                st.download_button("⬇️ Feature Scaler (.pkl)", _pkl(st.session_state.scaler),
                                   "feature_scaler.pkl", use_container_width=True)
            # ZIP bundle
            try:
                zbuf = io.BytesIO()
                with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED) as zf:
                    if st.session_state.kmeans_result:
                        zf.writestr("kmeans_model.pkl", _pkl(st.session_state.kmeans_result["model"]))
                    zf.writestr("rf_4class_mine_model.pkl", _pkl(st.session_state.mine_rf_result["model"]))
                    if st.session_state.scaler:
                        zf.writestr("feature_scaler.pkl", _pkl(st.session_state.scaler))
                    if st.session_state.selected_features:
                        zf.writestr("selected_features.txt",
                                    ",".join(st.session_state.selected_features))
                st.download_button("⬇️ All Models (ZIP)", zbuf.getvalue(),
                                   "models_bundle.zip", use_container_width=True)
            except Exception: pass

    # ── INFERENCE MODE SIDEBAR ──────────────────────────────────────────────
    else:
        st.markdown("### 🤖 Inference")
        st.caption("Upload model files and raw GeoTIFFs to classify without retraining.")

        st.markdown("#### Model Files")
        infer_scaler     = st.file_uploader("Feature Scaler (.pkl)", type=["pkl"], key="infer_scaler")
        infer_model_type = st.selectbox("Model type",
            ["4-Class Mine RF","3-Class RF","K-Means"], key="infer_model_type")
        infer_model_file = st.file_uploader("Model (.pkl)", type=["pkl"], key="infer_model_file")

        st.markdown("#### Band Names (same order as training)")
        infer_n_bands = st.number_input("Number of bands", 1, 10, 5, key="infer_n_bands")
        infer_band_names = []
        cols_inf = st.columns(min(int(infer_n_bands),5))
        for bi in range(int(infer_n_bands)):
            with cols_inf[bi % min(int(infer_n_bands),5)]:
                bname = st.selectbox(f"Band {bi+1}", KNOWN_BANDS,
                    index=bi if bi < len(KNOWN_BANDS)-1 else len(KNOWN_BANDS)-1,
                    key=f"infer_band_{bi}")
                infer_band_names.append(bname)

        infer_scale = st.number_input("Scale factor (DN→reflectance)", 1, 100000, 10000, key="infer_scale")
        infer_features_txt = st.text_area("Features (comma-separated)",
            value="B8_NIR,B11_SWIR_10m,NDVI,EVI,SAVI,BSI,FCI", key="infer_features_txt")

        st.markdown("#### Upload Image(s)")
        infer_tif  = st.file_uploader("📷 Image A (required)", type=["tif","tiff"], key="infer_tif")
        infer_tif2 = st.file_uploader("📷 Image B (optional — temporal)", type=["tif","tiff"], key="infer_tif2")
        if infer_tif2:
            c1,c2 = st.columns(2)
            infer_yr_a = c1.number_input("Year A", value=2020, step=1, key="infer_yr_a")
            infer_yr_b = c2.number_input("Year B", value=2024, step=1, key="infer_yr_b")

        for k in ["infer_result","infer_result2"]:
            if k not in st.session_state: st.session_state[k] = None

        run_infer = st.button("▶ Run Inference", type="primary",
            disabled=(infer_tif is None or infer_scaler is None or infer_model_file is None),
            key="run_inference_btn")

        def _infer_one(tif_upload, model, scaler_obj, band_names, scale, sel_feats):
            raw = tif_upload.read()
            bd, _, px_area, h, w = preprocess_raw_sentinel2(
                raw, tuple(band_names), int(scale))
            idx = _cached_indices(bd)
            for k,v in idx.items():
                if k not in bd: bd[k] = v
            miss = [f for f in sel_feats if f not in bd]
            if miss: raise ValueError(f"Missing features: {miss}. Available: {list(bd.keys())}")
            stack = np.stack([bd[f] for f in sel_feats], axis=0)
            Xf    = stack.reshape(len(sel_feats),-1).T
            valid = ~np.any(np.isnan(Xf), axis=1)
            Xsc   = np.full_like(Xf, np.nan)
            Xsc[valid] = scaler_obj.transform(Xf[valid])
            pred  = np.zeros(h*w, dtype=np.int16)
            BATCH = 500_000
            for s in range(0, valid.sum(), BATCH):
                idx_v = np.where(valid)[0][s:s+BATCH]
                pred[idx_v] = model.predict(Xsc[idx_v]).astype(np.int16)
            return {"image":pred.reshape(h,w),"model_type":infer_model_type,
                    "bands_dict":bd,"h":h,"w":w,"px_area":px_area}

        if run_infer:
            try:
                sel_f = [f.strip() for f in infer_features_txt.split(",") if f.strip()]
                m_    = pickle.loads(infer_model_file.read())
                sc_   = pickle.loads(infer_scaler.read())
                with st.spinner("Classifying Image A…"):
                    st.session_state.infer_result = _infer_one(
                        infer_tif, m_, sc_, infer_band_names, infer_scale, sel_f)
                if infer_tif2:
                    with st.spinner("Classifying Image B…"):
                        r2 = _infer_one(infer_tif2, m_, sc_, infer_band_names, infer_scale, sel_f)
                        r2["year_b"] = int(st.session_state.get("infer_yr_b", 2024))
                        st.session_state.infer_result["year_a"] = int(st.session_state.get("infer_yr_a", 2020))
                        st.session_state.infer_result2 = r2
                else:
                    st.session_state.infer_result2 = None
                st.success("✅ Done! Scroll down for results.")
            except Exception as e:
                import traceback
                st.error(f"Inference failed: {e}")
                st.code(traceback.format_exc())

    st.divider()

# ─────────────────────────────────────────────
# Main area — Training Pipeline
# ─────────────────────────────────────────────
_training = st.session_state.get("sidebar_mode","🛰️ Training Pipeline") == "🛰️ Training Pipeline"

if _training:
    st.markdown('<div class="main-header">🛰️ Satellite Image Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Raw Sentinel-2 → Reproject → Indices → K-Means → RF → Mine RF</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# STEP 1 — UPLOAD & PREPROCESS
# ══════════════════════════════════════════════
if _training:
    st.markdown('<div id="step_upload" class="step-badge">Step 1 — Upload Raw GeoTIFF</div>', unsafe_allow_html=True)
    st.markdown("""
    Upload a **raw 5-band Sentinel-2 GeoTIFF** exported from GEE
    (bands: B2, B3, B4, B8, B11 — DN values, not reflectance).
    The app will reproject to **EPSG:32647 (UTM 47N) at 10 m**, bilinear-resample B11 from 20 m,
    and divide by 10 000 to convert to reflectance [0–1].
    """)

uploaded = st.file_uploader("Upload raw GeoTIFF (.tif/.tiff)",
    type=["tif","tiff"], key="main_uploader") if _training else None

if uploaded and _training:
    raw_bytes = uploaded.read()
    st.session_state.tif_bytes = raw_bytes

    # Read tags to pre-fill band names
    with rasterio.open(io.BytesIO(raw_bytes)) as src:
        n_bands = src.count
        tags    = [src.tags(i).get("name", f"Band_{i}") for i in range(1, n_bands+1)]
        crs_str = str(src.crs) if src.crs else "unknown"
        H_raw, W_raw = src.height, src.width

    st.success(f"Loaded **{n_bands} bands** | **{H_raw} × {W_raw} px** | CRS: `{crs_str}`")
    st.info("Will reproject to **EPSG:32647 (UTM 47N) at 10 m**. B11 SWIR bilinear-resampled.")

    st.markdown("#### Assign Band Names")
    st.caption("Match the band order in your GEE export (typically B2, B3, B4, B8, B11).")
    cols_bn = st.columns(min(n_bands, 5))
    assigned = []
    for i in range(n_bands):
        default = tags[i] if tags[i] in KNOWN_BANDS else "Other"
        with cols_bn[i % min(n_bands, 5)]:
            name = st.selectbox(f"Band {i+1}\n(tag: {tags[i]})", KNOWN_BANDS,
                index=KNOWN_BANDS.index(default), key=f"band_name_{i}")
            assigned.append(name)

    scale_factor = st.number_input(
        "Scale factor (GEE Sentinel-2 SR = 10 000)",
        min_value=1, value=10000, key="step1_scale_factor",
        help="Raw DN ÷ scale_factor = reflectance. GEE S2_SR_HARMONIZED exports use 10 000.")

    if st.button("✅ Reproject & compute reflectance", type="primary"):
        with st.spinner("Reprojecting to UTM 47N at 10 m…"):
            bd, prof, px_area, dst_h, dst_w = preprocess_raw_sentinel2(
                raw_bytes, tuple(assigned), scale_factor)
        st.session_state.bands_dict      = bd
        st.session_state.band_names      = assigned
        st.session_state.profile         = prof
        st.session_state.pixel_area_m2   = px_area
        st.session_state.image_shape     = (dst_h, dst_w)
        st.session_state.indices_computed = None
        st.session_state.X_scaled        = None
        st.session_state.kmeans_result   = None
        st.session_state.mine_rf_result  = None
        st.rerun()

if st.session_state.bands_dict and _training:
    H, W = st.session_state.image_shape
    rgb  = _cached_rgb(st.session_state.bands_dict, H, W)
    if rgb is not None:
        st.markdown("### 🗺️ RGB Preview (after reprojection)")
        fig, ax = plt.subplots(figsize=(8,3.5))
        ax.imshow(rgb); ax.axis("off")
        st.pyplot(fig, use_container_width=True); plt.close()

# ══════════════════════════════════════════════
# STEP 2 — COMPUTE INDICES
# ══════════════════════════════════════════════
if st.session_state.bands_dict and _training:
    st.divider()
    st.markdown('<div id="step_indices" class="step-badge">Step 2 — Compute Spectral Indices</div>', unsafe_allow_html=True)
    st.markdown("""
    Computes 9 spectral indices from the reprojected reflectance bands.
    Each index highlights a specific surface type — NDVI for vegetation,
    BSI/MBI for bare soil and mine surfaces, FCI for iron-rich tailings.
    Only indices computable from your uploaded bands are shown.
    """)

    bd = st.session_state.bands_dict
    computable  = _cached_indices(bd)
    new_indices = [k for k in computable if k not in bd]

    if not new_indices:
        st.info("All indices already computed.")
        st.session_state.indices_computed = True
    else:
        sel_idx = st.multiselect("Select indices to add:", new_indices, default=new_indices,
            help="Notebook recommendation: NDVI, EVI, SAVI, NDWI, BSI, MBI, NDBI, CMI, FCI")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⚙️ Compute selected indices", type="primary"):
                for name in sel_idx:
                    bd[name] = computable[name]
                st.session_state.bands_dict       = bd
                st.session_state.indices_computed = True
                st.rerun()
        with col_b:
            if st.session_state.indices_computed:
                st.success(f"✅ {len([k for k in bd if k not in KNOWN_BANDS])} indices computed")

# ══════════════════════════════════════════════
# STEP 3 — FEATURE SELECTION
# ══════════════════════════════════════════════
if st.session_state.indices_computed and _training:
    st.divider()
    st.markdown('<div id="step_features" class="step-badge">Step 3 — Feature Selection & Scaling</div>', unsafe_allow_html=True)
    st.markdown("""
    Choose which bands and indices to feed into the classifier.
    The notebook recommendation drops B2/B3/B4 (redundant with indices) and NDBI/CMI
    (correlated with BSI). Features are standardised with `StandardScaler` (zero mean, unit variance).
    """)
    bd = st.session_state.bands_dict
    all_feats   = list(bd.keys())
    nb_defaults = ['B8_NIR','B11_SWIR_10m','NDVI','EVI','SAVI','BSI','FCI']
    default_sel = [f for f in nb_defaults if f in all_feats] or all_feats
    st.caption("Recommended: B8_NIR, B11_SWIR_10m, NDVI, EVI, SAVI, BSI, FCI")
    selected = st.multiselect("Features:", all_feats, default=default_sel)

    if selected and st.button("🔄 Build & scale feature matrix", type="primary"):
        with st.spinner("Building feature matrix…"):
            flat, valid, H, W = _cached_feature_matrix(bd, tuple(selected))
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(flat[valid])
        st.session_state.X_scaled          = X_scaled
        st.session_state.scaler            = scaler
        st.session_state.valid_mask        = valid
        st.session_state.selected_features = selected
        st.session_state.image_shape       = (H, W)
        st.session_state.kmeans_result     = None
        st.session_state.kmeans_classified = None
        st.session_state.mine_rf_result    = None
        st.rerun()

# ══════════════════════════════════════════════
# STEPS 4–6 — CLASSIFICATION TABS
# ══════════════════════════════════════════════
if st.session_state.X_scaled is not None and _training:
    st.divider()
    st.markdown('<div id="step_kmeans" class="step-badge">Steps 4–5 — Classification</div>', unsafe_allow_html=True)
    st.markdown("""
    Run the three classifiers in order. **K-Means** finds clusters; you assign class labels.
    **3-Class RF** uses those labels for supervised training. **4-Class Mine RF** adds a Mine
    class using a shapefile polygon.
    """)

    tab_km, tab_mine = st.tabs([
        "🔵 K-Means", "⛏️ 4-Class Mine RF"])

    X_scaled     = st.session_state.X_scaled
    scaler       = st.session_state.scaler
    valid_mask   = st.session_state.valid_mask
    sel_features = st.session_state.selected_features
    px_area      = st.session_state.pixel_area_m2 or 100.0
    H, W         = int(st.session_state.image_shape[0]), int(st.session_state.image_shape[1])

    # ── K-MEANS ──────────────────────────────────────────────────────────────
    with tab_km:
        st.subheader("K-Means Unsupervised Clustering")
        st.caption("Fits on a 50 000-pixel subsample for speed; predicts on the full image.")
        K = st.slider("Number of clusters (k)", 2, 6, 4, key="km_k_slider")

        if st.button(f"▶ Run K-Means (k={K})", type="primary"):
            with st.spinner("Fitting K-Means…"):
                MAX_KM = 50_000
                rng_km = np.random.default_rng(42)
                idx_km = rng_km.choice(len(X_scaled), min(MAX_KM, len(X_scaled)), replace=False)
                X_km   = X_scaled[idx_km]
                km = KMeans(n_clusters=K, init="k-means++", n_init=3,
                            max_iter=300, random_state=42)
                km.fit(X_km)
                # Batched predict on full image
                labels = np.empty(len(X_scaled), dtype=np.int32)
                BATCH  = 500_000
                for s in range(0, len(X_scaled), BATCH):
                    labels[s:s+BATCH] = km.predict(X_scaled[s:s+BATCH])
                idx_s = np.random.choice(len(X_scaled), min(20000, len(X_scaled)), replace=False)
                sil   = compute_silhouette(X_scaled[idx_s], labels[idx_s])
                cents = scaler.inverse_transform(km.cluster_centers_)
                df_c  = pd.DataFrame(cents, columns=sel_features,
                                     index=[f"Cluster {i}" for i in range(K)])
                limg  = np.full(H*W, -1, dtype=np.int16)
                limg[valid_mask] = labels
                limg  = limg.reshape(H, W)
                st.session_state.kmeans_result    = {
                    "model":km,"labels":labels,"label_img":limg,
                    "centroids":df_c,"sil":sil,"class_names":[None]*K,"trained_k":K}
                st.session_state.kmeans_classified = False
            st.rerun()

        if st.session_state.kmeans_result is not None:
            km_res    = st.session_state.kmeans_result
            trained_k = km_res["trained_k"]
            st.metric("Silhouette Score", f"{km_res['sil']:.4f}")
            st.markdown("#### Cluster Centroids")
            st.dataframe(km_res["centroids"].style.background_gradient(cmap="viridis", axis=1),
                         use_container_width=True)
            st.caption("Highest NDVI → Forest | Moderate → Sparse Vegetation | Lowest + high BSI → Bare Soil")

            st.markdown("#### Assign Class Labels")
            class_names = []
            cols_km = st.columns(trained_k)
            for i in range(trained_k):
                with cols_km[i]:
                    ndvi_s = (f"NDVI≈{km_res['centroids']['NDVI'].iloc[i]:.3f}"
                              if "NDVI" in km_res["centroids"].columns else "")
                    bsi_s  = (f" BSI≈{km_res['centroids']['BSI'].iloc[i]:.3f}"
                              if "BSI" in km_res["centroids"].columns else "")
                    n = st.selectbox(f"Cluster {i} ({ndvi_s}{bsi_s})",
                                     ["— select —"]+CLASSES, index=0, key=f"km_cls_{i}")
                    class_names.append(n)

            all_ok = all(n != "— select —" for n in class_names)
            if not all_ok:
                st.warning("Assign a class to every cluster.")

            if st.button("🗺️ Classify & Show Results", disabled=not all_ok):
                km_res["class_names"] = class_names
                if "NDVI" in st.session_state.bands_dict:
                    ndvi_band = st.session_state.bands_dict["NDVI"]
                    for i, n in enumerate(class_names):
                        if n != "Forest":
                            km_res["label_img"][(ndvi_band >= 0.75) & (km_res["label_img"] == i)] = \
                                class_names.index("Forest") if "Forest" in class_names else i
                st.session_state.kmeans_result    = km_res
                st.session_state.kmeans_classified = True
                st.rerun()

            if st.session_state.get("kmeans_classified"):
                col_norm = colors_for(class_names, CLASS_COLORS_3)
                cmap_km  = ListedColormap(col_norm)
                map_c, stat_c = st.columns([2,1])
                with map_c:
                    # PCA scatter
                    idx_p = np.random.choice(len(X_scaled), min(10000,len(X_scaled)), replace=False)
                    pca   = PCA(n_components=2, random_state=42)
                    X_2d  = pca.fit_transform(X_scaled[idx_p])
                    c2d   = pca.transform(km_res["model"].cluster_centers_)
                    fig_s, ax_s = plt.subplots(figsize=(8,5))
                    for ki in range(trained_k):
                        m = km_res["labels"][idx_p] == ki
                        ax_s.scatter(X_2d[m,0],X_2d[m,1],s=8,alpha=0.4,color=col_norm[ki],linewidths=0,label=class_names[ki])
                    ax_s.scatter(c2d[:,0],c2d[:,1],s=200,marker="X",color="white",edgecolors="black",linewidths=1.2,zorder=5)
                    ax_s.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
                    ax_s.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
                    ax_s.legend(markerscale=2,fontsize=9); ax_s.grid(True,linestyle="--",alpha=0.3)
                    st.pyplot(fig_s); plt.close()

                    rgb_img = _cached_rgb(st.session_state.bands_dict, H, W)
                    fig_m, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))
                    if rgb_img is not None: ax1.imshow(rgb_img)
                    ax1.set_title("True Colour",fontweight="bold"); ax1.axis("off")
                    ax2.imshow(km_res["label_img"],cmap=cmap_km,vmin=0,vmax=trained_k-1)
                    ax2.set_title(f"K-Means (k={trained_k})",fontweight="bold"); ax2.axis("off")
                    patches = [mpatches.Patch(color=col_norm[i],label=class_names[i]) for i in range(trained_k)]
                    ax2.legend(handles=patches,loc="lower right",fontsize=9)
                    plt.tight_layout(); st.pyplot(fig_m,use_container_width=True); plt.close()

                with stat_c:
                    from collections import Counter
                    valid_labels = km_res["label_img"][km_res["label_img"] >= 0].ravel()
                    cnt = Counter(valid_labels.tolist())
                    rows = []
                    for i,n in enumerate(class_names):
                        ha = cnt.get(i,0)*px_area/10000
                        pct= cnt.get(i,0)/len(valid_labels)*100 if len(valid_labels) else 0
                        rows.append({"Land Cover":n,"Hectares (ha)":round(ha,2),"Coverage":f"{pct:.1f}%"})
                    df_a = pd.DataFrame(rows)
                    st.dataframe(df_a, use_container_width=True)
                    fig_b,ax_b = plt.subplots(figsize=(5,3.5))
                    ax_b.barh(df_a["Land Cover"],df_a["Hectares (ha)"],color=col_norm)
                    ax_b.set_xlabel("Hectares"); ax_b.invert_yaxis()
                    ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)
                    st.pyplot(fig_b,use_container_width=True); plt.close()

    # ── 4-CLASS MINE RF ───────────────────────────────────────────────────────
    with tab_mine:
        st.markdown('<div id="step_mine_rf"></div>', unsafe_allow_html=True)
        st.subheader("⛏️ 4-Class Mine RF")
        st.markdown("Adds a **Mine** class from a polygon shapefile. Requires K-Means first.")

        if not HAS_GEO:
            st.error("geopandas/shapely not installed.")
        elif not st.session_state.get("kmeans_classified"):
            st.warning("Complete K-Means first.")
        else:
            c1,c2 = st.columns(2)
            with c1: shp_zip    = st.file_uploader("Shapefile ZIP (recommended)", type=["zip"], key="shp_zip")
            with c2: shp_single = st.file_uploader("Or .shp only", type=["shp"], key="shp_single")

            st.markdown("#### Mine Mask Parameters")
            cp1,cp2 = st.columns(2)
            ndvi_thresh = cp1.number_input("Max NDVI for mine pixels",0.0,1.0,0.55,0.05,key="mine_ndvi_thresh")
            ndwi_thresh = cp2.number_input("Min NDWI for mine pixels",-1.0,0.0,-0.22,0.05,key="mine_ndwi_thresh")
            cp3,cp4 = st.columns(2)
            n_per_mine = cp3.number_input("Samples per class",500,50000,5000,500,key="mine_n_per")
            n_est_mine = cp4.slider("Trees",50,300,200,key="mine_n_est")

            if st.button("▶ Rasterize & Build Mine Mask", type="primary",
                         disabled=(shp_zip is None and shp_single is None)):
                with st.spinner("Rasterizing polygons…"):
                    gdf,err = load_shapefile_from_upload(shp_single, shp_zip)
                    if err:
                        st.error(err)
                    else:
                        profile   = st.session_state.profile
                        transform = profile["transform"]   # reprojected UTM 47N transform
                        gdf       = gdf.to_crs(profile["crs"])
                        mine_mask = rasterize(
                            [(mapping(geom),1) for geom in gdf.geometry],
                            out_shape=(H,W), transform=transform, fill=0, dtype=np.uint8)
                        n_mine = mine_mask.sum()
                        st.session_state.mine_mask = mine_mask
                        st.success(f"✅ {len(gdf)} polygon(s) → {n_mine:,} mine pixels "
                                   f"(~{n_mine*px_area/10000:,.1f} ha)")

            if st.session_state.mine_mask is not None:
                if st.button("▶ Train 4-Class Mine RF", type="primary"):
                    with st.spinner("Training…"):
                        mine_mask  = st.session_state.mine_mask
                        km_res     = st.session_state.kmeans_result
                        class_names_km = km_res["class_names"]
                        bd         = st.session_state.bands_dict

                        stack_full = np.stack([bd[f] for f in sel_features], axis=0)
                        X_full     = stack_full.reshape(len(sel_features),-1).T

                        kmeans_img = km_res["label_img"]
                        NDVI_b = bd.get("NDVI"); NDWI_b = bd.get("NDWI")

                        f_idx = class_names_km.index("Forest") if "Forest" in class_names_km else -1
                        s_idx = class_names_km.index("Sparse Vegetation") if "Sparse Vegetation" in class_names_km else -1
                        b_idx = class_names_km.index("Bare Soil") if "Bare Soil" in class_names_km else -1

                        f_mask = ((kmeans_img==f_idx) & (NDVI_b>=0.70))
                        s_mask = kmeans_img == s_idx
                        b_mask = kmeans_img == b_idx

                        mine_train = mine_mask.astype(bool)
                        if NDVI_b is not None: mine_train &= (NDVI_b < ndvi_thresh)
                        if NDWI_b is not None: mine_train &= (NDWI_b > ndwi_thresh)

                        rng_m = np.random.default_rng(42)
                        X_f,_ = safe_sample(f_mask, X_full, n_per_mine, rng_m)
                        X_s,_ = safe_sample(s_mask, X_full, n_per_mine, rng_m)
                        X_b,_ = safe_sample(b_mask, X_full, n_per_mine, rng_m)
                        X_m,_ = safe_sample(mine_train, X_full, n_per_mine, rng_m)

                        st.write(f"Samples — Forest:`{len(X_f)}` Sparse:`{len(X_s)}` "
                                 f"Bare:`{len(X_b)}` Mine:`{len(X_m)}`")

                        X_tr_raw = np.concatenate([X_f,X_s,X_b,X_m])
                        y_train  = np.array(
                            [CLASS_TO_INT_4["Forest"]]*len(X_f)+
                            [CLASS_TO_INT_4["Sparse Vegetation"]]*len(X_s)+
                            [CLASS_TO_INT_4["Bare Soil"]]*len(X_b)+
                            [CLASS_TO_INT_4["Mine"]]*len(X_m), dtype=np.int32)

                        X_sc_tr = scaler.transform(X_tr_raw)
                        X_tr4,X_te4,y_tr4,y_te4 = train_test_split(X_sc_tr,y_train,test_size=0.2,random_state=42,stratify=y_train)
                        rf_mine = RandomForestClassifier(n_estimators=n_est_mine,class_weight="balanced",n_jobs=-1,random_state=42,oob_score=True)
                        rf_mine.fit(X_tr4,y_tr4)

                        # Batched full-image prediction
                        rf_full = np.zeros(H*W, dtype=np.uint8)
                        for s in range(0, len(X_scaled), 500_000):
                            rf_full[np.where(valid_mask)[0][s:s+500_000]] = \
                                rf_mine.predict(X_scaled[s:s+500_000]).astype(np.uint8)
                        rf_img = rf_full.reshape(H,W)

                        st.session_state.mine_rf_result = {
                            "model":rf_mine,"rf_image":rf_img,"oob":rf_mine.oob_score_,
                            "y_true_te":y_te4,"y_pred_te":rf_mine.predict(X_te4),
                            "X_train_sc":X_sc_tr,"y_train":y_train}
                        st.success(f"✅ OOB Accuracy: {rf_mine.oob_score_:.4f}")

            if st.session_state.mine_rf_result is not None:
                res      = st.session_state.mine_rf_result
                rf_image = res["rf_image"]
                color_map_int = {0:[0,0,0],**{CLASS_TO_INT_4[k]:v for k,v in CLASS_COLORS_4.items()}}
                rgb_cls  = build_class_rgb(rf_image, color_map_int, H, W)
                legend_p = [mpatches.Patch(color=np.array(CLASS_COLORS_4[c])/255,label=c) for c in CLASSES_4]

                fig_main,(ax1,ax2) = plt.subplots(1,2,figsize=(16,7))
                fig_main.suptitle("4-Class Mine RF Classification",fontsize=13,fontweight="bold")
                ax1.imshow(rgb_cls); ax1.axis("off")
                ax1.legend(handles=legend_p+[mpatches.Patch(color="black",label="No Data")],
                           loc="lower left",fontsize=9,framealpha=0.85,title="Land Cover")
                class_areas = [(rf_image==CLASS_TO_INT_4[c]).sum()*px_area/10000 for c in CLASSES_4]
                bars=ax2.barh(CLASSES_4,class_areas,color=[np.array(CLASS_COLORS_4[c])/255 for c in CLASSES_4],edgecolor="white")
                ax2.set_xlabel("Hectares"); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
                for bar,area in zip(bars,class_areas):
                    ax2.text(bar.get_width()+max(class_areas)*0.01,bar.get_y()+bar.get_height()/2,f"{area:,.0f} ha",va="center",fontsize=10)
                plt.tight_layout(); st.pyplot(fig_main,use_container_width=True); plt.close()

                rgb_true = _cached_rgb(st.session_state.bands_dict, H, W)
                if rgb_true is not None:
                    fig_tc,(ax_tc1,ax_tc2)=plt.subplots(1,2,figsize=(20,10),gridspec_kw={"wspace":0.02})
                    ax_tc1.imshow(rgb_true,aspect="auto"); ax_tc1.set_title("True Colour",fontsize=11,fontweight="bold"); ax_tc1.axis("off")
                    ax_tc2.imshow(rgb_cls,aspect="auto");  ax_tc2.set_title("4-Class RF",fontsize=11,fontweight="bold");  ax_tc2.axis("off")
                    ax_tc2.legend(handles=legend_p,loc="lower left",fontsize=10,framealpha=0.9)
                    plt.tight_layout(); st.pyplot(fig_tc,use_container_width=True); plt.close()

                df_a4=pd.DataFrame({"Land Cover":CLASSES_4,"Hectares (ha)":[round(a,2) for a in class_areas],
                    "Coverage":[f"{a/sum(class_areas)*100:.1f}%" for a in class_areas]})
                st.dataframe(df_a4, use_container_width=True)

                # Feature importance
                imp=res["model"].feature_importances_; sidx=np.argsort(imp)[::-1]
                fig_fi,ax_fi=plt.subplots(figsize=(10,4))
                ax_fi.bar([sel_features[i] for i in sidx],imp[sidx],color="#378ADD",edgecolor="white")
                ax_fi.set_ylabel("Importance (Gini)"); ax_fi.set_title("Feature Importance",fontweight="bold")
                ax_fi.spines["top"].set_visible(False); ax_fi.spines["right"].set_visible(False)
                plt.tight_layout(); st.pyplot(fig_fi,use_container_width=True); plt.close()

                # Confusion matrix
                ul4=sorted(np.unique(res["y_true_te"])); int2cls={v:k for k,v in CLASS_TO_INT_4.items()}
                unames4=[int2cls.get(i,str(i)) for i in ul4]
                c1,c2=st.columns([1,1.2])
                with c1:
                    cm4=confusion_matrix(res["y_true_te"],res["y_pred_te"],labels=ul4)
                    fig_cm4,ax_cm4=plt.subplots(figsize=(6,5))
                    im4=ax_cm4.imshow(cm4,cmap="Blues"); fig_cm4.colorbar(im4,ax=ax_cm4,fraction=0.046,pad=0.04)
                    ax_cm4.set_xticks(range(len(ul4))); ax_cm4.set_xticklabels(unames4,rotation=30,ha="right",fontsize=9)
                    ax_cm4.set_yticks(range(len(ul4))); ax_cm4.set_yticklabels(unames4,fontsize=9)
                    for i in range(cm4.shape[0]):
                        for j in range(cm4.shape[1]):
                            ax_cm4.text(j,i,format(cm4[i,j],'d'),ha="center",va="center",
                                       color="white" if cm4[i,j]>cm4.max()/2 else "black",fontweight="bold")
                    plt.tight_layout(); st.pyplot(fig_cm4,use_container_width=True); plt.close()
                with c2:
                    rpt4=classification_report(res["y_true_te"],res["y_pred_te"],labels=ul4,target_names=unames4,output_dict=True)
                    st.dataframe(pd.DataFrame(rpt4).T.style.format(precision=3)
                                 .background_gradient(cmap="Blues",subset=["precision","recall","f1-score"]),
                                 use_container_width=True)
                    st.caption(f"OOB: **{res['oob']:.4f}** | Test: **{(res['y_pred_te']==res['y_true_te']).mean():.4f}**")

# ══════════════════════════════════════════════
# STEP 7 — MULTI-YEAR ANALYSIS
# ══════════════════════════════════════════════
if st.session_state.mine_rf_result is not None and _training:
    st.divider()
    st.markdown('<div id="step_ts" class="step-badge">Step 7 — Multi-Year Change Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    Upload one raw GeoTIFF per year (same band order as Step 1).
    Each image is reprojected, scaled to reflectance, and indices are computed
    before being classified with the trained 4-class RF model.
    """)

    YEARS = [2020,2021,2022,2023,2024,2025]
    if "yearly_results" not in st.session_state or st.session_state.yearly_results is None:
        st.session_state.yearly_results = {}

    year_files = {}
    for row in [YEARS[:3], YEARS[3:]]:
        cols_yr = st.columns(3)
        for col,yr in zip(cols_yr,row):
            with col:
                f = st.file_uploader(f"📅 {yr}", type=["tif","tiff"], key=f"yr_{yr}")
                if f is not None: year_files[yr] = f

    base_year = st.selectbox("Baseline year",
        options=sorted(year_files.keys()) if year_files else YEARS, index=0)

    if len(year_files) < 2:
        st.info("Upload at least 2 years.")
    else:
        if st.button("▶ Classify All Years & Analyse Change", type="primary"):
            rf_model    = st.session_state.mine_rf_result["model"]
            scaler_     = st.session_state.scaler
            band_names_ = st.session_state.band_names
            scale_f_    = 10000  # same as Step 1

            yearly = {}
            prog = st.progress(0, text="Classifying…")
            for i,(yr,fobj) in enumerate(sorted(year_files.items())):
                prog.progress(i/len(year_files), text=f"Classifying {yr}…")
                try:
                    raw = fobj.read()
                    # Full pipeline: reproject → scale → indices → classify
                    bd_yr,_,pxa_yr,h_yr,w_yr = preprocess_raw_sentinel2(
                        raw, tuple(band_names_), scale_f_)
                    idx_yr = _cached_indices(bd_yr)
                    for k,v in idx_yr.items():
                        if k not in bd_yr: bd_yr[k] = v

                    miss = [f for f in sel_features if f not in bd_yr]
                    if miss: st.warning(f"{yr}: missing {miss} — skip."); continue

                    stack_yr = np.stack([bd_yr[f] for f in sel_features],axis=0)
                    Xf_yr    = stack_yr.reshape(len(sel_features),-1).T
                    valid_yr = ~np.any(np.isnan(Xf_yr),axis=1)
                    Xsc_yr   = np.full_like(Xf_yr, np.nan)
                    Xsc_yr[valid_yr] = scaler_.transform(Xf_yr[valid_yr])

                    labels_yr = np.zeros(h_yr*w_yr, dtype=np.uint8)
                    for s in range(0,valid_yr.sum(),500_000):
                        idx_v = np.where(valid_yr)[0][s:s+500_000]
                        labels_yr[idx_v] = rf_model.predict(Xsc_yr[idx_v]).astype(np.uint8)
                    img_yr = labels_yr.reshape(h_yr,w_yr)

                    areas_yr = {n:int((img_yr==v).sum())*pxa_yr/10000
                                for n,v in CLASS_TO_INT_4.items()}
                    yearly[yr] = {"image":img_yr,"areas":areas_yr,"bands_dict":bd_yr,"h":h_yr,"w":w_yr}
                    prog.progress((i+1)/len(year_files),
                        text=f"{yr} — Forest:{areas_yr['Forest']:,.0f} ha | Mine:{areas_yr['Mine']:,.0f} ha")
                except Exception as e:
                    st.error(f"{yr}: {e}")

            st.session_state.yearly_results = yearly
            prog.empty()
            st.success(f"Classified {len(yearly)} year(s).")

    # ── Display results ───────────────────────────────────────────────────────
    yearly = st.session_state.yearly_results
    if yearly and len(yearly) >= 1:
        years_sorted = sorted(yearly.keys())
        base_yr  = base_year if base_year in yearly else years_sorted[0]
        last_yr  = years_sorted[-1]
        span     = last_yr - years_sorted[0] if len(years_sorted) > 1 else 1

        forest_ha = [yearly[y]["areas"]["Forest"]            for y in years_sorted]
        sparse_ha = [yearly[y]["areas"]["Sparse Vegetation"] for y in years_sorted]
        bare_ha   = [yearly[y]["areas"]["Bare Soil"]         for y in years_sorted]
        mine_ha   = [yearly[y]["areas"]["Mine"]              for y in years_sorted]

        bf = yearly[base_yr]["areas"]["Forest"]
        bsv= yearly[base_yr]["areas"]["Sparse Vegetation"]
        bbs= yearly[base_yr]["areas"]["Bare Soil"]
        bm = yearly[base_yr]["areas"]["Mine"]

        total_valid_ha = {y: sum(yearly[y]["areas"].values()) for y in years_sorted}

        def pct_of_total(ha, yr): return ha / total_valid_ha[yr] * 100 if total_valid_ha[yr] > 0 else 0
        def pct_chg(v, base): return (v - base) / base * 100 if base > 0 else 0

        # ── TABS ──────────────────────────────────────────────────────────────
        tab_ov, tab_maps, tab_trend, tab_rate, tab_detail = st.tabs([
            "📋 Overview", "🗺️ Classification Maps", "📈 Trends",
            "📊 Change Rates", "🔍 Detailed Stats"
        ])

        # ════════════════════════════════════════
        # TAB 1 — OVERVIEW
        # ════════════════════════════════════════
        with tab_ov:
            st.markdown("#### 🔑 Key Findings")

            # Headline metrics — 8 boxes
            total_fl = bf - yearly[last_yr]["areas"]["Forest"]
            total_mg = yearly[last_yr]["areas"]["Mine"] - bm
            peak_mine_yr = max(years_sorted, key=lambda y: yearly[y]["areas"]["Mine"])
            peak_defor_yr = None
            if len(years_sorted) >= 2:
                defor_rates = [(yearly[y0]["areas"]["Forest"]-yearly[y1]["areas"]["Forest"])/(y1-y0)
                               for y0,y1 in zip(years_sorted[:-1],years_sorted[1:])]
                peak_defor_yr = years_sorted[defor_rates.index(max(defor_rates))]

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("🌲 Forest Loss",
                      f"{total_fl:,.0f} ha",
                      f"{pct_chg(yearly[last_yr]['areas']['Forest'], bf):+.1f}% vs {base_yr}",
                      delta_color="inverse")
            m2.metric("⛏️ Mine Growth",
                      f"{total_mg:,.0f} ha",
                      f"{pct_chg(yearly[last_yr]['areas']['Mine'], bm):+.1f}% vs {base_yr}" if bm>0 else "new area")
            m3.metric("📅 Avg Deforestation",
                      f"{total_fl/span:,.0f} ha/yr",
                      f"over {span} yr(s)")
            m4.metric("📅 Avg Mine Growth",
                      f"{total_mg/span:,.0f} ha/yr",
                      f"over {span} yr(s)")

            m5,m6,m7,m8 = st.columns(4)
            m5.metric("🌿 Forest (baseline)",
                      f"{bf:,.0f} ha",
                      f"{pct_of_total(bf, base_yr):.1f}% of study area")
            m6.metric("⛏️ Mine (baseline)",
                      f"{bm:,.0f} ha",
                      f"{pct_of_total(bm, base_yr):.1f}% of study area")
            m7.metric("🌲 Forest (latest)",
                      f"{yearly[last_yr]['areas']['Forest']:,.0f} ha",
                      f"{pct_of_total(yearly[last_yr]['areas']['Forest'], last_yr):.1f}% of study area")
            m8.metric("⛏️ Mine (latest)",
                      f"{yearly[last_yr]['areas']['Mine']:,.0f} ha",
                      f"{pct_of_total(yearly[last_yr]['areas']['Mine'], last_yr):.1f}% of study area")

            st.divider()

            # ── Full summary table with % coverage ────────────────────────────
            st.markdown("#### 📋 Annual Land Cover Summary")
            rows_sum = []
            for y in years_sorted:
                a = yearly[y]["areas"]
                tv = total_valid_ha[y]
                rows_sum.append({
                    "Year": y,
                    "Forest (ha)":      f"{a['Forest']:,.0f}",
                    "Forest (%)":       f"{pct_of_total(a['Forest'],y):.1f}%",
                    "Sparse Veg (ha)":  f"{a['Sparse Vegetation']:,.0f}",
                    "Sparse Veg (%)":   f"{pct_of_total(a['Sparse Vegetation'],y):.1f}%",
                    "Bare Soil (ha)":   f"{a['Bare Soil']:,.0f}",
                    "Bare Soil (%)":    f"{pct_of_total(a['Bare Soil'],y):.1f}%",
                    "Mine (ha)":        f"{a['Mine']:,.0f}",
                    "Mine (%)":         f"{pct_of_total(a['Mine'],y):.1f}%",
                    f"Forest Δ vs {base_yr}": f"{pct_chg(a['Forest'],bf):+.1f}%",
                    f"Mine Δ vs {base_yr}":   f"{pct_chg(a['Mine'],bm):+.1f}%" if bm>0 else "n/a",
                })
            df_sum = pd.DataFrame(rows_sum)
            st.dataframe(df_sum, use_container_width=True, hide_index=True)

            # Download button
            csv_sum = pd.DataFrame([{
                "Year": y,
                "Forest_ha":      yearly[y]["areas"]["Forest"],
                "Forest_pct":     pct_of_total(yearly[y]["areas"]["Forest"],y),
                "SparseVeg_ha":   yearly[y]["areas"]["Sparse Vegetation"],
                "SparseVeg_pct":  pct_of_total(yearly[y]["areas"]["Sparse Vegetation"],y),
                "BareSoil_ha":    yearly[y]["areas"]["Bare Soil"],
                "BareSoil_pct":   pct_of_total(yearly[y]["areas"]["Bare Soil"],y),
                "Mine_ha":        yearly[y]["areas"]["Mine"],
                "Mine_pct":       pct_of_total(yearly[y]["areas"]["Mine"],y),
                "Forest_delta_pct": pct_chg(yearly[y]["areas"]["Forest"],bf),
                "Mine_delta_pct":   pct_chg(yearly[y]["areas"]["Mine"],bm) if bm>0 else None,
            } for y in years_sorted])
            st.download_button("⬇️ Download Summary CSV",
                csv_sum.to_csv(index=False).encode(),
                "land_cover_summary.csv", mime="text/csv")

        # ════════════════════════════════════════
        # TAB 2 — CLASSIFICATION MAPS
        # ════════════════════════════════════════
        with tab_maps:
            st.markdown("#### 🗺️ Year-by-Year: True Colour vs Classification")
            st.caption("Each row shows the true-colour composite (left) and 4-class RF classification (right) for that year.")

            color_map_int_yr = {0:[0,0,0], **{CLASS_TO_INT_4[k]:v for k,v in CLASS_COLORS_4.items()}}
            legend_p_yr = [mpatches.Patch(color=np.array(CLASS_COLORS_4[c])/255, label=c) for c in CLASSES_4] +                           [mpatches.Patch(color="black", label="No Data")]

            def _ds(arr, mx=600):
                h,w = arr.shape[:2]
                f = max(1, max(h,w)//mx)
                return arr[::f,::f]

            for yr in years_sorted:
                res_yr   = yearly[yr]
                h_yr, w_yr = res_yr["h"], res_yr["w"]
                rgb_yr   = _cached_rgb(res_yr["bands_dict"], h_yr, w_yr)
                cls_rgb  = np.zeros((h_yr, w_yr, 3), dtype=np.uint8)
                for lbl,col in color_map_int_yr.items():
                    cls_rgb[res_yr["image"]==lbl] = col

                a = res_yr["areas"]
                tv = total_valid_ha[yr]

                st.markdown(f"**{yr}** — Forest: {a['Forest']:,.0f} ha ({pct_of_total(a['Forest'],yr):.1f}%) | "
                            f"Sparse Veg: {a['Sparse Vegetation']:,.0f} ha ({pct_of_total(a['Sparse Vegetation'],yr):.1f}%) | "
                            f"Bare Soil: {a['Bare Soil']:,.0f} ha ({pct_of_total(a['Bare Soil'],yr):.1f}%) | "
                            f"Mine: {a['Mine']:,.0f} ha ({pct_of_total(a['Mine'],yr):.1f}%)")

                fig_yr, (ax_tc, ax_cl) = plt.subplots(1, 2, figsize=(16, 5),
                                                        gridspec_kw={"wspace": 0.02})
                fig_yr.suptitle(f"Land Cover Classification — {yr}", fontsize=12, fontweight="bold")
                if rgb_yr is not None:
                    ax_tc.imshow(_ds(rgb_yr)); ax_tc.set_title("True Colour (B4/B3/B2)", fontweight="bold")
                else:
                    ax_tc.set_title("True Colour (unavailable)"); ax_tc.text(0.5,0.5,"No RGB",ha="center",transform=ax_tc.transAxes)
                ax_tc.axis("off")
                ax_cl.imshow(_ds(cls_rgb)); ax_cl.set_title("4-Class RF Classification", fontweight="bold"); ax_cl.axis("off")
                ax_cl.legend(handles=legend_p_yr, loc="lower left", fontsize=8,
                             framealpha=0.85, title="Land Cover", title_fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_yr, use_container_width=True); plt.close()
                st.divider()

            # First vs last year side-by-side comparison
            if len(years_sorted) >= 2:
                st.markdown(f"#### 🔍 First vs Last Year: {years_sorted[0]} → {last_yr}")
                fig_cmp, axes_cmp = plt.subplots(2, 2, figsize=(18, 12),
                                                  gridspec_kw={"wspace":0.02,"hspace":0.12})
                fig_cmp.suptitle(f"Land Cover Change: {years_sorted[0]} → {last_yr}",
                                  fontsize=13, fontweight="bold")
                for col_i, yr_c in enumerate([years_sorted[0], last_yr]):
                    res_c = yearly[yr_c]
                    h_c, w_c = res_c["h"], res_c["w"]
                    rgb_c = _cached_rgb(res_c["bands_dict"], h_c, w_c)
                    cls_c = np.zeros((h_c, w_c, 3), dtype=np.uint8)
                    for lbl,col in color_map_int_yr.items():
                        cls_c[res_c["image"]==lbl] = col
                    if rgb_c is not None:
                        axes_cmp[0,col_i].imshow(_ds(rgb_c))
                    axes_cmp[0,col_i].set_title(f"{yr_c} — True Colour", fontweight="bold"); axes_cmp[0,col_i].axis("off")
                    axes_cmp[1,col_i].imshow(_ds(cls_c)); axes_cmp[1,col_i].axis("off")
                    axes_cmp[1,col_i].set_title(f"{yr_c} — Classification", fontweight="bold")
                axes_cmp[1,1].legend(handles=legend_p_yr, loc="lower left", fontsize=9,
                                     framealpha=0.85, title="Land Cover")
                plt.tight_layout()
                st.pyplot(fig_cmp, use_container_width=True); plt.close()

        # ════════════════════════════════════════
        # TAB 3 — TRENDS
        # ════════════════════════════════════════
        with tab_trend:
            st.markdown("#### 📈 Land Cover Trends Over Time")

            # Stacked area
            fig_sa, ax_sa = plt.subplots(figsize=(12, 5))
            ax_sa.stackplot(years_sorted, forest_ha, sparse_ha, bare_ha, mine_ha,
                            labels=CLASSES_4,
                            colors=[np.array(CLASS_COLORS_4[c])/255 for c in CLASSES_4], alpha=0.85)
            ax_sa.set_xlabel("Year"); ax_sa.set_ylabel("Area (ha)")
            ax_sa.set_title("Land Cover Area Over Time", fontweight="bold")
            ax_sa.legend(loc="upper right", fontsize=9)
            ax_sa.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:,.0f}"))
            ax_sa.spines["top"].set_visible(False); ax_sa.spines["right"].set_visible(False)
            plt.tight_layout(); st.pyplot(fig_sa, use_container_width=True); plt.close()

            # % coverage stacked bar
            st.markdown("##### % Land Cover Composition per Year")
            fig_pct, ax_pct = plt.subplots(figsize=(12, 5))
            bottoms = np.zeros(len(years_sorted))
            for cls_n in CLASSES_4:
                pcts = [pct_of_total(yearly[y]["areas"][cls_n], y) for y in years_sorted]
                ax_pct.bar(years_sorted, pcts, bottom=bottoms,
                           color=np.array(CLASS_COLORS_4[cls_n])/255,
                           label=cls_n, edgecolor="white", width=0.5)
                # Label bars ≥ 3%
                for xi, (ys, pv, bv) in enumerate(zip(years_sorted, pcts, bottoms)):
                    if pv >= 3:
                        ax_pct.text(ys, bv + pv/2, f"{pv:.1f}%",
                                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")
                bottoms += np.array(pcts)
            ax_pct.set_ylabel("% of Study Area"); ax_pct.set_ylim(0,100)
            ax_pct.set_title("Land Cover Composition (%) by Year", fontweight="bold")
            ax_pct.legend(loc="lower right", fontsize=9)
            ax_pct.spines["top"].set_visible(False); ax_pct.spines["right"].set_visible(False)
            plt.tight_layout(); st.pyplot(fig_pct, use_container_width=True); plt.close()

            # Forest vs Mine dual-axis
            st.markdown("##### Forest Cover vs Mine Expansion")
            fig_fm, ax_f = plt.subplots(figsize=(12, 5))
            ax_m = ax_f.twinx()
            l1, = ax_f.plot(years_sorted, forest_ha, "o-",
                            color=np.array(CLASS_COLORS_4["Forest"])/255,
                            linewidth=2.5, markersize=8, label="Forest")
            l2, = ax_m.plot(years_sorted, mine_ha, "s--",
                            color=np.array(CLASS_COLORS_4["Mine"])/255,
                            linewidth=2.5, markersize=8, label="Mine")
            # Annotate each point
            for y, fv, mv in zip(years_sorted, forest_ha, mine_ha):
                ax_f.annotate(f"{fv:,.0f}", (y, fv), textcoords="offset points",
                              xytext=(0,8), ha="center", fontsize=8,
                              color=np.array(CLASS_COLORS_4["Forest"])/255)
                ax_m.annotate(f"{mv:,.0f}", (y, mv), textcoords="offset points",
                              xytext=(0,-14), ha="center", fontsize=8,
                              color=np.array(CLASS_COLORS_4["Mine"])/255)
            ax_f.set_ylabel("Forest Area (ha)"); ax_m.set_ylabel("Mine Area (ha)")
            ax_f.set_title("Forest Cover vs Mine Expansion", fontweight="bold")
            ax_f.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:,.0f}"))
            ax_m.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:,.0f}"))
            ax_f.legend(handles=[l1,l2], loc="center left", fontsize=9)
            ax_f.grid(True, linestyle="--", alpha=0.3)
            ax_f.spines["top"].set_visible(False)
            plt.tight_layout(); st.pyplot(fig_fm, use_container_width=True); plt.close()

            # Cumulative % change vs baseline
            st.markdown(f"##### Cumulative % Change vs {base_yr} Baseline")
            fig_cum, ax_cum = plt.subplots(figsize=(12, 5))
            for cls_n, col_key in [("Forest","Forest"),("Sparse Vegetation","Sparse Vegetation"),
                                    ("Bare Soil","Bare Soil"),("Mine","Mine")]:
                base_v = yearly[base_yr]["areas"][cls_n]
                if base_v > 0:
                    cpcts = [pct_chg(yearly[y]["areas"][cls_n], base_v) for y in years_sorted]
                    ax_cum.plot(years_sorted, cpcts, "o-",
                                color=np.array(CLASS_COLORS_4[col_key])/255,
                                linewidth=2, markersize=7, label=cls_n)
                    for y, cp in zip(years_sorted, cpcts):
                        if y != base_yr:
                            ax_cum.annotate(f"{cp:+.1f}%", (y, cp),
                                            textcoords="offset points", xytext=(0,7),
                                            ha="center", fontsize=8)
            ax_cum.axhline(0, color="grey", linewidth=1, linestyle="--")
            ax_cum.set_ylabel(f"Change vs {base_yr} (%)"); ax_cum.set_xlabel("Year")
            ax_cum.set_title(f"Cumulative % Change from {base_yr} Baseline", fontweight="bold")
            ax_cum.legend(fontsize=9)
            ax_cum.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:+.1f}%"))
            ax_cum.grid(True, linestyle="--", alpha=0.3)
            ax_cum.spines["top"].set_visible(False); ax_cum.spines["right"].set_visible(False)
            plt.tight_layout(); st.pyplot(fig_cum, use_container_width=True); plt.close()

        # ════════════════════════════════════════
        # TAB 4 — CHANGE RATES
        # ════════════════════════════════════════
        with tab_rate:
            st.markdown("#### 📊 Period-by-Period Change Analysis")

            if len(years_sorted) >= 2:
                yr_labels, defor_rate, mine_rate = [], [], []
                sparse_rate, bare_rate = [], []
                rows_rate = []
                for y0, y1 in zip(years_sorted[:-1], years_sorted[1:]):
                    gap = y1 - y0
                    label = f"{y0}–{y1}"
                    yr_labels.append(label)
                    f0, f1 = yearly[y0]["areas"]["Forest"],            yearly[y1]["areas"]["Forest"]
                    s0, s1 = yearly[y0]["areas"]["Sparse Vegetation"], yearly[y1]["areas"]["Sparse Vegetation"]
                    b0, b1 = yearly[y0]["areas"]["Bare Soil"],         yearly[y1]["areas"]["Bare Soil"]
                    m0, m1 = yearly[y0]["areas"]["Mine"],              yearly[y1]["areas"]["Mine"]
                    dr = (f0-f1)/gap; mr = (m1-m0)/gap
                    sr = (s1-s0)/gap; br = (b1-b0)/gap
                    defor_rate.append(dr); mine_rate.append(mr)
                    sparse_rate.append(sr); bare_rate.append(br)
                    rows_rate.append({
                        "Period": label,
                        "Forest Loss (ha/yr)":      f"{dr:+,.1f}",
                        "Forest Loss (%)":           f"{pct_chg(f1,f0):+.1f}%",
                        "Mine Growth (ha/yr)":       f"{mr:+,.1f}",
                        "Mine Growth (%)":           f"{pct_chg(m1,m0):+.1f}%" if m0>0 else "new",
                        "Sparse Veg Δ (ha/yr)":     f"{sr:+,.1f}",
                        "Bare Soil Δ (ha/yr)":       f"{br:+,.1f}",
                    })

                st.dataframe(pd.DataFrame(rows_rate), use_container_width=True, hide_index=True)

                # Grouped bar — deforestation vs mine growth rates
                x = np.arange(len(yr_labels)); w = 0.35
                fig_cr, ax_cr = plt.subplots(figsize=(12, 5))
                b1_bars = ax_cr.bar(x - w/2, defor_rate, w,
                                    label="Forest Loss (ha/yr)", color="#c0392b", edgecolor="white")
                b2_bars = ax_cr.bar(x + w/2, mine_rate,  w,
                                    label="Mine Growth (ha/yr)", color="#0050C8", edgecolor="white")
                # Value labels
                for bar in b1_bars:
                    v = bar.get_height()
                    ax_cr.text(bar.get_x()+bar.get_width()/2,
                               v + max(abs(r) for r in defor_rate+mine_rate)*0.02,
                               f"{v:+,.0f}", ha="center", fontsize=8, color="#c0392b", fontweight="bold")
                for bar in b2_bars:
                    v = bar.get_height()
                    ax_cr.text(bar.get_x()+bar.get_width()/2,
                               v + max(abs(r) for r in defor_rate+mine_rate)*0.02,
                               f"{v:+,.0f}", ha="center", fontsize=8, color="#0050C8", fontweight="bold")
                ax_cr.set_xticks(x); ax_cr.set_xticklabels(yr_labels, fontsize=9)
                ax_cr.axhline(0, color="black", linewidth=0.8, linestyle="--")
                ax_cr.set_ylabel("ha/yr"); ax_cr.set_title("Annual Change Rate by Period", fontweight="bold")
                ax_cr.legend(fontsize=9)
                ax_cr.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:+,.0f}"))
                ax_cr.spines["top"].set_visible(False); ax_cr.spines["right"].set_visible(False)
                plt.tight_layout(); st.pyplot(fig_cr, use_container_width=True); plt.close()

                # All 4 classes rate
                fig_all, ax_all = plt.subplots(figsize=(12, 5))
                x4 = np.arange(len(yr_labels)); w4 = 0.2
                for ki, (rates, cls_n) in enumerate([
                    ([-r for r in defor_rate], "Forest"),
                    (sparse_rate,              "Sparse Vegetation"),
                    (bare_rate,                "Bare Soil"),
                    (mine_rate,                "Mine"),
                ]):
                    ax_all.bar(x4 + (ki-1.5)*w4, rates, w4,
                               label=cls_n,
                               color=np.array(CLASS_COLORS_4[cls_n])/255,
                               edgecolor="white")
                ax_all.set_xticks(x4); ax_all.set_xticklabels(yr_labels, fontsize=9)
                ax_all.axhline(0, color="black", linewidth=0.8, linestyle="--")
                ax_all.set_ylabel("Net Change (ha/yr)")
                ax_all.set_title("Net Annual Change — All Classes", fontweight="bold")
                ax_all.legend(fontsize=9)
                ax_all.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:+,.0f}"))
                ax_all.spines["top"].set_visible(False); ax_all.spines["right"].set_visible(False)
                plt.tight_layout(); st.pyplot(fig_all, use_container_width=True); plt.close()

            else:
                st.info("Need at least 2 years to compute change rates.")

        # ════════════════════════════════════════
        # TAB 5 — DETAILED STATS
        # ════════════════════════════════════════
        with tab_detail:
            st.markdown("#### 🔍 Per-Year Detailed Breakdown")

            sel_yr = st.selectbox("Select year to inspect:", years_sorted,
                                   index=len(years_sorted)-1, key="detail_yr_sel")
            res_d   = yearly[sel_yr]
            a_d     = res_d["areas"]
            tv_d    = total_valid_ha[sel_yr]
            a_base  = yearly[base_yr]["areas"]

            # 4 class metrics
            dc1,dc2,dc3,dc4 = st.columns(4)
            for col_w, cls_n, icon in zip([dc1,dc2,dc3,dc4], CLASSES_4,
                                           ["🌲","🌿","🟤","⛏️"]):
                chg = pct_chg(a_d[cls_n], a_base[cls_n]) if a_base[cls_n]>0 else None
                col_w.metric(f"{icon} {cls_n}",
                             f"{a_d[cls_n]:,.0f} ha",
                             f"{pct_of_total(a_d[cls_n],sel_yr):.1f}% of study area")

            # Horizontal bar chart with % labels
            fig_d, ax_d = plt.subplots(figsize=(10, 4))
            bars_d = ax_d.barh(
                CLASSES_4,
                [a_d[c] for c in CLASSES_4],
                color=[np.array(CLASS_COLORS_4[c])/255 for c in CLASSES_4],
                edgecolor="white", height=0.5)
            max_ha = max(a_d[c] for c in CLASSES_4)
            for bar_d, cls_n in zip(bars_d, CLASSES_4):
                v = bar_d.get_width()
                pct_v = pct_of_total(v, sel_yr)
                chg_v = pct_chg(v, a_base[cls_n]) if a_base[cls_n]>0 else None
                chg_str = f" ({chg_v:+.1f}% vs {base_yr})" if chg_v is not None else ""
                ax_d.text(v + max_ha*0.01,
                          bar_d.get_y() + bar_d.get_height()/2,
                          f"{v:,.0f} ha  |  {pct_v:.1f}%{chg_str}",
                          va="center", fontsize=9, fontweight="bold")
            ax_d.set_xlabel("Area (hectares)")
            ax_d.set_title(f"Land Cover — {sel_yr}", fontweight="bold")
            ax_d.set_xlim(0, max_ha * 1.55)
            ax_d.spines["top"].set_visible(False); ax_d.spines["right"].set_visible(False)
            ax_d.spines["left"].set_visible(False)
            ax_d.tick_params(axis="y", length=0)
            ax_d.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:,.0f}"))
            plt.tight_layout(); st.pyplot(fig_d, use_container_width=True); plt.close()

            # Change vs baseline table
            st.markdown(f"##### Change vs {base_yr} Baseline")
            rows_chg = []
            for cls_n in CLASSES_4:
                bv = a_base[cls_n]; cv = a_d[cls_n]
                rows_chg.append({
                    "Class":          cls_n,
                    f"{base_yr} (ha)": f"{bv:,.0f}",
                    f"{sel_yr} (ha)":  f"{cv:,.0f}",
                    "Absolute Δ (ha)": f"{cv-bv:+,.0f}",
                    "Relative Δ (%)":  f"{pct_chg(cv,bv):+.1f}%" if bv>0 else "n/a",
                    f"{base_yr} (%)":  f"{pct_of_total(bv,base_yr):.1f}%",
                    f"{sel_yr} (%)":   f"{pct_of_total(cv,sel_yr):.1f}%",
                    "Coverage Δ (pp)": f"{pct_of_total(cv,sel_yr)-pct_of_total(bv,base_yr):+.1f} pp",
                })
            st.dataframe(pd.DataFrame(rows_chg), use_container_width=True, hide_index=True)

            # Pie chart comparison
            if base_yr != sel_yr:
                fig_pie, (ax_p1, ax_p2) = plt.subplots(1, 2, figsize=(12, 5))
                pie_cols = [np.array(CLASS_COLORS_4[c])/255 for c in CLASSES_4]
                for ax_p, yr_p in [(ax_p1, base_yr), (ax_p2, sel_yr)]:
                    vals_p = [yearly[yr_p]["areas"][c] for c in CLASSES_4]
                    wedges, texts, autotexts = ax_p.pie(
                        vals_p, labels=CLASSES_4, colors=pie_cols,
                        autopct="%1.1f%%", startangle=90,
                        wedgeprops={"edgecolor":"white","linewidth":1.2})
                    for at in autotexts: at.set_fontsize(9)
                    ax_p.set_title(str(yr_p), fontweight="bold", fontsize=12)
                fig_pie.suptitle("Land Cover Composition Comparison", fontsize=12, fontweight="bold")
                plt.tight_layout(); st.pyplot(fig_pie, use_container_width=True); plt.close()

# ══════════════════════════════════════════════
# INFERENCE RESULTS
# ══════════════════════════════════════════════
def _cls_map_colors(model_type, pred_img):
    if "4-Class" in model_type or "Mine" in model_type:
        return CLASS_TO_INT_4, CLASS_COLORS_4, CLASSES_4
    elif "3-Class" in model_type:
        return CLASS_TO_INT_3, CLASS_COLORS_3, CLASSES
    else:
        unique = sorted(set(pred_img[pred_img>=0].ravel()))
        palette=[[34,85,34],[255,191,0],[139,69,19],[0,80,200],[180,60,180],[200,100,0]]
        cls_map={f"Cluster {l}":l for l in unique}
        clr_map={f"Cluster {l}":palette[i%len(palette)] for i,l in enumerate(unique)}
        return cls_map, clr_map, list(cls_map.keys())

def _render_infer_result(res, title=""):
    h,w       = res["h"],res["w"]
    pred_img  = res["image"]
    cls_map,clr_map,classes = _cls_map_colors(res["model_type"], pred_img)
    rgb_cls = np.zeros((h,w,3),dtype=np.uint8)
    for cn,lv in cls_map.items():
        rgb_cls[pred_img==lv] = clr_map.get(cn,[128,128,128])

    col_m,col_s = st.columns([2,1])
    with col_m:
        st.markdown(f"### 🗺️ Classification Map {title}")
        rgb_t = _cached_rgb(res["bands_dict"],h,w)
        if rgb_t is not None:
            fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,7))
            ax1.imshow(rgb_t); ax1.set_title("True Colour",fontweight="bold"); ax1.axis("off")
            ax2.imshow(rgb_cls); ax2.set_title("Classification",fontweight="bold"); ax2.axis("off")
        else:
            fig,ax2=plt.subplots(figsize=(8,7))
            ax2.imshow(rgb_cls); ax2.set_title("Classification",fontweight="bold"); ax2.axis("off")
        patches=[mpatches.Patch(color=np.array(clr_map[c])/255,label=c) for c in classes]
        ax2.legend(handles=patches,loc="lower right",fontsize=9)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

    with col_s:
        st.markdown("### 📊 Area Statistics")
        rows_i=[]
        for cn,lv in cls_map.items():
            cnt=int((pred_img==lv).sum())
            rows_i.append({"Land Cover":cn,"Hectares (ha)":round(cnt*res["px_area"]/10000,2),
                           "Coverage":f"{cnt/(h*w)*100:.1f}%"})
        df_i=pd.DataFrame(rows_i)
        st.dataframe(df_i,use_container_width=True)
        cols_bar=[np.array(clr_map[c])/255 for c in classes]
        fig_b,ax_b=plt.subplots(figsize=(5,3.5))
        ax_b.barh(df_i["Land Cover"],df_i["Hectares (ha)"],color=cols_bar)
        ax_b.set_xlabel("Hectares"); ax_b.invert_yaxis()
        ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)
        st.pyplot(fig_b,use_container_width=True); plt.close()

    st.download_button(f"⬇️ Download CSV {title}",
        data=pd.DataFrame({"class_label":pred_img.ravel()}).to_csv(index=False).encode(),
        file_name=f"classification{title.replace(' ','_')}.csv", mime="text/csv")
    return df_i

if (st.session_state.get("sidebar_mode")=="🤖 Inference (Load Models)"
        and st.session_state.get("infer_result") is not None):

    res_i  = st.session_state.infer_result
    res_i2 = st.session_state.get("infer_result2")

    st.markdown('<div class="main-header">🤖 Inference Results</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Model: {res_i["model_type"]}</div>', unsafe_allow_html=True)

    if res_i2 is None:
        _render_infer_result(res_i)
    else:
        yr_a = res_i.get("year_a","A"); yr_b = res_i2.get("year_b","B")
        tab_a,tab_b,tab_ch = st.tabs([f"📷 {yr_a}",f"📷 {yr_b}",f"📊 Change {yr_a}→{yr_b}"])
        with tab_a: df_a = _render_infer_result(res_i, f"({yr_a})")
        with tab_b: df_b = _render_infer_result(res_i2,f"({yr_b})")
        with tab_ch:
            st.markdown(f"### Land Cover Change: {yr_a} → {yr_b}")
            mg = df_a.rename(columns={"Hectares (ha)":f"ha_{yr_a}","Coverage":f"cov_{yr_a}"})[["Land Cover",f"ha_{yr_a}"]]\
                    .merge(df_b.rename(columns={"Hectares (ha)":f"ha_{yr_b}","Coverage":f"cov_{yr_b}"})[["Land Cover",f"ha_{yr_b}"]],
                           on="Land Cover",how="outer").fillna(0)
            mg["Change (ha)"] = (mg[f"ha_{yr_b}"]-mg[f"ha_{yr_a}"]).round(2)
            mg["Change (%)"]  = mg.apply(lambda r: f"{r['Change (ha)']/r[f'ha_{yr_a}']*100:+.1f}%"
                                          if r[f"ha_{yr_a}"]>0 else "n/a", axis=1)
            st.dataframe(mg,use_container_width=True,hide_index=True)
            # Net change bar
            fig_n,ax_n=plt.subplots(figsize=(8,4))
            colors_n=["#e74c3c" if v>=0 else "#27ae60" for v in mg["Change (ha)"]]
            ax_n.barh(mg["Land Cover"],mg["Change (ha)"],color=colors_n,edgecolor="white")
            ax_n.axvline(0,color="black",linewidth=1)
            ax_n.set_xlabel("Net Change (ha)"); ax_n.set_title(f"Net Change {yr_a}→{yr_b}",fontweight="bold")
            ax_n.spines["top"].set_visible(False); ax_n.spines["right"].set_visible(False)
            ax_n.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:+,.0f}"))
            plt.tight_layout(); st.pyplot(fig_n,use_container_width=True); plt.close()