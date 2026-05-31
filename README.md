# 🛰️ Satellite Image Classifier

A Streamlit app for classifying raw Sentinel-2 satellite imagery into land cover types using K-Means clustering and Random Forest models.

## 🗺️ Classes
| Class | Color |
|-------|-------|
| Forest | Dark Green |
| Sparse Vegetation | Yellow |
| Bare Soil | Brown/Red |
| Mine | Blue |

## ⚙️ Pipeline
1. Upload a raw **5-band GeoTIFF** exported from Google Earth Engine
2. Reproject to EPSG:32647 (UTM 47N) at 10m resolution
3. Scale DN values to reflectance [0–1]
4. Compute 9 spectral indices (NDVI, EVI, SAVI, NDWI, BSI, MBI, NDBI, CMI, FCI)
5. Feature selection → StandardScaler
6. K-Means (k=4) → 3-Class RF → 4-Class Mine RF
7. Multi-year change analysis

## 📦 Input Format
A **5-band GeoTIFF** from Google Earth Engine with bands in this order:
| Band | Description |
|------|-------------|
| B2 | Blue |
| B3 | Green |
| B4 | Red |
| B8 | NIR |
| B11 | SWIR (20m, auto-resampled to 10m) |

## 🚀 Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run satellite_classifier_app_Finalized.py
```

Then open your browser at `http://localhost:8501`

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set the main file to `satellite_classifier_app_Finalized.py`
5. Click **Deploy**

> **Note:** If the build fails due to GDAL/rasterio issues, make sure `packages.txt` is present in the repo (included here).

## 📁 Project Structure
```
├── satellite_classifier_app_Finalized.py   # Main app
├── requirements.txt                         # Python dependencies
├── packages.txt                             # System-level dependencies (for cloud deploy)
├── .streamlit/
│   └── config.toml                          # App configuration
├── .gitignore
└── README.md
```

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) — UI framework
- [Rasterio](https://rasterio.readthedocs.io/) — GeoTIFF reading & reprojection
- [Scikit-learn](https://scikit-learn.org/) — K-Means & Random Forest
- [GeoPandas](https://geopandas.org/) — Shapefile support (optional)
- [Matplotlib](https://matplotlib.org/) — Visualizations

## Data Sources
| Input | Drive Link |
|-------|-------|
| Satellite Composite Images | https://drive.google.com/drive/folders/1efQnLPzSvYUCsKor9ZxT8UkOSBPe0_7W?usp=sharing |
| Rare-earth mine polygons | https://drive.google.com/drive/folders/1qIl4wOfkxR_6hcwbb5jtwQyXvUGvAljD?usp=sharing |
| Region of Interest polygon | https://drive.google.com/drive/folders/1ipbcjRhMmFmLuigefTjT0UA_6y_Cx5s-?usp=sharing |

