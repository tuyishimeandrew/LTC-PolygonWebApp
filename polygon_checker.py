import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
import io
import re
import numpy as np

###############################################################################
# ⚡ Latitude Inspections Inconsistency Checker — Optimised v2 (Complete)     #
###############################################################################

st.set_page_config(page_title="Latitude Inspections Inconsistency Checker", layout="wide")
st.title("Latitude Inspections Inconsistency Checker ⚡")

# ---------------------------
# 1. FILE UPLOAD + ROBUST LOADER
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Main Inspection File")
    main_file = st.file_uploader(
        "Upload Main Inspection Form (CSV or Excel)", type=["xlsx", "csv"], key="main_upload"
    )
with col2:
    st.subheader("Redo Polygon File")
    redo_file = st.file_uploader(
        "Upload Redo Polygon Form (CSV or Excel)", type=["xlsx", "csv"], key="redo_upload"
    )

if main_file is None or redo_file is None:
    st.info("Please upload both Main and Redo files to continue.")
    st.stop()

@st.cache_data(show_spinner=False)
def _read_tabular(file):
    """Fast and fault-tolerant CSV/XLSX loader with multi-step fallback"""
    def _try(fn):
        file.seek(0)
        return fn()
    try:
        if file.name.lower().endswith('.xlsx'):
            df = _try(lambda: pd.read_excel(file, engine='openpyxl'))
        else:
            try:
                df = _try(lambda: pd.read_csv(file, engine='pyarrow'))
                if df.empty or df.columns.size == 0:
                    raise ValueError
            except Exception:
                try:
                    df = _try(lambda: pd.read_csv(file))
                    if df.empty or df.columns.size == 0:
                        raise ValueError
                except Exception:
                    df = _try(lambda: pd.read_csv(
                        file, engine='python', sep=None, skip_blank_lines=True, on_bad_lines='skip'
                    ))
                    if df.empty or df.columns.size == 0:
                        raise ValueError("File appears empty or has no columns.")
    except Exception as e:
        st.error(f"Error loading {file.name}: {e}")
        st.stop()
    df.columns = df.columns.map(str).str.strip()
    return df

main_df = _read_tabular(main_file)
redo_df = _read_tabular(redo_file)

# Normalize Farmercode in redo_df
redo_df = redo_df.rename(columns={
    'farmer_code': 'Farmercode', 'Farmer_Code': 'Farmercode', 'farmercode': 'Farmercode'
})

# ---------------------------
# 2. VALIDATION
# ---------------------------
required_main = {"Farmercode", "polygonplot"}
required_redo = {"Farmercode", "selectplot", "polygonplot"}
if not required_main.issubset(main_df.columns):
    st.error("Main file must contain Farmercode & polygonplot columns.")
    st.stop()
if not required_redo.issubset(redo_df.columns):
    st.error("Redo file must contain Farmercode, selectplot & polygonplot columns.")
    st.stop()

# ---------------------------
# 3. MERGE REDO POLYGONS
# ---------------------------
if {"SubmissionDate", "endtime"}.issubset(redo_df.columns):
    redo_df['SubmissionDate'] = pd.to_datetime(redo_df['SubmissionDate'], errors='coerce')
    redo_df['endtime'] = pd.to_datetime(redo_df['endtime'], errors='coerce')
    redo_latest = (
        redo_df.sort_values(['Farmercode','SubmissionDate','endtime'])
        .drop_duplicates('Farmercode', keep='last')
        .rename(columns={'selectplot':'redo_selectplot','polygonplot':'redo_polygonplot'})
    )
    main_df = main_df.merge(
        redo_latest[['Farmercode','redo_selectplot','redo_polygonplot']],
        on='Farmercode', how='left'
    )
    plot_map = {'Plot1':'polygonplot','Plot2':'polygonplotnew_1','Plot3':'polygonplotnew_2','Plot4':'polygonplotnew_3','Plot5':'polygonplotnew_4'}
    for sel, col in plot_map.items():
        if col in main_df:
            m = (main_df['redo_selectplot']==sel) & main_df[col].notna()
            main_df.loc[m, col] = main_df.loc[m, 'redo_polygonplot'].astype(str)
    main_df.drop(columns=['redo_selectplot','redo_polygonplot'], errors='ignore', inplace=True)

# ---------------------------
# 4. DATE FILTER
# ---------------------------
date_col = 'Submissiondate'
if 'Submissiondate' in main_df.columns:
    main_df[date_col] = pd.to_datetime(main_df['Submissiondate'], errors='coerce')
elif 'SubmissionDate' in main_df.columns:
    main_df[date_col] = pd.to_datetime(main_df['SubmissionDate'], errors='coerce')
if main_df[date_col].notna().any():
    dmin, dmax = main_df[date_col].min().date(), main_df[date_col].max().date()
    sel_range = st.slider("Select Submission Date Range", dmin, dmax, (dmin, dmax))
    main_df = main_df[(main_df[date_col].dt.date>=sel_range[0]) & (main_df[date_col].dt.date<=sel_range[1])]
else:
    st.warning("Submission date not available; skipping filter.")

# ---------------------------
# 5. GEOMETRY BUILD & STRtree
# ---------------------------
POLY_COLS = [c for c in main_df.columns if c.startswith('polygonplot')]
@st.cache_resource(show_spinner=False)
def build_gdf(df):
    def str2poly(s):
        if not isinstance(s, str): return None
        coords = np.fromstring(re.sub(r'[;,]', ' ', s), sep=' ')
        if coords.size < 6 or coords.size % 3: return None
        pts = coords.reshape(-1,3)[:,:2]
        try: return Polygon(pts)
        except: return None
    geoms = []
    for _, row in df[POLY_COLS].iterrows():
        parts = [str2poly(row[c]) for c in POLY_COLS if pd.notna(row[c])]
        parts = [p.buffer(0) if p and not p.is_valid else p for p in parts if p]
        geoms.append(parts[0] if len(parts)==1 else unary_union(parts) if parts else None)
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geoms, crs='EPSG:4326')
    gdf = gdf[gdf.geometry.notna()].to_crs('EPSG:2109')
    gdf.geometry = gdf.geometry.buffer(0)
    tree = STRtree(gdf.geometry)
    idx_map = {geom:i for i, geom in enumerate(gdf.geometry)}
    return gdf, tree, idx_map
GDF, STR_TREE, IDX_MAP = build_gdf(main_df)

def check_overlaps(code):
    return OVERLAPS.get(code, ([], 0))

# ---------------------------
# 6. OVERLAP PRECOMPUTE
# ---------------------------
@st.cache_resource(show_spinner=False)
def precompute_overlaps(gdf, tree, idx_map):
    cache = {}
    for i, row in gdf.iterrows():
        tgt = row.geometry; tot = tgt.area; union = None; ovls = []
        for geom in tree.query(tgt):
            j = idx_map[geom]
            if j==i: continue
            intr = tgt.intersection(geom)
            if intr.is_empty: continue
            area = intr.area
            ovls.append({
                'Farmercode': gdf.iloc[j]['Farmercode'],
                'overlap_area': area, 'total_area': tot, 'intersection': intr
            })
            union = intr if union is None else union.union(intr)
        pct = (union.area/tot*100) if union else 0
        cache[row['Farmercode']] = (ovls, pct)
    return cache
OVERLAPS = precompute_overlaps(GDF, STR_TREE, IDX_MAP)

# ---------------------------
# 7. PRODUCTIVE PLANTS METRICS
# ---------------------------
@st.cache_resource(show_spinner=False)
def compute_prod_metrics(df):
    area_cols = [c for c in df if re.search('acres_polygonplot', c, re.I)]
    prod_cols = [c for c in df if re.search('productiveplants', c, re.I)]
    A = df[area_cols].fillna(0).astype(float).sum(1)
    P = df[prod_cols].fillna(0).astype(float).sum(1)
    E = A * 450
    return pd.DataFrame({
        'Total_Area': A, 'Total_Productive_Plants': P,
        'Expected_Plants': E, 'Half_Expected_Plants': E/2,
        'Pct125_Expected_Plants': E*1.25,
        'Productive_Plants_Inconsistency': np.where(P<E/2, 'Less than Expected Productive Plants', np.where(P>1.25*E, 'More than expected Productive Plants',''))
    }, index=df.index)
PM = compute_prod_metrics(main_df)

# ---------------------------
# 8. NONCOMPLIANCE FLAGS
# ---------------------------
def flag_yes(s, *vals): return s.astype(str).str.strip().str.lower().isin([v.lower() for v in vals])
LM = (flag_yes(main_df['childrenworkingconditions'],'any_time_when_needed')|
      flag_yes(main_df['prisoners'],'yes')|
      flag_yes(main_df['contractsworkers'],'no')|
      flag_yes(main_df['drinkingwaterworkers'],'no')) & (main_df['noncompliancesfound_Labour'].fillna(0)==0)
EM = (flag_yes(main_df['cutnativetrees'],'yes')|
      flag_yes(main_df['cutforests'],'yes')|
      flag_yes(main_df['toiletdischarge'],'yes')|
      flag_yes(main_df['separatewaste'],'no')) & (main_df['noncompliancesfound_Environmental'].fillna(0)==0)
AG = (main_df[[
    'methodspestdiseasemanagement_using_chemicals','fertilizerchemicals_Pesticides',
    'fertilizerchemicals_Fungicides','fertilizerchemicals_Herbicides',
    'childrenlabouractivities_spraying_of_chemicals','typeworkvulnerable_Spraying_of_chemicals',
    'agriculturalinputs_synthetic_chemicals_or_fertilize']].fillna(0).astype(float).eq(1).any(1)) & (main_df['noncompliancesfound_Agro_chemical'].fillna(0)==0)
AM = (flag_yes(main_df['pruning'],'no')|
      flag_yes(main_df['desuckering'],'no')|
      flag_yes(main_df['manageweeds'],'no')|
      flag_yes(main_df['knowledgeIPM'],'no')) & (main_df['noncompliancesfound_Agronomic'].fillna(0)==0)
PMis = (flag_yes(main_df['ripepods'],'no')|
        flag_yes(main_df['storedrycocoa'],'no')|
        flag_yes(main_df['separatebasins'],'no')) & (main_df['noncompliancesfound_Harvest_and_postharvestt'].fillna(0)==0)
PhM = ~flag_yes(main_df['phone_match'],'match')
TM = main_df['duration'].fillna(0).astype(float) < 900
flags_df = pd.DataFrame({
    'Farmercode': main_df['Farmercode'], 'username': main_df.get('username',''),
    'Labour':LM, 'Environmental':EM, 'Agrochemical':AG, 'Agronomic':AM,
    'PostHarvest':PMis, 'Phone':PhM, 'Time':TM
})
long = flags_df.melt(['Farmercode','username'], var_name='check', value_name='flag')
inc_df = long[long['flag']].drop('flag',1)
map_text = {
    'Labour':'Labour-Noncompliance-Mismatch','Environmental':'Environmental-Noncompliance-Mismatch',
    'Agrochemical':'Agrochemical-Noncompliance-Mismatch','Agronomic':'Agronomic-Noncompliance-Mismatch',
    'PostHarvest':'PostHarvest-Noncompliance-Mismatch','Phone':'Phone number mismatch','Time':'Time inconsistency: Inspection < 15 mins'
}
inc_df['inconsistency'] = inc_df['check'].map(map_text)
# Overlap inconsistencies
overlaps_list = []
for farmer,(ov,pct) in OVERLAPS.items():
    if pct >= 5:
        usr = main_df[main_df['Farmercode']==farmer]['username'].iloc[0]
        txt = 'Overlap > 10%' if pct>10 else 'Overlap 5-10%'
        overlaps_list.append({'Farmercode':farmer,'username':usr,'inconsistency':txt})
ov_df = pd.DataFrame(overlaps_list)
all_incons = pd.concat([inc_df[['Farmercode','username','inconsistency']], ov_df], ignore_index=True)

# ---------------------------
# 9. AGGREGATE + RISK + TRUST
# ---------------------------
risk_func = lambda t: 'High' if re.search('noncompliance-mismatch|less than expected|more than expected|time inconsistency', t, re.I) else ('Medium' if re.search('Overlap', t, re.I) else 'Low')
agg = all_incons.groupby(['Farmercode','username'], as_index=False).agg({'inconsistency':lambda x:', '.join(x.unique())})
agg['Risk Rating'] = agg['inconsistency'].apply(risk_func)
agg['Trust Responses'] = np.where(agg['Risk Rating']=='High','No','Yes')

# ---------------------------
# 10. INSPECTOR SCORING
# ---------------------------
def score_row(r):
    s=0
    if str(r.get('phone_match','')).lower()=='match': s+=1
    if r.get('duration',0)>900: s+=1
    if pd.notna(r.get('kgsold')) and pd.notna(r.get('harvestflyseason')) and pd.notna(r.get('totalharvest')) and r['kgsold']<=r['harvestflyseason']+r['totalharvest']: s+=1
    if pd.notna(r.get('acreagetotalplot')) and pd.notna(r.get('cocoaacreage')) and r['acreagetotalplot']<r['cocoaacreage']: s+=1
    if pd.notna(r.get('Productiveplants')) and r['Productiveplants']>=(r.get('youngplants',0)+r.get('stumpedplants',0)+r.get('shadeplants',0)): s+=1
    for flag_series in [LM,EM,AG,AM,PMis]:
        if not flag_series.loc[r.name]: s+=1
    if PM.loc[r.name]=='': s+=1
    _, pct = OVERLAPS[r['Farmercode']]
    if pct < 5: s+=1
    return s
main_df['total_rating'] = main_df.apply(score_row, axis=1)
inspector_rating = main_df.groupby('username')['total_rating'].mean().reset_index()

# ---------------------------
# 11. UI: BEST INSPECTORS
# ---------------------------
if 'district' in main_df.columns:
    districts = sorted(main_df['district'].dropna().unique())
    sel_d = st.selectbox('Select District', ['All Districts']+districts)
    if sel_d!='All Districts':
        inspector_rating = main_df[main_df['district']==sel_d].groupby('username')['total_rating'].mean().reset_index()
st.subheader('Best Inspectors by Average Rating')
if not inspector_rating.empty:
    top10 = inspector_rating.sort_values('total_rating',ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(top10['username'], top10['total_rating'])
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)
else:
    st.write('No rating data available.')

# ---------------------------
# 12. UI: FARMER INCONSISTENCIES & MAP
# ---------------------------
sel_farmer = st.selectbox('Select Farmer Code', GDF['Farmercode'].unique().tolist())
st.subheader(f'Inconsistencies for Farmer {sel_farmer}')
farmer_inc = agg[agg['Farmercode']==sel_farmer]
if not farmer_inc.empty:
    st.dataframe(farmer_inc)
else:
    st.write('No inconsistencies found.')

ov_list, ov_pct = OVERLAPS.get(sel_farmer, ([],0))
st.write(f"**Total Overlap Percentage:** {ov_pct:.2f}%")
if ov_list:
    fig2, ax2 = plt.subplots(figsize=(8,8))
    target_geom = GDF[GDF['Farmercode']==sel_farmer].geometry.iloc[0]
    if hasattr(target_geom,'exterior'):
        x,y = target_geom.exterior.xy
        ax2.fill(x,y,alpha=0.4,fc='blue',ec='black',label=f"Target {sel_farmer}")
    else:
        for part in target_geom.geoms:
            x,y = part.exterior.xy
            ax2.fill(x,y,alpha=0.4,fc='blue',ec='black')
    for o in ov_list:
        inter = o['intersection']
        if hasattr(inter,'exterior'):
            x,y = inter.exterior.xy
            ax2.fill(x,y,alpha=0.5,fc='red',ec='darkred')
        else:
            for part in inter.geoms:
                x,y = part.exterior.xy
                ax2.fill(x,y,alpha=0.5,fc='red',ec='darkred')
    ax2.legend(loc='upper right',fontsize='small')
    st.pyplot(fig2)
else:
    st.success('No overlaps for this farmer.')

# ---------------------------
# 13. EXPORT FUNCTION
# ---------------------------
def prepare_export(df):
    df_out = df.copy()
    df_out = pd.concat([df_out, PM], axis=1)
    df_out['Acres'] = gpd.GeoSeries(df_out['geometry'], crs=GDF.crs).area * 0.000247105
    df_out['geometry'] = df_out['geometry'].apply(lambda g: g.wkt if g else '')
    merged = df_out.merge(agg, on=['Farmercode','username'], how='left')
    merged[['inconsistency','Risk Rating','Trust Responses']] = merged[['inconsistency','Risk Rating','Trust Responses']].fillna({
        'inconsistency':'No Inconsistency','Risk Rating':'None','Trust Responses':'Yes'
    })
    merged['total_rating'] = merged.apply(score_row, axis=1)
    avg = merged.groupby('username')['total_rating'].mean().reset_index().rename(columns={'total_rating':'average_rating_per_username'})
    return merged.merge(avg, on='username', how='left')

if st.button('Export Updated Form to Excel'):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        prepare_export(main_df).to_excel(writer, index=False, sheet_name='Updated')
    buf.seek(0)
    st.download_button(
        'Download Excel', data=buf,
        file_name='updated_inspection_form.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
