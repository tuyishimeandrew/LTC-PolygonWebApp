import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
import io

st.set_page_config(layout="wide")
st.title("Latitude Inspections Inconsistency Checker")

# ---------------------------
# FILE UPLOAD
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Main Inspection File")
    main_file = st.file_uploader("Upload Main Inspection Form (CSV or Excel)", type=["xlsx", "csv"], key="main_upload")
with col2:
    st.subheader("Redo Polygon File")
    redo_file = st.file_uploader("Upload Redo Polygon Form (CSV or Excel)", type=["xlsx", "csv"], key="redo_upload")

if main_file is None or redo_file is None:
    st.info("Please upload both Main Inspection and Redo Polygon files.")
    st.stop()

# Load main file
try:
    if main_file.name.endswith('.xlsx'):
        df = pd.read_excel(main_file, engine='openpyxl')
    else:
        df = pd.read_csv(main_file)
except Exception as e:
    st.error(f"Error loading main file: {e}")
    st.stop()

# Validate main columns
if 'Farmercode' not in df.columns or 'polygonplot' not in df.columns:
    st.error("Main Inspection Form must contain 'Farmercode' and 'polygonplot'.")
    st.stop()

# Load redo file
try:
    if redo_file.name.endswith('.xlsx'):
        df_redo = pd.read_excel(redo_file, engine='openpyxl')
    else:
        df_redo = pd.read_csv(redo_file)
except Exception as e:
    st.error(f"Error loading redo file: {e}")
    st.stop()

# Normalize redo columns
if 'farmer_code' in df_redo.columns:
    df_redo = df_redo.rename(columns={'farmer_code': 'Farmercode'})
required = ['Farmercode', 'selectplot', 'polygonplot']
if not all(c in df_redo.columns for c in required):
    st.error("Redo Polygon Form must contain 'Farmercode', 'selectplot', 'polygonplot'.")
    st.stop()

# Merge latest redo per farmer
if set(['SubmissionDate','endtime']).issubset(df_redo.columns):
    df_redo['SubmissionDate'] = pd.to_datetime(df_redo['SubmissionDate'], errors='coerce')
    df_redo['endtime'] = pd.to_datetime(df_redo['endtime'], errors='coerce')
    df_redo = df_redo.sort_values(['SubmissionDate','endtime']).groupby('Farmercode', as_index=False).last()
    df_redo = df_redo.rename(columns={'selectplot':'redo_selectplot','polygonplot':'redo_polygonplot'})
    df = df.merge(df_redo[['Farmercode','redo_selectplot','redo_polygonplot']], on='Farmercode', how='left')
    # apply
    base_plot = 'polygonplot'
    df.loc[df[base_plot].notna() & (df.redo_selectplot=='Plot1'), base_plot] = df.loc[df[base_plot].notna() & (df.redo_selectplot=='Plot1'), 'redo_polygonplot'].astype(str)
    for col, val in zip(['polygonplotnew_1','polygonplotnew_2','polygonplotnew_3','polygonplotnew_4'], ['Plot2','Plot3','Plot4','Plot5']):
        if col in df.columns:
            mask = df[col].notna() & (df.redo_selectplot==val)
            df.loc[mask, col] = df.loc[mask, 'redo_polygonplot'].astype(str)
    df.drop(['redo_selectplot','redo_polygonplot'], axis=1, inplace=True)

# ---------------------------
# DATE FILTER
# ---------------------------
if 'Submissiondate' in df.columns or 'SubmissionDate' in df.columns:
    col = 'Submissiondate' if 'Submissiondate' in df.columns else 'SubmissionDate'
    df['Submissiondate'] = pd.to_datetime(df[col], errors='coerce')
else:
    df['Submissiondate'] = pd.NaT
if df['Submissiondate'].notna().any():
    mn, mx = df['Submissiondate'].min().date(), df['Submissiondate'].max().date()
    rng = st.slider("Submission Date Range", mn, mx, (mn,mx))
    df = df[df['Submissiondate'].dt.date.between(rng[0], rng[1])]
else:
    st.warning("No submission dates available; skipping date filter.")

# ---------------------------
# POLYGON PARSING
# ---------------------------
def parse_polygon(s):
    if not isinstance(s,str): return None
    pts = [tuple(map(float,p.split()[:2])) for p in s.split(';') if len(p.split())>=3]
    return Polygon(pts) if len(pts)>=3 else None

def combine(row):
    cols = [c for c in ['polygonplot','polygonplotnew_1','polygonplotnew_2','polygonplotnew_3','polygonplotnew_4'] if c in row]
    polys = [parse_polygon(row[c]) for c in cols if pd.notna(row[c])]
    valid=[]
    for p in polys:
        if p and not p.is_valid: p = p.buffer(0)
        if p and p.is_valid: valid.append(p)
    return valid[0] if len(valid)==1 else unary_union(valid) if valid else None

# build GeoDataFrame
df['geometry']=df.apply(combine,axis=1)
df=df[df.geometry.notna()]
gdf=gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326').to_crs('EPSG:2109')
gdf.geometry=gdf.geometry.buffer(0)
gdf=gdf[gdf.is_valid]

# ---------------------------
# OVERLAP CHECK
# ---------------------------
def check_overlaps(gdf, code):
    tgt=gdf[gdf.Farmercode==code]
    if tgt.empty: return [],0
    poly=tgt.geometry.iloc[0]
    total=poly.area
    idxs=list(gdf.sindex.intersection(poly.bounds))
    union=None; overlaps=[]
    for i in idxs:
        r=gdf.iloc[i]
        if r.Farmercode==code: continue
        if not poly.intersects(r.geometry): continue
        inter=poly.intersection(r.geometry)
        if inter.is_empty: continue
        a=inter.area
        overlaps.append({'Farmercode':r.Farmercode,'overlap_area':a,'total_area':total,'intersection':inter})
        union=inter if union is None else union.union(inter)
    pct=union.area/total*100 if union is not None else 0
    return overlaps, pct

overlap_dict={code: check_overlaps(gdf,code)[1] for code in gdf.Farmercode.unique()}
gdf['overlap_pct']=gdf.Farmercode.map(overlap_dict)

# ---------------------------
# NONCOMPLIANCE & METRICS
# ---------------------------
def compute_productive(row):
    area=sum(float(row[c]) for c in row.index if 'acres_polygonplot' in c.lower() and pd.notna(row[c]) )
    plants=sum(float(row[c]) for c in row.index if 'productiveplants' in c.lower() and pd.notna(row[c]))
    exp=area*450; half=exp/2; pct125=exp*1.25
    inc='';
    if plants<half: inc='Less than Expected'
    elif plants>pct125: inc='More than expected'
    return area,plants,exp,half,pct125,inc

def check_flag(cond): return cond

def check_labour(row):
    c=row.get; f= lambda x: str(c(x,'')).lower()
    cond = f('childrenworkingconditions')=='any_time_when_needed' or f('prisoners')=='yes' or f('contractsworkers')=='no' or f('drinkingwaterworkers')=='no'
    return cond, float(row.get('noncompliancesfound_Labour',0))
def check_environment(row):
    f=lambda x: str(row.get(x,'')).lower()
    cond=f('cutnativetrees')=='yes' or f('cutforests')=='yes' or f('toiletdischarge')=='yes' or f('separatewaste')=='no'
    return cond, float(row.get('noncompliancesfound_Environmental',0))
def check_agro(row):
    cols=['methodspestdiseasemanagement_using_chemicals','fertilizerchemicals_Pesticides','fertilizerchemicals_Fungicides','fertilizerchemicals_Herbicides','childrenlabouractivities_spraying_of_chemicals','typeworkvulnerable_Spraying_of_chemicals','agriculturalinputs_synthetic_chemicals_or_fertilize']
    cond=any(float(row.get(c,0))==1 for c in cols)
    return cond, float(row.get('noncompliancesfound_Agro_chemical',0))
def check_agronomic(row):
    cols=['pruning','desuckering','manageweeds','knowledgeIPM']
    cond=any(str(row.get(c,'')).lower()=='no' for c in cols)
    return cond, float(row.get('noncompliancesfound_Agronomic',0))
def check_post(row):
    cols=['ripepods','storedrycocoa','separatebasins']
    cond=any(str(row.get(c,'')).lower()=='no' for c in cols)
    return cond, float(row.get('noncompliancesfound_Harvest_and_postharvestt',0))
def get_flags(row):
    flags={}
    # counts
    area,plants,exp,half,pct125,inc=compute_productive(row)
    fns=[('Labour',check_labour),('Environmental',check_environment),('Agro_chemical',check_agro),('Agronomic',check_agronomic),('PostHarvest',check_post)]
    for name,fn in fns:
        cond,found=fn(row)
        flags[f'{name}_mismatch']= cond and found==0
        flags[f'{name}_found']= found>0
    flags['Productiveplants_flag']= plants>exp*1.25
    # other checks
    flags['ID_format_flag']= (str(row.get('IDtype','')).lower()=='national_id') and not str(row.get('IDnumber','')).startswith(('CM','CF'))
    flags['children_vs_schoolaged_flag']= row.get('children',0) < row.get('schoolaged',0)
    flags['attendingschool_vs_schoolaged_flag']= row.get('attendingschool',0) > row.get('schoolaged',0)
    flags['kgsold_vs_totalharvest_flag']= row.get('kgsold',0) > row.get('totalharvest',0)
    flags['sale_to_latitude_flag']= (row.get('salesmadeto_Latitude',0)==0) and row.get('kgsold',0)>0
    flags['plot_vs_farm_flag']= row.get('acreage_totalplot',0) > row.get('total_acreage_farm',0)
    flags['cocoa_vs_plot_flag']= row.get('cocoa_acreage',0) > row.get('acreage_totalplot',0)
    flags['plot_diff_flag']= (row.get('acreage_totalplot',0) - row.get('total_acreage_farm',0))>=2
    return flags

# apply data cleaning flags
dc_flags = pd.DataFrame(df.apply(get_flags, axis=1).tolist())
merged = pd.concat([df.reset_index(drop=True), dc_flags], axis=1)

# ---------------------------
# STREAMLIT UI and EXPORT
# ---------------------------
def to_excel(df):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as w:
        df.to_excel(w, index=False, sheet_name='Cleaned')
    buf.seek(0)
    return buf

st.subheader("Data Cleaning Flags Preview")
st.dataframe(merged.iloc[:100])

if st.button("Download Cleaned Data"):
    buf = to_excel(merged)
    st.download_button("Download Excel", data=buf, file_name='cleaned_inspection.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
