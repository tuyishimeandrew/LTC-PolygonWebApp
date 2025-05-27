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
t ry:
    if redo_file.name.endswith('.xlsx'):
        df_redo = pd.read_excel(redo_file, engine='openpyxl')
    else:
        df_redo = pd.read_csv(redo_file)
except Exception as e:
    st.error(f"Error loading redo file: {e}")
    st.stop()

# Normalize redo columns
if 'farmer_code' in df_redo.columns:
    df_redo.rename(columns={'farmer_code': 'Farmercode'}, inplace=True)
required = ['Farmercode', 'selectplot', 'polygonplot']
if not all(c in df_redo.columns for c in required):
    st.error("Redo Polygon Form must contain 'Farmercode', 'selectplot', and 'polygonplot'.")
    st.stop()

# Merge latest redo per farmer
if set(['SubmissionDate','endtime']).issubset(df_redo.columns):
    df_redo['SubmissionDate'] = pd.to_datetime(df_redo['SubmissionDate'], errors='coerce')
    df_redo['endtime'] = pd.to_datetime(df_redo['endtime'], errors='coerce')
    df_redo = df_redo.sort_values(['SubmissionDate','endtime']).groupby('Farmercode', as_index=False).last()
    df_redo.rename(columns={'selectplot':'redo_selectplot','polygonplot':'redo_polygonplot'}, inplace=True)
    df = df.merge(df_redo[['Farmercode','redo_selectplot','redo_polygonplot']], on='Farmercode', how='left')
    # Apply redo polygons
    base = 'polygonplot'
    df.loc[df[base].notna() & (df.redo_selectplot=='Plot1'), base] = df.loc[df[base].notna() & (df.redo_selectplot=='Plot1'),'redo_polygonplot'].astype(str)
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
    sel = st.slider("Select Submission Date Range", mn, mx, (mn,mx))
    df = df[df['Submissiondate'].dt.date.between(sel[0], sel[1])]
else:
    st.warning("No submission dates; skipping date filter.")

# ---------------------------
# POLYGON PARSING & COMBINING
# ---------------------------
def parse_polygon_z(s):
    if not isinstance(s, str): return None
    pts = [tuple(map(float, p.strip().split()[:2])) for p in s.split(';') if p.strip() and len(p.split())>=3]
    return Polygon(pts) if len(pts)>=3 else None

def combine_polygons(row):
    cols = [c for c in ['polygonplot','polygonplotnew_1','polygonplotnew_2','polygonplotnew_3','polygonplotnew_4'] if c in row]
    polys=[]
    for c in cols:
        poly = parse_polygon_z(row[c])
        if poly is not None:
            if not poly.is_valid: poly = poly.buffer(0)
            if poly.is_valid: polys.append(poly)
    if not polys: return None
    return polys[0] if len(polys)==1 else unary_union(polys)

# Build GeoDataFrame
df['geometry'] = df.apply(combine_polygons, axis=1)
df = df[df['geometry'].notna()]
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326').to_crs('EPSG:2109')
gdf['geometry'] = gdf['geometry'].buffer(0)
gdf = gdf[gdf.is_valid]

# ---------------------------
# OVERLAP CHECK
# ---------------------------
def check_overlaps(gdf, code):
    tgt = gdf[gdf['Farmercode']==code]
    if tgt.empty: return [], 0
    poly = tgt.geometry.iloc[0]
    tot = poly.area
    idxs = list(gdf.sindex.intersection(poly.bounds))
    union=None; overlaps=[]
    for i in idxs:
        r = gdf.iloc[i]
        if r.Farmercode==code: continue
        if not poly.intersects(r.geometry): continue
        inter = poly.intersection(r.geometry)
        if inter.is_empty: continue
        a = inter.area
        overlaps.append({'Farmercode':r.Farmercode, 'overlap_area':a, 'total_area':tot, 'intersection':inter})
        union = inter if union is None else union.union(inter)
    pct = union.area/tot*100 if union is not None else 0
    return overlaps, pct

# Precompute and map overlap percentages
overlap_dict = {code: check_overlaps(gdf, code)[1] for code in gdf['Farmercode'].unique()}
gdf['overlap_pct'] = gdf['Farmercode'].map(overlap_dict)

# ---------------------------
# METRICS & NONCOMPLIANCE CHECKS
# ---------------------------
# (Compute productive plants, labour, environmental, agrochemical, agronomic, postharvest, phone, time checks as before)
# [Functions omitted for brevity but include compute_productive_plants_metrics, check_labour_mismatch, etc.]

# ---------------------------
# AGGREGATED INCONSISTENCIES & RATINGS
# ---------------------------
# [Same logic as original: collect inconsistencies_list, overlap_incons_list, agg_incons, compute_rating, inspector_rating]

# ---------------------------
# UI: Best Inspectors Chart, Farmer Selection, Plots
# ---------------------------
# [Same as original: district filter, bar chart, selectbox for farmer, dataframes and plots]

# ---------------------------
# EXPORT FUNCTION WITH CLEANING FLAGS
# ---------------------------
def export_with_inconsistencies_merged(main_df, agg_incons_df):
    export_df = main_df.copy()
    # inconsistency flags
    flags = export_df.apply(lambda r: pd.Series(get_inconsistency_flags(r)), axis=1)
    export_df = pd.concat([export_df, flags], axis=1)
    # geometry to WKT and acres
    export_df['Acres'] = gpd.GeoSeries(export_df['geometry'], crs=gdf.crs).area * 0.000247105
    export_df['geometry'] = export_df['geometry'].apply(lambda g: g.wkt)
    merged_df = export_df.merge(
        agg_incons_df[['Farmercode','username','inconsistency','Risk Rating','Trust Responses']],
        on=['Farmercode','username'], how='left'
    )
    merged_df[['inconsistency','Risk Rating','Trust Responses']] = merged_df[['inconsistency','Risk Rating','Trust Responses']].fillna({'inconsistency':'No Inconsistency','Risk Rating':'None','Trust Responses':'Yes'})
    # recalc ratings
    merged_df['total_rating'] = merged_df.apply(compute_rating, axis=1)
    avg_rt = merged_df.groupby('username')['total_rating'].mean().reset_index().rename(columns={'total_rating':'average_rating_per_username'})
    merged_df = merged_df.merge(avg_rt, on='username', how='left')
    # Data cleaning flags
    # 1. number_fields
    plot_cols = [c for c in ['polygonplot','polygonplotnew_1','polygonplotnew_2','polygonplotnew_3','polygonplotnew_4'] if c in merged_df]
    merged_df['number_fields'] = merged_df[plot_cols].notna().sum(axis=1)
    merged_df['number_fields_flag'] = merged_df['number_fields'] > 5
    # 2. acreage farm vs cocoa
    merged_df['acreage_farm_vs_cocoa_flag'] = (
        (merged_df['total_acreage_farm'] < merged_df['total_acreage_cocoa']) |
        (merged_df['total_acreage_farm'] < 10) | (merged_df['total_acreage_farm'] > 20) |
        (merged_df['total_acreage_cocoa'] < 10) | (merged_df['total_acreage_cocoa'] > 20)
    )
    # 3. ID format
    merged_df['ID_format_flag'] = (
        merged_df['IDtype'].str.lower()=='national_id') & ~merged_df['IDnumber'].str.startswith(('CM','CF'), na=False)
    # 4. children vs schoolaged
    merged_df['children_vs_schoolaged_flag'] = merged_df['children'] < merged_df['schoolaged']
    # 5. attending school vs schoolaged
    merged_df['attendingschool_vs_schoolaged_flag'] = merged_df['attendingschool'] > merged_df['schoolaged']
    # 6. kgsold vs totalharvest
    merged_df['kgsold_vs_totalharvest_flag'] = merged_df['kgsold'] > merged_df['totalharvest']
    # 7. sale to Latitude
    merged_df['sale_to_latitude_flag'] = (merged_df['salesmadeto_Latitude']==0) & (merged_df['kgsold']>0)
    # 8. plot acreage vs farm
    merged_df['plot_acreage_vs_farm_flag'] = merged_df['acreage_totalplot'] > merged_df['total_acreage_farm']
    # 9. cocoa vs plot
    merged_df['cocoa_acreage_vs_plot_flag'] = merged_df['cocoa_acreage'] > merged_df['acreage_totalplot']
    # 10. productiveplants vs expected >125%
    merged_df['productiveplants_vs_expected_flag'] = merged_df['Productiveplants'] > (merged_df['total_acreage_farm']*450*1.25)
    # 11. plot vs farm diff >=2
    merged_df['plot_vs_farm_diff_flag'] = (merged_df['acreage_totalplot'] - merged_df['total_acreage_farm']) >= 2
    # 12. noncompliance without advice
    nc_pairs = [
        ('noncompliancesfound_Agro_chemical','Agrochemical_Noncompliance_Advice'),
        ('noncompliancesfound_Harvest_and_postharvestt','PostHarvest_Noncompliance_Advice'),
        ('noncompliancesfound_Agronomic','Agronomic_Noncompliance_Advice'),
        ('noncompliancesfound_Environmental','Environmental_Noncompliance_Advice'),
        ('noncompliancesfound_Labour','Labour_Noncompliance_Advice')
    ]
    mask_nc_no_adv = False
    for found, adv in nc_pairs:
        mask_nc_no_adv |= (merged_df.get(found,0)>0) & (merged_df.get(adv)=='None of the above')
    merged_df['noncompliance_without_advice_flag'] = mask_nc_no_adv
    # 13. agrochemical violations found
    merged_df['agrochemical_violations_flag'] = merged_df.get('noncompliancesfound_Agro_chemical',0) > 0
    # reorder flags last
    flag_cols = [c for c in merged_df.columns if c.endswith('_flag')]
    cols = [c for c in merged_df.columns if c not in flag_cols] + flag_cols
    merged_df = merged_df[cols]
    # export to Excel
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        merged_df.to_excel(writer, index=False, sheet_name='Updated_Form')
    buf.seek(0)
    st.download_button(
        label="Download Updated Form with Flags",
        data=buf,
        file_name="updated_inspection_form_with_flags.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if st.button("Export Updated Form to Excel"):
    export_with_inconsistencies_merged(gdf, agg_incons)
