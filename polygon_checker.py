import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
import io

st.set_page_config(layout="wide")
st.title("Latitude Inspections Inconsistency & Cleaning Checker (Optimized)")

# ---------------------------
# FILE UPLOAD & DATA LOAD
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Main Inspection File")
    main_file = st.file_uploader("Upload Main Inspection Form (CSV or Excel)", type=["xlsx","csv"], key="main_upload")
with col2:
    st.subheader("Redo Polygon File")
    redo_file = st.file_uploader("Upload Redo Polygon Form (CSV or Excel)", type=["xlsx","csv"], key="redo_upload")
if not main_file or not redo_file:
    st.info("Please upload both Main Inspection and Redo Polygon files.")
    st.stop()
try:
    df = pd.read_excel(main_file, engine='openpyxl') if main_file.name.endswith('.xlsx') else pd.read_csv(main_file)
    df_redo = pd.read_excel(redo_file, engine='openpyxl') if redo_file.name.endswith('.xlsx') else pd.read_csv(redo_file)
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# ---------------------------
# MERGE LATEST REDO
# ---------------------------
if 'farmer_code' in df_redo.columns:
    df_redo.rename(columns={'farmer_code':'Farmercode'}, inplace=True)
if not {'Farmercode','selectplot','polygonplot'}.issubset(df_redo.columns):
    st.error("Redo file missing required columns: Farmercode, selectplot, polygonplot.")
    st.stop()
df_redo['SubmissionDate'] = pd.to_datetime(df_redo.get('SubmissionDate'), errors='coerce')
df_redo['endtime'] = pd.to_datetime(df_redo.get('endtime'), errors='coerce')
df_redo = df_redo.sort_values(['SubmissionDate','endtime']).groupby('Farmercode', as_index=False).last()
df_redo.rename(columns={'selectplot':'redo_selectplot','polygonplot':'redo_polygonplot'}, inplace=True)
df = df.merge(df_redo[['Farmercode','redo_selectplot','redo_polygonplot']], on='Farmercode', how='left')
for orig,plot in [('polygonplot','Plot1'),('polygonplotnew_1','Plot2'),('polygonplotnew_2','Plot3'),('polygonplotnew_3','Plot4'),('polygonplotnew_4','Plot5')]:
    if orig in df.columns:
        mask = df[orig].notna() & (df['redo_selectplot']==plot)
        df.loc[mask, orig] = df.loc[mask, 'redo_polygonplot']
df.drop(['redo_selectplot','redo_polygonplot'], axis=1, inplace=True)

# ---------------------------
# DATE FILTER
# ---------------------------
df['Submissiondate'] = pd.to_datetime(df.get('Submissiondate', df.get('SubmissionDate')), errors='coerce')
if df['Submissiondate'].notna().any():
    mn, mx = df['Submissiondate'].min().date(), df['Submissiondate'].max().date()
    sel = st.slider("Submission Date Range", mn, mx, (mn,mx))
    df = df[df['Submissiondate'].dt.date.between(sel[0], sel[1])]
else:
    st.warning("No submission dates found; skipping date filter.")

# ---------------------------
# PARSE & COMBINE POLYGONS (Optimized)
# ---------------------------
def parse_poly(s):
    if not isinstance(s,str): return None
    pts=[tuple(map(float,p.split()[:2])) for p in s.split(';') if len(p.split())>=2]
    return Polygon(pts) if len(pts)>=3 else None
poly_cols=[c for c in df.columns if c.startswith('polygonplot')]
for c in poly_cols:
    df[c] = df[c].map(parse_poly)
combined=[]
for polys in df[poly_cols].itertuples(index=False):
    valid=[p.buffer(0) if p and not p.is_valid else p for p in polys if p]
    combined.append(unary_union(valid) if len(valid)>1 else (valid[0] if valid else None))
df['geometry']=combined
df = df[df['geometry'].notna()]

# ---------------------------
# GeoDataFrame & CRS
# ---------------------------
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326').to_crs('EPSG:2109')
gdf['geometry'] = gdf['geometry'].buffer(0)
gdf = gdf[gdf.is_valid]

# ---------------------------
# PRECOMPUTE OVERLAP %
# ---------------------------
bounds_idx = gdf.sindex
areas = gdf.geometry.area
overlap_pct = {}
for i,row in gdf.iterrows():
    poly=row.geometry
    ints=None
    for j in bounds_idx.intersection(poly.bounds):
        if i==j: continue
        inter=poly.intersection(gdf.geometry.iat[j])
        if inter.area>1e-6:
            ints = inter if ints is None else ints.union(inter)
    overlap_pct[row['Farmercode']] = (ints.area/areas.iat[i]*100) if ints is not None else 0
gdf['overlap_pct'] = gdf['Farmercode'].map(overlap_pct)

# ---------------------------
# PRODUCTIVE PLANTS METRICS
# ---------------------------
temp = df.filter(regex='(?i)acres_polygonplot').sum(axis=1)
plants = df.filter(regex='(?i)productiveplants').sum(axis=1)
expected = temp*450
df['Total_Area']=temp
df['Total_Productive_Plants']=plants
df['Expected_Plants']=expected
df['Half_Expected_Plants']=expected/2
df['Pct125_Expected_Plants']=expected*1.25
def prod_inconsistency(pl,exp):
    if pl<exp/2: return 'Less than Expected'
    if pl>exp*1.25: return 'More than Expected'
    return ''
df['Productive_Plants_Inconsistency'] = df.apply(lambda r: prod_inconsistency(r['Total_Productive_Plants'],r['Expected_Plants']), axis=1)

# ---------------------------
# NONCOMPLIANCE & RATING FUNCTIONS
# ---------------------------
def labour_registered(r): return any(
    str(r.get(x,'')).strip().lower() in ['any_time_when_needed','yes','no']
    for x in ['childrenworkingconditions','prisoners','contractsworkers','drinkingwaterworkers'])
def check_labour_mismatch(r): return 'Labour-Noncompliance-Mismatch' if labour_registered(r) and r.get('noncompliancesfound_Labour',0)==0 else None
def check_environmental_mismatch(r): return 'Environmental-Noncompliance-Mismatch' if (str(r.get('cutnativetrees','')).lower()=='yes' or str(r.get('cutforests','')).lower()=='yes' or str(r.get('toiletdischarge','')).lower()=='yes' or str(r.get('separatewaste','')).lower()=='no') and r.get('noncompliancesfound_Environmental',0)==0 else None
def agro_registered(r):
    for x in ['methodspestdiseasemanagement_using_chemicals','fertilizerchemicals_Pesticides','fertilizerchemicals_Fungicides','fertilizerchemicals_Herbicides','childrenlabouractivities_spraying_of_chemicals','typeworkvulnerable_Spraying_of_chemicals','agriculturalinputs_synthetic_chemicals_or_fertilize']:
        try: 
            if float(r.get(x,0))==1: return True
        except: pass
    return False
def check_agrochemical_mismatch(r): return 'Agrochemical-Noncompliance-Mismatch' if agro_registered(r) and r.get('noncompliancesfound_Agro_chemical',0)==0 else None
def agronomic_registered(r): return any(str(r.get(x,'')).lower()=='no' for x in ['pruning','desuckering','manageweeds','knowledgeIPM'])
def check_agronomic_mismatch(r): return 'Agronomic-Noncompliance-Mismatch' if agronomic_registered(r) and r.get('noncompliancesfound_Agronomic',0)==0 else None
def postharvest_registered(r): return any(str(r.get(x,'')).lower()=='no' for x in ['ripepods','storedrycocoa','separatebasins'])
def check_postharvest_mismatch(r): return 'PostHarvest-Noncompliance-Mismatch' if postharvest_registered(r) and r.get('noncompliancesfound_Harvest_and_postharvestt',0)==0 else None
def check_phone_mismatch(r): return 'Phone number mismatch' if str(r.get('phone_match','')).lower()!='match' else None
def check_time_inconsistency(r): return 'Time inconsistency' if float(r.get('duration',0))<900 else None
def get_inconsistency_flags(r):
    flags={
        'Labour_Advice':check_labour_mismatch(r) or 'None',
        'Environmental_Advice':check_environmental_mismatch(r) or 'None',
        'Agrochemical_Advice':check_agrochemical_mismatch(r) or 'None',
        'Agronomic_Advice':check_agronomic_mismatch(r) or 'None',
        'PostHarvest_Advice':check_postharvest_mismatch(r) or 'None',
        'Phone_Advice':check_phone_mismatch(r) or 'None',
        'Time_Advice':check_time_inconsistency(r) or 'None',
        'Overlap_Advice': (f"Overlap {overlap_pct.get(r['Farmercode'],0):.2f}%") if overlap_pct.get(r['Farmercode'],0)>=5 else 'None'
    }
    flags.update({
        'Total_Area':r['Total_Area'],'Total_Productive_Plants':r['Total_Productive_Plants'],'Expected_Plants':r['Expected_Plants'],
        'Half_Expected_Plants':r['Half_Expected_Plants'],'Pct125_Expected_Plants':r['Pct125_Expected_Plants'],'Productive_Plants_Inconsistency':r['Productive_Plants_Inconsistency']
    })
    return pd.Series(flags)
def compute_rating(r):
    s=0
    s+=1 if str(r.get('phone_match','')).lower()=='match' else 0
    s+=1 if r.get('duration',0)>900 else 0
    try: s+=1 if r['kgsold']<=r['harvestflyseason']+r['totalharvest'] else 0
    except: pass
    s+=1 if r.get('acreagetotalplot',0)<r.get('cocoaacreage',0) else 0
    try: s+=1 if r['Productiveplants']>=(r['youngplants']+r['stumpedplants']+r['shadeplants']) else 0
    except: pass
    s+=1 if not(labour_registered(r) and r.get('noncompliancesfound_Labour',0)==0) else 0
    s+=1 if not((str(r.get('cutnativetrees','')).lower()=='yes') and r.get('noncompliancesfound_Environmental',0)==0) else 0
    s+=1 if not(agro_registered(r) and r.get('noncompliancesfound_Agro_chemical',0)==0) else 0
    s+=1 if not(agronomic_registered(r) and r.get('noncompliancesfound_Agronomic',0)==0) else 0
    s+=1 if not(postharvest_registered(r) and r.get('noncompliancesfound_Harvest_and_postharvestt',0)==0) else 0
    s+=1 if r['Productive_Plants_Inconsistency']=='' else 0
    s+=1 if overlap_pct.get(r['Farmercode'],0)<5 else 0
    return s
df['total_rating']=df.apply(compute_rating,axis=1)

# ---------------------------
# BEST INSPECTORS CHART
# ---------------------------
inspector_rating = df.groupby('username')['total_rating'].mean().reset_index()
if 'district' in df.columns:
    sel_dist=st.selectbox('Select District',['All']+sorted(df['district'].dropna().unique()))
    if sel_dist!='All':
        inspector_rating=df[df['district']==sel_dist].groupby('username')['total_rating'].mean().reset_index()
fig, ax = plt.subplots(figsize=(8,4))
best=inspector_rating.sort_values('total_rating',ascending=False).head(10)
ax.bar(best['username'],best['total_rating'])
ax.set_xticklabels(best['username'],rotation=45,ha='right')
st.pyplot(fig)

# ---------------------------
# FARMER INCONSISTENCY DISPLAY & MAP
# ---------------------------
agg = pd.concat([df.apply(lambda r: pd.Series(get_inconsistency_flags(r)),axis=1), df[['Farmercode','username']]], axis=1)
agg['Risk'] = agg.drop(columns=['Farmercode','username']).apply(lambda x: 'High' if any('Mismatch' in str(v) for v in x) else 'Low', axis=1)
agg_summary = agg.groupby(['Farmercode','username'], as_index=False).agg({'Risk':'max'})
farmer_list = gdf['Farmercode'].unique().tolist()
sel_farmer = st.selectbox('Select Farmer', farmer_list)
sub = agg_summary[agg_summary['Farmercode']==sel_farmer]
st.subheader(f"Inconsistencies for {sel_farmer}")
if not sub.empty:
    st.table(sub)
else:
    st.write("No inconsistencies detected.")

row = gdf[gdf['Farmercode']==sel_farmer]
if not row.empty:
    st.subheader("Overlap Map")
    base = row.geometry.plot(alpha=0.5, figsize=(6,6))
    for code, pct in overlap_pct.items():
        if code!=sel_farmer and pct>0:
            gdf[gdf['Farmercode']==code].geometry.plot(ax=base, alpha=0.3)
    st.pyplot(base.get_figure())

# ---------------------------
# EXPORT FUNCTION
# ---------------------------
def export_data():
    export_df = df.copy()
    flags = export_df.apply(lambda r: get_inconsistency_flags(r), axis=1)
    export_df = pd.concat([export_df, flags], axis=1)
    export_df['overlap_pct'] = export_df['Farmercode'].map(overlap_pct)
    # cleaning flags
    export_df['number_fields_flag']=export_df['number_fields']>5
    export_df['acreage_farm_vs_cocoa_flag']=(export_df['total_acreage_farm']<export_df['total_acreage_cocoa'])|(export_df['total_acreage_farm']>20)|(export_df['total_acreage_cocoa']>20)
    export_df['ID_format_flag']=(export_df['IDtype'].str.lower()=='national_id')&(~export_df['IDnumber'].str.startswith(('CM','CF')))
    export_df['children_vs_schoolaged_flag']=export_df['children']<export_df['schoolaged']
    export_df['attendingschool_vs_schoolaged_flag']=export_df['attendingschool']>export_df['schoolaged']
    export_df['kgsold_vs_totalharvest_flag']=export_df['kgsold']>export_df['totalharvest']
    export_df['sale_to_latitude_flag']=(export_df['salesmadeto_Latitude']==0)&(export_df['kgsold']>0)
    export_df['plot_acreage_vs_farm_flag']=export_df['acreage_totalplot']>export_df['total_acreage_farm']
    export_df['cocoa_acreage_vs_plot_flag']=export_df['cocoa_acreage']>export_df['acreage_totalplot']
    export_df['productiveplants_vs_expected_flag']=export_df['Total_Productive_Plants']>1.25*export_df['Total_Area']*450
    export_df['plot_vs_farm_diff_flag']=(export_df['acreage_totalplot']-export_df['total_acreage_farm'])>=2
    export_df['noncompliance_without_advice_flag']=((export_df['noncompliancesfound_Labour']>0)&(export_df['Labour_Advice']=='None'))|((export_df['noncompliancesfound_Environmental']>0)&(export_df['Environmental_Advice']=='None'))
    export_df['agrochemical_violations_flag']=export_df['noncompliancesfound_Agro_chemical']>0

    buf=io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        export_df.to_excel(writer, index=False, sheet_name='Results')
    buf.seek(0)
    st.download_button("Download Full Report", buf, "inspection_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if st.button('Export Full Report'):
    export_data()
