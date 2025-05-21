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
                    df = _try(lambda: pd.read_csv(file, engine='python', sep=None, skip_blank_lines=True, on_bad_lines='skip'))
                    if df.empty or df.columns.size == 0:
                        raise ValueError("File appears empty or has no columns.")
    except Exception as e:
        st.error(f"Error loading {file.name}: {e}")
        st.stop()
    df.columns = df.columns.map(str).str.strip()
    return df

main_df = _read_tabular(main_file)
redo_df = _read_tabular(redo_file)

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
redo_df = redo_df.rename(columns={'farmer_code': 'Farmercode'})

# ---------------------------
# 3. MERGE REDO POLYGONS
# ---------------------------
if {"SubmissionDate","endtime"}.issubset(redo_df.columns):
    redo_df['SubmissionDate'] = pd.to_datetime(redo_df['SubmissionDate'], errors='coerce')
    redo_df['endtime'] = pd.to_datetime(redo_df['endtime'], errors='coerce')
    redo_df = (
        redo_df.sort_values(['Farmercode','SubmissionDate','endtime'])
        .drop_duplicates('Farmercode', keep='last')
        .rename(columns={'selectplot':'redo_selectplot','polygonplot':'redo_polygonplot'})
    )
    main_df = main_df.merge(
        redo_df[['Farmercode','redo_selectplot','redo_polygonplot']],
        on='Farmercode', how='left'
    )
    lookup = {'Plot1':'polygonplot','Plot2':'polygonplotnew_1','Plot3':'polygonplotnew_2','Plot4':'polygonplotnew_3','Plot5':'polygonplotnew_4'}
    for sel,col in lookup.items():
        if col in main_df:
            mask = (main_df['redo_selectplot']==sel)&main_df[col].notna()
            main_df.loc[mask,col]=main_df.loc[mask,'redo_polygonplot'].astype(str)
    main_df.drop(columns=['redo_selectplot','redo_polygonplot'],errors='ignore', inplace=True)

# ---------------------------
# 4. DATE FILTER
# ---------------------------
date_col = 'Submissiondate'
if 'Submissiondate' in main_df:
    main_df[date_col] = pd.to_datetime(main_df['Submissiondate'], errors='coerce')
elif 'SubmissionDate' in main_df:
    main_df[date_col] = pd.to_datetime(main_df['SubmissionDate'], errors='coerce')
if main_df[date_col].notna().any():
    mn, mx = main_df[date_col].min().date(), main_df[date_col].max().date()
    sel = st.slider("Select Submission Date Range", mn, mx, (mn,mx))
    main_df = main_df[(main_df[date_col].dt.date>=sel[0])&(main_df[date_col].dt.date<=sel[1])]
else:
    st.warning("Submission date not available; skipping filter.")

# ---------------------------
# 5. GEOMETRY & STRtree
# ---------------------------
POLY_COLS = [c for c in main_df.columns if c.startswith('polygonplot')]

@st.cache_resource
def build_gdf(df):
    def str2poly(s):
        if not isinstance(s,str): return None
        arr = np.fromstring(re.sub('[;,]',' ',s),sep=' ')
        if arr.size<6 or arr.size%3: return None
        pts=arr.reshape(-1,3)[:,:2]
        try: return Polygon(pts)
        except: return None
    geoms=[]
    for _,r in df[POLY_COLS].iterrows():
        parts=[str2poly(r[c]) for c in POLY_COLS if pd.notna(r[c])]
        parts=[p.buffer(0) if p and not p.is_valid else p for p in parts if p]
        geoms.append(parts[0] if len(parts)==1 else unary_union(parts) if parts else None)
    gdf=gpd.GeoDataFrame(df.copy(),geometry=geoms,crs='EPSG:4326')
    gdf=gdf[gdf.geometry.notna()].to_crs('EPSG:2109')
    gdf.geometry=gdf.geometry.buffer(0)
    tree=STRtree(gdf.geometry)
    idx_map={geom:i for i,geom in enumerate(gdf.geometry)}
    return gdf,tree,idx_map

GDF,STR_TREE,IDX=build_gdf(main_df)

# ---------------------------
# 6. OVERLAP PRECOMPUTE
# ---------------------------
@st.cache_resource
def precompute(gdf,tree,idxmap):
    cache={}
    for i,r in gdf.iterrows():
        tgt=r.geometry;ta=tgt.area;u=None;ov=[]
        for g in tree.query(tgt):
            j=idxmap[g]
            if j==i: continue
            intr=tgt.intersection(g)
            if intr.is_empty: continue
            a=intr.area;ov.append({'Farmercode':gdf.iloc[j]['Farmercode'],'overlap_area':a,'total_area':ta,'intersection':intr})
            u=intr if u is None else u.union(intr)
        pct=(u.area/ta*100) if u else 0
        cache[r['Farmercode']]=(ov,pct)
    return cache

OVERLAPS=precompute(GDF,STR_TREE,IDX)
def check_overlaps(code): return OVERLAPS.get(code,([],0))

# ---------------------------
# 7. PRODUCTIVE PLANTS METRICS
# ---------------------------
@st.cache_resource
def prod_metrics(df):
    ar=[c for c in df if re.search('acres_polygonplot',c,re.I)]
    pr=[c for c in df if re.search('productiveplants',c,re.I)]
    A=df[ar].fillna(0).astype(float).sum(1)
    P=df[pr].fillna(0).astype(float).sum(1)
    E=A*450
    return pd.DataFrame({'Total_Area':A,'Total_Productive_Plants':P,'Expected_Plants':E,'Half_Expected_Plants':E/2,'Pct125_Expected_Plants':E*1.25,'Productive_Plants_Inconsistency':np.where(P<E/2,'Less than Expected Productive Plants',np.where(P>1.25*E,'More than expected Productive Plants',''))})

PM=prod_metrics(main_df)

# ---------------------------
# 8. NONCOMPLIANCE FLAGS
# ---------------------------
def f_yes(s,*vs): return s.astype(str).str.strip().str.lower().isin([v.lower() for v in vs])
LM=(f_yes(main_df['childrenworkingconditions'],'any_time_when_needed')|f_yes(main_df['prisoners'],'yes')|f_yes(main_df['contractsworkers'],'no')|f_yes(main_df['drinkingwaterworkers'],'no'))&(main_df['noncompliancesfound_Labour'].fillna(0)==0)
EM=(f_yes(main_df['cutnativetrees'],'yes')|f_yes(main_df['cutforests'],'yes')|f_yes(main_df['toiletdischarge'],'yes')|f_yes(main_df['separatewaste'],'no'))&(main_df['noncompliancesfound_Environmental'].fillna(0)==0)
AG=(main_df[[ 'methodspestdiseasemanagement_using_chemicals','fertilizerchemicals_Pesticides','fertilizerchemicals_Fungicides','fertilizerchemicals_Herbicides','childrenlabouractivities_spraying_of_chemicals','typeworkvulnerable_Spraying_of_chemicals','agriculturalinputs_synthetic_chemicals_or_fertilize']].fillna(0).astype(float).eq(1).any(1))&(main_df['noncompliancesfound_Agro_chemical'].fillna(0)==0)
AM=(f_yes(main_df['pruning'],'no')|f_yes(main_df['desuckering'],'no')|f_yes(main_df['manageweeds'],'no')|f_yes(main_df['knowledgeIPM'],'no'))&(main_df['noncompliancesfound_Agronomic'].fillna(0)==0)
PMis=(f_yes(main_df['ripepods'],'no')|f_yes(main_df['storedrycocoa'],'no')|f_yes(main_df['separatebasins'],'no'))&(main_df['noncompliancesfound_Harvest_and_postharvestt'].fillna(0)==0)
PhM=~f_yes(main_df['phone_match'],'match')
TM=(main_df['duration'].fillna(0).astype(float)<900)
flags=pd.DataFrame({'Farmercode':main_df['Farmercode'],'username':main_df.get('username',''),'Labour':LM,'Environmental':EM,'Agrochemical':AG,'Agronomic':AM,'PostHarvest':PMis,'Phone':PhM,'Time':TM})
mp=flags.melt(['Farmercode','username'],None,'val');inc=mp[mp['val']].drop('val',1)
mapf={'Labour':'Labour-Noncompliance-Mismatch','Environmental':'Environmental-Noncompliance-Mismatch','Agrochemical':'Agrochemical-Noncompliance-Mismatch','Agronomic':'Agronomic-Noncompliance-Mismatch','PostHarvest':'PostHarvest-Noncompliance-Mismatch','Phone':'Phone number mismatch','Time':'Time inconsistency: Inspection < 15 mins'}
inc['inconsistency']=inc['variable'].map(mapf)
# overlap flags
ovrs=[]
for f,(o,p) in OVERLAPS.items():
    if p>=5:
        u=main_df[main_df['Farmercode']==f].iloc[0]
        txt='Overlap >10%' if p>10 else 'Overlap 5-10%'
        ovrs.append({'Farmercode':f,'username':u.get('username',''),'inconsistency':txt})
inc=pd.concat([inc[['Farmercode','username','inconsistency']],pd.DataFrame(ovrs)],0,ignore_index=True)

# ---------------------------
# 9. AGGREGATE + RISK + TRUST
# ---------------------------
risk=lambda t: 'High' if re.search('noncompliance-mismatch|less than expected|more than expected|time inconsistency',t,re.I) else 'Medium' if re.search('overlap >',t,re.I) else 'Low'
ag=inc.groupby(['Farmercode','username']).agg({'inconsistency':lambda s:','.join(s.unique())}).reset_index()
ag['Risk Rating']=ag['inconsistency'].apply(risk)
ag['Trust Responses']=np.where(ag['Risk Rating']=='High','No','Yes')

# ---------------------------
# 10. INSPECTOR SCORING
# ---------------------------
def row_score(r):
    s=0
    if str(r.get('phone_match','')).lower()=='match': s+=1
    if r.get('duration',0)>900: s+=1
    if pd.notna(r.get('kgsold')) and pd.notna(r.get('harvestflyseason')) and pd.notna(r.get('totalharvest')) and r['kgsold']<=r['harvestflyseason']+r['totalharvest']: s+=1
    if pd.notna(r.get('acreagetotalplot')) and pd.notna(r.get('cocoaacreage')) and r['acreagetotalplot']<r['cocoaacreage']: s+=1
    if pd.notna(r.get('Productiveplants')) and r['Productiveplants']>=(r.get('youngplants',0)+r.get('stumpedplants',0)+r.get('shadeplants',0)): s+=1
    for flag in [LM,EM,AG,AM,PMis]:
        if not flag.loc[r.name]: s+=1
    if PM.loc[r.name]=='': s+=1
    _,p=OVERLAPS[r['Farmercode']]
    if p<5: s+=1
    return s
main_df['total_rating']=main_df.apply(row_score,1)
rating=main_df.groupby('username')['total_rating'].mean().reset_index()

# ---------------------------
# 11. UI: BEST INSPECTORS
# ---------------------------
if 'district' in main_df:
    dists=sorted(main_df['district'].dropna().unique())
    sel=st.selectbox('District',['All']+dists)
    if sel!='All': rating=main_df[main_df['district']==sel].groupby('username')['total_rating'].mean().reset_index()
st.subheader('Best Inspectors by Average Rating')
if not rating.empty:
    top=rating.sort_values('total_rating',False).head(10)
    fig,ax=plt.subplots(figsize=(10,6))
    ax.bar(top['username'],top['total_rating'])
    plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
    st.pyplot(fig)
else:
    st.write('No rating data available.')

# ---------------------------
# 12. UI: FARMER INCONSISTENCIES & MAP
# ---------------------------
sel=st.selectbox('Select Farmer Code',GDF['Farmercode'].unique())
st.subheader(f'Inconsistencies for Farmer {sel}')
df_sel=ag[ag['Farmercode']==sel]
if not df_sel.empty: st.dataframe(df_sel)
else: st.write('No inconsistencies found.')
r,gpct=check_overlaps(sel)
st.subheader('Overlap Map')
if r:
    fig2,ax2=plt.subplots(figsize=(8,8))
    tgt=GDF[GDF['Farmercode']==sel].geometry.iloc[0]
    xs,ys=(tgt.exterior.xy if hasattr(tgt,'exterior') else ([],[]))
    ax2.fill(xs,ys,alpha=0.4,fc='blue',ec='black')
    for o in r:
        x,y=(o['intersection'].exterior.xy if hasattr(o['intersection'],'exterior') else ([],[]))
        ax2.fill(x,y,alpha=0.5,fc='red',ec='darkred')
    st.pyplot(fig2)
else:
    st.success('No overlaps for this farmer.')

# ---------------------------
# 13. EXPORT
# ---------------------------
def prepare_export(df):
    out=df.copy()
    out=pd.concat([out,PM],1)
    out['Acres'] = gpd.GeoSeries(out['geometry'], crs=GDF.crs).area * 0.000247105
    out['geometry']=out['geometry'].apply(lambda g: g.wkt if g else '')
    merged=out.merge(ag,on=['Farmercode','username'],how='left')
    merged[['inconsistency','Risk Rating','Trust Responses']]=merged[['inconsistency','Risk Rating','Trust Responses']].fillna({'inconsistency':'No Inconsistency','Risk Rating':'None','Trust Responses':'Yes'})
    merged['total_rating']=merged.apply(row_score,1)
    avg=merged.groupby('username')['total_rating'].mean().reset_index().rename(columns={'total_rating':'average_rating_per_username'})
    return merged.merge(avg,on='username',how='left')

if st.button('Export Updated Form to Excel'):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='xlsxwriter') as w: prepare_export(main_df).to_excel(w,index=False,sheet_name='Updated')
    buf.seek(0)
    st.download_button('Download Excel',data=buf,file_name='updated_inspection_form.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
