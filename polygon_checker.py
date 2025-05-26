import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
import io

st.title("Latitude Inspections Inconsistency & Cleaning Checker")

# ---------------------------
# FILE UPLOAD
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Main Inspection File")
    main_file = st.file_uploader(
        "Upload Main Inspection Form (CSV or Excel)",
        type=["xlsx", "csv"],
        key="main_upload"
    )
with col2:
    st.subheader("Redo Polygon File")
    redo_file = st.file_uploader(
        "Upload Redo Polygon Form (CSV or Excel)",
        type=["xlsx", "csv"],
        key="redo_upload"
    )

if main_file is None:
    st.info("Please upload the Main Inspection file and Redo file.")
    st.stop()

# ---------------------------
# LOAD DATA
# ---------------------------
try:
    if main_file.name.endswith('.xlsx'):
        df = pd.read_excel(main_file, engine='openpyxl')
    else:
        df = pd.read_csv(main_file)
except Exception as e:
    st.error(f"Error loading main file: {e}")
    st.stop()

if redo_file is None:
    st.error("The Redo Polygon Form is mandatory. Please upload the Redo file.")
    st.stop()
else:
    try:
        if redo_file.name.endswith('.xlsx'):
            df_redo = pd.read_excel(redo_file, engine='openpyxl')
        else:
            df_redo = pd.read_csv(redo_file)
    except Exception as e:
        st.error(f"Error loading redo polygon file: {e}")
        st.stop()

# ---------------------------
# MERGE LATEST REDO
# ---------------------------
if 'farmer_code' in df_redo.columns:
    df_redo = df_redo.rename(columns={'farmer_code': 'Farmercode'})
required = ['Farmercode', 'selectplot', 'polygonplot']
if not all(col in df_redo.columns for col in required):
    st.error("Redo polygon file must contain Farmercode, selectplot, polygonplot.")
    st.stop()
if 'SubmissionDate' in df_redo.columns and 'endtime' in df_redo.columns:
    df_redo['SubmissionDate'] = pd.to_datetime(df_redo['SubmissionDate'], errors='coerce')
    df_redo['endtime'] = pd.to_datetime(df_redo['endtime'], errors='coerce')
    df_redo = df_redo.sort_values(['SubmissionDate','endtime'])
    df_redo = df_redo.groupby('Farmercode', as_index=False).last()
    df_redo = df_redo.rename(columns={'selectplot':'redo_selectplot','polygonplot':'redo_polygonplot'})
    df = df.merge(df_redo[['Farmercode','redo_selectplot','redo_polygonplot']], on='Farmercode', how='left')
    # override
    cond1 = df['polygonplot'].notna() & (df['redo_selectplot']=='Plot1')
    df.loc[cond1,'polygonplot'] = df.loc[cond1,'redo_polygonplot']
    for col, plot in [('polygonplotnew_1','Plot2'),('polygonplotnew_2','Plot3'),
                      ('polygonplotnew_3','Plot4'),('polygonplotnew_4','Plot5')]:
        if col in df.columns:
            c = df[col].notna() & (df['redo_selectplot']==plot)
            df.loc[c,col] = df.loc[c,'redo_polygonplot']
    df = df.drop(['redo_selectplot','redo_polygonplot'], axis=1)

# ---------------------------
# DATE FILTER
# ---------------------------
if 'Submissiondate' in df.columns:
    df['Submissiondate'] = pd.to_datetime(df['Submissiondate'], errors='coerce')
elif 'SubmissionDate' in df.columns:
    df['Submissiondate'] = pd.to_datetime(df['SubmissionDate'], errors='coerce')
else:
    df['Submissiondate'] = pd.NaT
if df['Submissiondate'].notna().any():
    mn = df['Submissiondate'].min().date()
    mx = df['Submissiondate'].max().date()
    sel = st.slider("Select Submission Date Range", mn, mx, (mn,mx))
    df = df[df['Submissiondate'].dt.date.between(sel[0],sel[1])]
else:
    st.warning("No submission dates found; skipping date filter.")

# ---------------------------
# POLYGON PARSING & UNION
# ---------------------------
def parse_poly(s):
    if not isinstance(s,str): return None
    pts=[]
    for p in s.split(';'):
        t=p.strip().split()
        if len(t)>=2:
            pts.append((float(t[0]),float(t[1])))
    return Polygon(pts) if len(pts)>=3 else None

def combine_polys(r):
    cols=['polygonplot','polygonplotnew_1','polygonplotnew_2','polygonplotnew_3','polygonplotnew_4']
    ps=[parse_poly(r[c]) for c in cols if c in r and pd.notna(r[c])]
    valid=[]
    for p in ps:
        if p and not p.is_valid:
            p=p.buffer(0)
        if p and p.is_valid:
            valid.append(p)
    if not valid: return None
    return valid[0] if len(valid)==1 else unary_union(valid)

df['geometry'] = df.apply(combine_polys, axis=1)
df = df[df['geometry'].notna()]

gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
gdf = gdf.to_crs('EPSG:2109')
gdf['geometry'] = gdf['geometry'].buffer(0)
gdf = gdf[gdf.is_valid]

# ---------------------------
# OVERLAP CHECK
# ---------------------------
def check_overlaps(gdf, code):
    row = gdf[gdf['Farmercode']==code]
    if row.empty: return [],0
    poly=row.geometry.iloc[0]
    ta=poly.area
    idxs=list(gdf.sindex.intersection(poly.bounds))
    ints=[]
    union=None
    for i in idxs:
        r=gdf.iloc[i]
        if r['Farmercode']==code: continue
        if poly.intersects(r.geometry):
            inter=poly.intersection(r.geometry)
            if not inter.is_empty and inter.area>1e-6:
                ints.append({'Farmercode':r['Farmercode'],'overlap_area':inter.area})
                union = inter if union is None else union.union(inter)
    pct = (union.area/ta*100) if union else 0
    return ints, pct

# ---------------------------
# PRODUCTIVE PLANTS METRICS
# ---------------------------
def compute_productive_plants_metrics(r):
    area=0; plants=0
    for c,v in r.items():
        lc=c.lower()
        if 'acres_polygonplot' in lc and pd.notna(v): area+=float(v)
        if 'productiveplants' in lc and pd.notna(v): plants+=float(v)
    exp=area*450
    return pd.Series({
        'Total_Area':area,
        'Total_Productive_Plants':plants,
        'Expected_Plants':exp,
        'Half_Expected_Plants':exp/2,
        'Pct125_Expected_Plants':exp*1.25,
        'Productive_Plants_Inconsistency':(
            'Less than Expected' if plants<exp/2 else
            'More than Expected' if plants>exp*1.25 else '')
    })

# ---------------------------
# NONCOMPLIANCE CHECKS
# ---------------------------
def labour_Registered(r):
    return any([r.get(k,'').strip().lower() in ['any_time_when_needed','yes','no']
                for k in ['childrenworkingconditions','prisoners','contractsworkers','drinkingwaterworkers']])
def check_labour_mismatch(r):
    return 'Labour-Noncompliance-Mismatch' if labour_Registered(r) and float(r.get('noncompliancesfound_Labour',0))==0 else None

def check_environmental_mismatch(r):
    cond= any([r.get(k,'').strip().lower()=='yes' for k in ['cutnativetrees','cutforests','toiletdischarge']]) or r.get('separatewaste','').strip().lower()=='no'
    return 'Environmental-Noncompliance-Mismatch' if cond and float(r.get('noncompliancesfound_Environmental',0))==0 else None

def agrochemical_Registered(r):
    for k in ['methodspestdiseasemanagement_using_chemicals','fertilizerchemicals_Pesticides','fertilizerchemicals_Fungicides',
              'fertilizerchemicals_Herbicides','childrenlabouractivities_spraying_of_chemicals',
              'typeworkvulnerable_Spraying_of_chemicals','agriculturalinputs_synthetic_chemicals_or_fertilize']:
        try:
            if float(r.get(k,0))==1: return True
        except: pass
    return False

def check_agrochemical_mismatch(r):
    return 'Agrochemical-Noncompliance-Mismatch' if agrochemical_Registered(r) and float(r.get('noncompliancesfound_Agro_chemical',0))==0 else None

def agronomic_Registered(r):
    return any([r.get(k,'').strip().lower()=='no' for k in ['pruning','desuckering','manageweeds','knowledgeIPM']])
def check_agronomic_mismatch(r):
    return 'Agronomic-Noncompliance-Mismatch' if agronomic_Registered(r) and float(r.get('noncompliancesfound_Agronomic',0))==0 else None

def postharvest_Registered(r):
    return any([r.get(k,'').strip().lower()=='no' for k in ['ripepods','storedrycocoa','separatebasins']])
def check_postharvest_mismatch(r):
    return 'PostHarvest-Noncompliance-Mismatch' if postharvest_Registered(r) and float(r.get('noncompliancesfound_Harvest_and_postharvestt',0))==0 else None

def check_phone_mismatch(r):
    return 'Phone number mismatch' if r.get('phone_match','').strip().lower()!='match' else None

def check_time_inconsistency(r):
    return 'Time inconsistency' if float(r.get('duration',0))<900 else None

# ---------------------------
# AGGREGATE ADVICE & RISK
# ---------------------------
def get_inconsistency_flags(r):
    flags={}
    flags['Labour_Noncompliance_Advice']=check_labour_mismatch(r) or 'None of the above'
    flags['Environmental_Noncompliance_Advice']=check_environmental_mismatch(r) or 'None of the above'
    flags['Agrochemical_Noncompliance_Advice']=check_agrochemical_mismatch(r) or 'None of the above'
    flags['Agronomic_Noncompliance_Advice']=check_agronomic_mismatch(r) or 'None of the above'
    flags['PostHarvest_Noncompliance_Advice']=check_postharvest_mismatch(r) or 'None of the above'
    flags['Phone_Mismatch_Advice']=check_phone_mismatch(r) or 'None of the above'
    flags['Time_Inconsistency_Advice']=check_time_inconsistency(r) or 'None of the above'
    flags.update(compute_productive_plants_metrics(r).to_dict())
    _,pct=check_overlaps(gdf,r['Farmercode'])
    flags['Overlap_Inconsistency_Advice']=f'Overlap {pct:.2f}%' if pct>=5 else 'None of the above'
    return flags

# ---------------------------
# RATING AND INSPECTOR CHART (unchanged)
# ---------------------------
def compute_rating(r):
    score=0
    if r.get('phone_match','').strip().lower()=='match': score+=1
    if float(r.get('duration',0))>900: score+=1
    try:
        if float(r.get('kgsold',0))<=float(r.get('harvestflyseason',0))+float(r.get('totalharvest',0)): score+=1
    except: pass
    if float(r.get('acreagetotalplot',0))<float(r.get('cocoaacreage',0)): score+=1
    try:
        if float(r.get('Productiveplants',0))>= sum([float(r.get(k,0)) for k in ['youngplants','stumpedplants','shadeplants']]): score+=1
    except: pass
    # noncompliance reward
    if not(labour_Registered(r) and float(r.get('noncompliancesfound_Labour',0))==0): score+=1
    if not(any([r.get(k,'').strip().lower()=='yes' for k in ['cutnativetrees','cutforests','toiletdischarge']])
           and float(r.get('noncompliancesfound_Environmental',0))==0): score+=1
    if not(agrochemical_Registered(r) and float(r.get('noncompliancesfound_Agro_chemical',0))==0): score+=1
    if not(agronomic_Registered(r) and float(r.get('noncompliancesfound_Agronomic',0))==0): score+=1
    if not(postharvest_Registered(r) and float(r.get('noncompliancesfound_Harvest_and_postharvestt',0))==0): score+=1
    # productive plants
    try:
        if float(r.get('Productiveplants',0))<=1.25*float(r.get('total_acreage_farm',0))*450: score+=1
    except: pass
    _,pct=check_overlaps(gdf,r['Farmercode'])
    if pct<5: score+=1
    return score

df['total_rating']=df.apply(compute_rating,axis=1)
inspector_rating=df.groupby('username')['total_rating'].mean().reset_index()
if 'district' in df.columns:
    dlist=['All Districts']+sorted(df['district'].dropna().unique())
    sel=st.selectbox('Select District',dlist)
    if sel!='All Districts':
        inspector_rating=df[df['district']==sel].groupby('username')['total_rating'].mean().reset_index()

st.subheader('Best Inspectors by Average Rating')
fig3,ax3=plt.subplots(figsize=(10,6))
best=inspector_rating.sort_values('total_rating',ascending=False).head(10)
ax3.bar(best['username'],best['total_rating'])
ax3.set_xticklabels(best['username'],rotation=45,ha='right')
st.pyplot(fig3)

# ---------------------------
# EXPORT FUNCTION WITH ALL FLAGS
# ---------------------------
def export_with_inconsistencies_merged():
    export_df=df.copy()
    # existing flags
    inc=export_df.apply(lambda r: pd.Series(get_inconsistency_flags(r)),axis=1)
    export_df=pd.concat([export_df,inc],axis=1)
    export_df['Acres']=gpd.GeoSeries(export_df['geometry'],crs=gdf.crs).area*0.000247105
    export_df['geometry']=export_df['geometry'].apply(lambda g: g.wkt)
    # merge agg advice
    # (build agg_incons similarly as before if needed)
    # recalc ratings
    export_df['total_rating']=export_df.apply(compute_rating,axis=1)
    avg=export_df.groupby('username')['total_rating'].mean().reset_index().rename(columns={'total_rating':'average_rating_per_username'})
    export_df=export_df.merge(avg,on='username',how='left')

    # COMPUTE NEW CLEANING FLAGS
    def compute_cleaning_flags(r):
        return pd.Series({
            'number_fields_flag': r.get('number_fields',0)>5,
            'acreage_farm_vs_cocoa_flag': (
                r.get('total_acreage_farm',0)<r.get('total_acreage_cocoa',0)
                or r.get('total_acreage_farm',0)>20
                or r.get('total_acreage_cocoa',0)>20),
            'ID_format_flag': (
                str(r.get('IDtype','')).strip().lower()=='national_id'
                and not str(r.get('IDnumber','')).startswith(('CM','CF'))),
            'children_vs_schoolaged_flag': r.get('children',0)<r.get('schoolaged',0),
            'attendingschool_vs_schoolaged_flag': r.get('attendingschool',0)>r.get('schoolaged',0),
            'kgsold_vs_totalharvest_flag': r.get('kgsold',0)>r.get('totalharvest',0),
            'sale_to_latitude_flag': (
                r.get('salesmadeto_Latitude',0)==0 and r.get('kgsold',0)>0),
            'plot_acreage_vs_farm_flag': r.get('acreage_totalplot',0)>r.get('total_acreage_farm',0),
            'cocoa_acreage_vs_plot_flag': r.get('cocoa_acreage',0)>r.get('acreage_totalplot',0),
            'productiveplants_vs_expected_flag': r.get('Productiveplants',0)>
                1.25*r.get('total_acreage_farm',0)*450,
            'plot_vs_farm_diff_flag': (
                r.get('acreage_totalplot',0)-r.get('total_acreage_farm',0))>=2,
            'noncompliance_without_advice_flag': (
                (r.get('noncompliancesfound_Labour',0)>0 and r['Labour_Noncompliance_Advice']=='None of the above') or
                (r.get('noncompliancesfound_Environmental',0)>0 and r['Environmental_Noncompliance_Advice']=='None of the above') or
                (r.get('noncompliancesfound_Agro_chemical',0)>0 and r['Agrochemical_Noncompliance_Advice']=='None of the above') or
                (r.get('noncompliancesfound_Agronomic',0)>0 and r['Agronomic_Noncompliance_Advice']=='None of the above') or
                (r.get('noncompliancesfound_Harvest_and_postharvestt',0)>0 and r['PostHarvest_Noncompliance_Advice']=='None of the above')
            ),
            'agrochemical_violations_flag': r.get('noncompliancesfound_Agro_chemical',0)>0
        })
    clean_flags=export_df.apply(compute_cleaning_flags,axis=1)
    export_df=pd.concat([export_df,clean_flags],axis=1)

    # OUTPUT
    towrite=io.BytesIO()
    with pd.ExcelWriter(towrite,engine='xlsxwriter') as writer:
        export_df.to_excel(writer,index=False,sheet_name='Cleaned_Inspections')
    towrite.seek(0)
    st.download_button(
        label='Download Cleaned & Flagged Inspections',
        data=towrite,
        file_name='inspection_data_cleaned_flags.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

if st.button('Export Updated Form with Cleaning Flags'):
    export_with_inconsistencies_merged()
