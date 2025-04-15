import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
import io

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

if main_file is None:
    st.info("Please upload the Main Inspection file and Redo file.")
    st.stop()

try:
    if main_file.name.endswith('.xlsx'):
        df = pd.read_excel(main_file, engine='openpyxl')
    else:
        df = pd.read_csv(main_file)
except Exception as e:
    st.error("Error loading main file: " + str(e))
    st.stop()

# Require essential columns in main file.
if 'Farmercode' not in df.columns or 'polygonplot' not in df.columns:
    st.error("The Main Inspection Form must contain 'Farmercode' and 'polygonplot' columns.")
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
        st.error("Error loading redo polygon file: " + str(e))
        st.stop()

if "farmer_code" in df_redo.columns:
    df_redo = df_redo.rename(columns={'farmer_code': 'Farmercode'})
required_redo_cols = ['Farmercode', 'selectplot', 'polygonplot']
if not all(col in df_redo.columns for col in required_redo_cols):
    st.error("Redo polygon file must contain 'Farmercode', 'selectplot', and 'polygonplot' columns.")
    st.stop()

if 'SubmissionDate' in df_redo.columns and 'endtime' in df_redo.columns:
    df_redo['SubmissionDate'] = pd.to_datetime(df_redo['SubmissionDate'], errors='coerce')
    df_redo['endtime'] = pd.to_datetime(df_redo['endtime'], errors='coerce')
    df_redo = df_redo.sort_values(by=['SubmissionDate', 'endtime'])
    df_redo = df_redo.groupby('Farmercode', as_index=False).last()
    df_redo = df_redo.rename(columns={'selectplot': 'redo_selectplot',
                                      'polygonplot': 'redo_polygonplot'})
    df = df.merge(df_redo[['Farmercode', 'redo_selectplot', 'redo_polygonplot']], on='Farmercode', how='left')
    cond1 = df['polygonplot'].notna() & (df['redo_selectplot'] == 'Plot1')
    df.loc[cond1, 'polygonplot'] = df.loc[cond1, 'redo_polygonplot'].apply(lambda x: str(x) if pd.notnull(x) else x)
    for new_col, plot_val in [('polygonplotnew_1', 'Plot2'),
                              ('polygonplotnew_2', 'Plot3'),
                              ('polygonplotnew_3', 'Plot4'),
                              ('polygonplotnew_4', 'Plot5')]:
        if new_col in df.columns:
            cond = df[new_col].notna() & (df['redo_selectplot'] == plot_val)
            df.loc[cond, new_col] = df.loc[cond, 'redo_polygonplot'].apply(lambda x: str(x) if pd.notnull(x) else x)
    df = df.drop(columns=['redo_selectplot', 'redo_polygonplot'])

# ---------------------------
# DATE SLIDER FILTERING
# ---------------------------
if 'Submissiondate' in df.columns:
    df['Submissiondate'] = pd.to_datetime(df['Submissiondate'], errors='coerce')
elif 'SubmissionDate' in df.columns:
    df['Submissiondate'] = pd.to_datetime(df['SubmissionDate'], errors='coerce')
else:
    df['Submissiondate'] = pd.NaT

if df['Submissiondate'].notna().sum() > 0:
    min_date = df['Submissiondate'].min().date()
    max_date = df['Submissiondate'].max().date()
    selected_date_range = st.slider("Select Submission Date Range", min_date, max_date, (min_date, max_date))
    df = df[(df['Submissiondate'].dt.date >= selected_date_range[0]) &
            (df['Submissiondate'].dt.date <= selected_date_range[1])]
else:
    st.warning("Submission date not available. Proceeding without date filtering.")

# ---------------------------
# POLYGON HANDLING FUNCTIONS
# ---------------------------
def parse_polygon_z(polygon_str):
    if not isinstance(polygon_str, str):
        return None
    vertices = [tuple(map(float, point.strip().split()[:2]))
                for point in polygon_str.split(';') if point.strip() and len(point.split()) >= 3]
    return Polygon(vertices) if len(vertices) >= 3 else None

def combine_polygons(row):
    # Combine available polygon columns for mapping
    cols = ['polygonplot', 'polygonplotnew_1', 'polygonplotnew_2', 'polygonplotnew_3', 'polygonplotnew_4']
    polys = [parse_polygon_z(row[col]) for col in cols if col in row and pd.notna(row[col])]
    valid_polys = []
    for p in polys:
        if p is not None:
            if not p.is_valid:
                p = p.buffer(0)
            if p.is_valid:
                valid_polys.append(p)
    if not valid_polys:
        return None
    return valid_polys[0] if len(valid_polys) == 1 else unary_union(valid_polys)

df['geometry'] = df.apply(combine_polygons, axis=1)
df = df[df['geometry'].notna()]

# Create a GeoDataFrame and re-project to Uganda's CRS (EPSG:2109)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
gdf = gdf.to_crs('EPSG:2109')
gdf['geometry'] = gdf['geometry'].buffer(0)
gdf = gdf[gdf.is_valid]

# ---------------------------
# SPATIAL INDEXED OVERLAP CHECK
# ---------------------------
def check_overlaps(gdf, target_code):
    target_row = gdf[gdf['Farmercode'] == target_code]
    if target_row.empty:
        return [], 0
    target_poly = target_row.geometry.iloc[0]
    total_area = target_poly.area
    overlaps = []
    union_overlap = None
    sindex = gdf.sindex
    candidate_idxs = list(sindex.intersection(target_poly.bounds))
    for idx in candidate_idxs:
        row = gdf.iloc[idx]
        if row['Farmercode'] == target_code:
            continue
        other_poly = row.geometry
        if not target_poly.intersects(other_poly):
            continue
        intersection = target_poly.intersection(other_poly)
        area = intersection.area if not intersection.is_empty else 0
        if area > 1e-6:
            overlaps.append({
                'Farmercode': row['Farmercode'],
                'overlap_area': area,
                'total_area': total_area,
                'intersection': intersection
            })
            union_overlap = intersection if union_overlap is None else union_overlap.union(intersection)
    overall_pct = (union_overlap.area / total_area * 100) if union_overlap else 0
    return overlaps, overall_pct

def plot_geometry(ax, geom, color, label, text_label):
    if hasattr(geom, 'exterior'):
        x, y = geom.exterior.xy
        ax.fill(x, y, alpha=0.5, fc=color, ec='darkred', label=label)
        cx, cy = geom.centroid.x, geom.centroid.y
        ax.text(cx, cy, f"{text_label:.1f}%", fontsize=10, color='white', ha='center', va='center')
    elif geom.geom_type in ['MultiPolygon', 'GeometryCollection']:
        for part in geom.geoms:
            if hasattr(part, 'exterior'):
                x, y = part.exterior.xy
                ax.fill(x, y, alpha=0.5, fc=color, ec='darkred', label=label)
                cx, cy = part.centroid.x, part.centroid.y
                ax.text(cx, cy, f"{text_label:.1f}%", fontsize=10, color='white', ha='center', va='center')

# ---------------------------
# PRODUCTIVE PLANTS CALCULATION
# ---------------------------
def compute_productive_plants_metrics(row):
    total_area = 0
    for col in row.index:
        if "acres_polygonplot" in col.lower() and pd.notnull(row[col]):
            try:
                total_area += float(row[col])
            except:
                continue
    total_plants = 0
    for col in row.index:
        if "productiveplants" in col.lower() and pd.notnull(row[col]):
            try:
                total_plants += float(row[col])
            except:
                continue
    expected = total_area * 450
    half_expected = expected / 2
    pct125_expected = expected * 1.25
    inconsistency = ""
    if total_plants < half_expected:
        inconsistency = "Less than Expected Productive Plants"
    elif total_plants > pct125_expected:
        inconsistency = "More than expected Productive Plants"
    return pd.Series({
        "Total_Area": total_area,
        "Total_Productive_Plants": total_plants,
        "Expected_Plants": expected,
        "Half_Expected_Plants": half_expected,
        "Pct125_Expected_Plants": pct125_expected,
        "Productive_Plants_Inconsistency": inconsistency
    })

# ---------------------------
# NONCOMPLIANCE CHECK FUNCTIONS
# ---------------------------
def labour_Registered(row):
    return (str(row.get('childrenworkingconditions', '')).strip().lower() == "any_time_when_needed" or
            str(row.get('prisoners', '')).strip().lower() == "yes" or
            str(row.get('contractsworkers', '')).strip().lower() == "no" or
            str(row.get('drinkingwaterworkers', '')).strip().lower() == "no")

def check_labour_mismatch(row):
    try:
        found = float(row.get("noncompliancesfound_Labour", 0))
    except:
        found = 0
    return "Labour-Noncompliance-Mismatch" if labour_Registered(row) and found == 0 else None

def check_environmental_mismatch(row):
    cond = (str(row.get('cutnativetrees', '')).strip().lower() == "yes" or
            str(row.get('cutforests', '')).strip().lower() == "yes" or
            str(row.get('toiletdischarge', '')).strip().lower() == "yes" or
            str(row.get('separatewaste', '')).strip().lower() == "no")
    try:
        found = float(row.get("noncompliancesfound_Environmental", 0))
    except:
        found = 0
    return "Environmental-Noncompliance-Mismatch" if cond and found == 0 else None

def agrochemical_Registered(row):
    try:
        cond1 = (float(row.get('methodspestdiseasemanagement_using_chemicals', 0)) == 1)
    except:
        cond1 = False
    try:
        cond2 = (float(row.get('fertilizerchemicals_Pesticides', 0)) == 1)
    except:
        cond2 = False
    try:
        cond3 = (float(row.get('fertilizerchemicals_Fungicides', 0)) == 1)
    except:
        cond3 = False
    try:
        cond4 = (float(row.get('fertilizerchemicals_Herbicides', 0)) == 1)
    except:
        cond4 = False
    try:
        cond5 = (float(row.get('childrenlabouractivities_spraying_of_chemicals', 0)) == 1)
    except:
        cond5 = False
    try:
        cond6 = (float(row.get('typeworkvulnerable_Spraying_of_chemicals', 0)) == 1)
    except:
        cond6 = False
    try:
        cond7 = (float(row.get('agriculturalinputs_synthetic_chemicals_or_fertilize', 0)) == 1)
    except:
        cond7 = False
    return cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or cond7

def check_agrochemical_mismatch(row):
    try:
        found = float(row.get("noncompliancesfound_Agro_chemical", 0))
    except:
        found = 0
    return "Agrochemical-Noncompliance-Mismatch" if agrochemical_Registered(row) and found == 0 else None

def agronomic_Registered(row):
    return (str(row.get('pruning', '')).strip().lower() == "no" or
            str(row.get('desuckering', '')).strip().lower() == "no" or
            str(row.get('manageweeds', '')).strip().lower() == "no" or
            str(row.get('knowledgeIPM', '')).strip().lower() == "no")

def check_agronomic_mismatch(row):
    try:
        found = float(row.get("noncompliancesfound_Agronomic", 0))
    except:
        found = 0
    return "Agronomic-Noncompliance-Mismatch" if agronomic_Registered(row) and found == 0 else None

def postharvest_Registered(row):
    return (str(row.get('ripepods', '')).strip().lower() == "no" or
            str(row.get('storedrycocoa', '')).strip().lower() == "no" or
            str(row.get('separatebasins', '')).strip().lower() == "no")

def check_postharvest_mismatch(row):
    try:
        found = float(row.get("noncompliancesfound_Harvest_and_postharvestt", 0))
    except:
        found = 0
    return "PostHarvest-Noncompliance-Mismatch" if postharvest_Registered(row) and found == 0 else None

def check_phone_mismatch(row):
    return "Phone number mismatch" if str(row.get('phone_match', "")).strip().lower() != "match" else None

def check_time_inconsistency(row):
    try:
        duration = float(row.get('duration', 0))
    except:
        duration = None
    return "Time inconsistency: Inspection completed in less than 15 mins" if duration is not None and duration < 900 else None

# ---------------------------
# AGGREGATED FLAGS FOR NONCOMPLIANCE
# ---------------------------
def get_inconsistency_flags(row):
    flags = {}
    flags['Labour_Noncompliance_Advice'] = check_labour_mismatch(row) or "None of the above"
    flags['Environmental_Noncompliance_Advice'] = check_environmental_mismatch(row) or "None of the above"
    flags['Agrochemical_Noncompliance_Advice'] = check_agrochemical_mismatch(row) or "None of the above"
    flags['Agronomic_Noncompliance_Advice'] = check_agronomic_mismatch(row) or "None of the above"
    flags['PostHarvest_Noncompliance_Advice'] = check_postharvest_mismatch(row) or "None of the above"
    flags['Phone_Mismatch_Advice'] = check_phone_mismatch(row) or "None of the above"
    flags['Time_Inconsistency_Advice'] = check_time_inconsistency(row) or "None of the above"
    # Productive plants metrics:
    prod_metrics = compute_productive_plants_metrics(row)
    flags = {**flags, **prod_metrics.to_dict()}
    # Overlap check:
    overlaps, overall_pct = check_overlaps(gdf, row['Farmercode'])
    flags['Overlap_Inconsistency_Advice'] = (f"Overlap {overall_pct:.2f}%" if overall_pct >= 5 else "None of the above")
    return flags

# ---------------------------
# DETECT INDIVIDUAL INCONSISTENCIES
# ---------------------------
inconsistencies_list = []
for idx, row in df.iterrows():
    farmer = row['Farmercode']
    user = row.get('username', '')
    for check_fn in [check_labour_mismatch, check_environmental_mismatch,
                     check_agronomic_mismatch, check_postharvest_mismatch,
                     check_agrochemical_mismatch, check_phone_mismatch,
                     check_time_inconsistency,
                     lambda r: compute_productive_plants_metrics(r)["Productive_Plants_Inconsistency"]]:
        msg = check_fn(row)
        if msg:
            inconsistencies_list.append({'Farmercode': farmer, 'username': user, 'inconsistency': msg})
overlap_incons_list = []
for code in df['Farmercode'].unique():
    overlaps, overall_pct = check_overlaps(gdf, code)
    text = None
    if overall_pct > 10:
        text = "Overlap > 10%"
    elif overall_pct >= 5:
        text = "Overlap 5-10%"
    if text:
        target_username = df.loc[df['Farmercode'] == code, 'username'].iloc[0]
        overlap_incons_list.append({'Farmercode': code, 'username': target_username, 'inconsistency': text})
inconsistencies_df = pd.DataFrame(inconsistencies_list + overlap_incons_list)

# Risk rating (for aggregated message)
risk_order = {"High": 3, "Medium": 2, "Low": 1, "None": 0}
def get_risk_rating(inc_text):
    inc_text_lower = inc_text.lower()
    if ("noncompliance-mismatch" in inc_text_lower or 
        "less than expected" in inc_text_lower or 
        "more than expected" in inc_text_lower or 
        "time inconsistency" in inc_text_lower):
        return "High"
    if "overlap >" in inc_text_lower:
        return "Medium"
    if "overlap 5-10%" in inc_text_lower:
        return "Low"
    if "phone" in inc_text_lower:
        return "Medium"
    return "Low"
if not inconsistencies_df.empty:
    inconsistencies_df['Risk Rating'] = inconsistencies_df['inconsistency'].apply(get_risk_rating)
    agg_incons = inconsistencies_df.groupby(['Farmercode','username'], as_index=False).agg({
        'inconsistency': lambda x: ", ".join(x.unique()),
        'Risk Rating': lambda x: max(x, key=lambda r: risk_order.get(r, 0))
    })
    agg_incons['Trust Responses'] = agg_incons['Risk Rating'].apply(lambda x: "No" if x=="High" else "Yes")
else:
    agg_incons = pd.DataFrame(columns=['Farmercode','username','inconsistency','Risk Rating','Trust Responses'])

# ---------------------------
# BEST INSPECTORS CHART (by average rating)
# ---------------------------
# First, calculate total_rating for each record (using the updated compute_rating)
def compute_rating(row):
    score = 0
    # (1) Basic Conditions (removed contract & receipts conditions)
    if str(row.get("phone_match", "")).strip().lower() == "match":
        score += 1
    if row.get("duration", 0) > 900:
        score += 1
    kgsold = row.get("kgsold")
    harvestflyseason = row.get("harvestflyseason")
    totalharvest = row.get("totalharvest")
    if pd.notnull(kgsold) and pd.notnull(harvestflyseason) and pd.notnull(totalharvest):
        if kgsold <= (harvestflyseason + totalharvest):
            score += 1
    acreagetotalplot = row.get("acreagetotalplot")
    cocoaacreage = row.get("cocoaacreage")
    if pd.notnull(acreagetotalplot) and pd.notnull(cocoaacreage):
        if acreagetotalplot < cocoaacreage:
            score += 1
    productiveplants = row.get("Productiveplants")
    youngplants = row.get("youngplants", 0)
    stumpedplants = row.get("stumpedplants", 0)
    shadeplants = row.get("shadeplants", 0)
    if pd.notnull(productiveplants):
        if productiveplants >= (youngplants + stumpedplants + shadeplants):
            score += 1
    # (2) Labour Noncompliance
    labour_flag = (str(row.get("childrenworkingconditions", "")).strip().lower() == "any_time_when_needed" or
                   str(row.get("prisoners", "")).strip().lower() == "yes" or
                   str(row.get("contractsworkers", "")).strip().lower() == "no" or
                   str(row.get("drinkingwaterworkers", "")).strip().lower() == "no")
    if not (labour_flag and (row.get("noncompliancesfound_Labour", 0) == 0)):
        score += 1
    # (3) Environmental Noncompliance
    env_flag = (str(row.get("cutnativetrees", "")).strip().lower() == "yes" or
                str(row.get("cutforests", "")).strip().lower() == "yes" or
                str(row.get("toiletdischarge", "")).strip().lower() == "yes" or
                str(row.get("separatewaste", "")).strip().lower() == "no")
    if not (env_flag and (row.get("noncompliancesfound_Environmental", 0) == 0)):
        score += 1
    # (4) Agrochemical Noncompliance
    agro_columns = ["methodspestdiseasemanagement_using_chemicals",
                    "fertilizerchemicals_Pesticides",
                    "fertilizerchemicals_Fungicides",
                    "fertilizerchemicals_Herbicides",
                    "childrenlabouractivities_spraying_of_chemicals",
                    "typeworkvulnerable_Spraying_of_chemicals",
                    "agriculturalinputs_synthetic_chemicals_or_fertilize"]
    agro_flag = False
    for col in agro_columns:
        try:
            if float(row.get(col, 0)) == 1:
                agro_flag = True
                break
        except (ValueError, TypeError):
            pass
    if not (agro_flag and (row.get("noncompliancesfound_Agro_chemical", 0) == 0)):
        score += 1
    # (5) Agronomic Noncompliance
    agronomic_columns = ["pruning", "desuckering", "manageweeds", "knowledgeIPM"]
    if not (any(str(row.get(col, "")).strip().lower() == "no" for col in agronomic_columns) and 
            (row.get("noncompliancesfound_Agronomic", 0) == 0)):
        score += 1
    # (6) Postharvest Noncompliance
    postharvest_columns = ["ripepods", "storedrycocoa", "separatebasins"]
    if not (any(str(row.get(col, "")).strip().lower() == "no" for col in postharvest_columns) and
            (row.get("noncompliancesfound_Harvest_and_postharvestt", 0) == 0)):
        score += 1
    # (7) Productive Plants Inconsistency
    total_area = 0
    total_productive_plants = 0
    for col in row.index:
        if "acres_polygonplot" in col.lower():
            try:
                total_area += float(row[col])
            except (ValueError, TypeError):
                pass
        if "productiveplants" in col.lower():
            try:
                total_productive_plants += float(row[col])
            except (ValueError, TypeError):
                pass
    expected_plants = total_area * 450
    if total_productive_plants <= 1.25 * expected_plants:
        score += 1
    # (8) Polygon Overlap Condition (Extra point if overlap is less than 5%)
    farmer = row.get("Farmercode")
    if pd.notnull(farmer):
        _, overall_pct = check_overlaps(gdf, farmer)
        if overall_pct < 5:
            score += 1
    return score

# Compute total_rating and merge average rating per inspector
df["total_rating"] = df.apply(compute_rating, axis=1)
inspector_rating = df.groupby('username')["total_rating"].mean().reset_index()

# If district column exists, allow filtering via selectbox
if "district" in df.columns:
    districts = sorted(df["district"].dropna().unique().tolist())
    selected_district = st.selectbox("Select District", ["All Districts"] + districts)
    if selected_district != "All Districts":
        district_filter = df["district"] == selected_district
        inspector_rating = df[district_filter].groupby('username')["total_rating"].mean().reset_index()

st.subheader("Best Inspectors by Average Rating")
if not inspector_rating.empty:
    best_inspectors = inspector_rating.sort_values(by="total_rating", ascending=False).head(10)
    fig3, ax3 = plt.subplots(figsize=(10,6))
    ax3.bar(best_inspectors["username"], best_inspectors["total_rating"], color='green')
    ax3.set_xlabel("Inspector (username)")
    ax3.set_ylabel("Average Total Rating")
    ax3.set_title("Top Inspectors by Average Rating")
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig3)
else:
    st.write("No rating data available.")

# ---------------------------
# UI: Show Aggregated Inconsistencies for a Selected Farmer
# ---------------------------
farmer_list = gdf['Farmercode'].dropna().unique().tolist()
selected_code = st.selectbox("Select Farmer Code", farmer_list)
selected_incons = agg_incons[agg_incons['Farmercode'] == selected_code]
st.subheader(f"Inconsistencies for Farmer {selected_code}")
if not selected_incons.empty:
    st.dataframe(selected_incons)
else:
    st.write("No inconsistencies found for this code.")

target_row = gdf[gdf['Farmercode'] == selected_code]
if not target_row.empty:
    area = target_row.geometry.iloc[0].area
    st.subheader("Target Polygon Area (Union):")
    st.write(f"{area:.2f} m²")
    overlaps, overall_pct = check_overlaps(gdf, selected_code)
    st.subheader(f"Overlap Results for Farmer {selected_code}")
    if overlaps:
        for res in overlaps:
            pct = res['overlap_area'] / res['total_area'] * 100
            st.write(f"Farmer {res['Farmercode']} overlap: {res['overlap_area']:.2f} m² ({pct:.2f}%)")
        st.write(f"Total Overlap Percentage: {overall_pct:.2f}%")
    else:
        st.success("No overlaps found for this farmer.")
    
    if overlaps:
        target_poly = target_row.geometry.iloc[0]
        fig, ax = plt.subplots(figsize=(8,8))
        if hasattr(target_poly, 'exterior'):
            x, y = target_poly.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='blue', ec='black', label=f"Target: {selected_code}")
        else:
            for part in target_poly.geoms:
                x, y = part.exterior.xy
                ax.fill(x, y, alpha=0.5, fc='blue', ec='black', label=f"Target: {selected_code}")
        for res in overlaps:
            overlap_pct = res['overlap_area'] / res['total_area'] * 100
            plot_geometry(ax, res['intersection'], 'red', f"Overlap {res['Farmercode']}", overlap_pct)
        ax.set_title(f"Overlap Map for Farmer {selected_code}")
        ax.set_xlabel("Easting")
        ax.set_ylabel("Northing")
        ax.legend(loc='upper right', fontsize='small')
        st.pyplot(fig)

# ---------------------------
# EXPORT FUNCTION
# ---------------------------
def export_with_inconsistencies_merged(main_gdf, agg_incons_df):
    export_df = df.copy()
    flags = export_df.apply(lambda row: pd.Series(get_inconsistency_flags(row)), axis=1)
    export_df = pd.concat([export_df, flags], axis=1)
    export_df['Acres'] = gpd.GeoSeries(export_df['geometry'], crs=gdf.crs).area * 0.000247105
    export_df['geometry'] = export_df['geometry'].apply(lambda geom: geom.wkt)
    merged_df = export_df.merge(
        agg_incons_df[['Farmercode', 'username', 'inconsistency', 'Risk Rating', 'Trust Responses']],
        on=['Farmercode', 'username'], how='left'
    )
    merged_df['inconsistency'] = merged_df['inconsistency'].fillna("No Inconsistency")
    merged_df['Risk Rating'] = merged_df['Risk Rating'].fillna("None")
    merged_df['Trust Responses'] = merged_df['Trust Responses'].fillna("Yes")
    # Recalculate total_rating and merge average rating per inspector
    merged_df["total_rating"] = merged_df.apply(compute_rating, axis=1)
    avg_rating = merged_df.groupby('username')["total_rating"].mean().reset_index().rename(columns={"total_rating": "average_rating_per_username"})
    merged_df = merged_df.merge(avg_rating, on='username', how='left')
    
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
        merged_df.to_excel(writer, index=False, sheet_name="Updated Form")
    towrite.seek(0)
    st.download_button(
        label="Download Updated Form with Rating Columns",
        data=towrite,
        file_name="updated_inspection_form_merged.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if st.button("Export Updated Form to Excel"):
    export_with_inconsistencies_merged(gdf, agg_incons)
