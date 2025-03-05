import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
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
    df.loc[cond1, 'polygonplot'] = df.loc[cond1, 'redo_polygonplot']
    for new_col, plot_val in [('polygonplotnew_1', 'Plot2'),
                              ('polygonplotnew_2', 'Plot3'),
                              ('polygonplotnew_3', 'Plot4'),
                              ('polygonplotnew_4', 'Plot5')]:
        if new_col in df.columns:
            cond = df[new_col].notna() & (df['redo_selectplot'] == plot_val)
            df.loc[cond, new_col] = df.loc[cond, 'redo_polygonplot']
    df = df.drop(columns=['redo_selectplot', 'redo_polygonplot'])

# ---------------------------
# DATE SLIDER FILTERING
# ---------------------------
# Use "Submissiondate" if available; otherwise "SubmissionDate"
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
    # Only take X and Y (ignoring Z)
    vertices = [tuple(map(float, point.strip().split()[:2]))
                for point in polygon_str.split(';') if point.strip() and len(point.split()) >= 3]
    return Polygon(vertices) if len(vertices) >= 3 else None

def combine_polygons(row):
    # Combine available polygon columns for mapping purposes
    polys = [parse_polygon_z(row[col]) for col in ['polygonplot', 'polygonplotnew_1',
                                                    'polygonplotnew_2', 'polygonplotnew_3',
                                                    'polygonplotnew_4'] if col in row and pd.notna(row[col])]
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

# Create a GeoDataFrame and project to Uganda's CRS (EPSG:2109)
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
# HELPER FUNCTIONS FOR INCONSISTENCY CHECKS
# ---------------------------
# (a) Noncompliance mismatch checks
def check_labour_mismatch(row):
    cond = (
        (str(row.get('childrenworkingconditions', '')).strip().lower() == "any_time_when_needed") or
        (str(row.get('prisoners', '')).strip().lower() == "yes") or
        (str(row.get('contractsworkers', '')).strip().lower() == "no") or
        (str(row.get('drinkingwaterworkers', '')).strip().lower() == "no")
    )
    try:
        found = float(row.get("noncompliancesfound_Labour", 0))
    except:
        found = 0
    if cond and found == 0:
        return "Labour-Noncompliance-Mismatch"
    return None

def check_environmental_mismatch(row):
    cond = (
        (str(row.get('cutnativetrees', '')).strip().lower() == "yes") or
        (str(row.get('cutforests', '')).strip().lower() == "yes") or
        (str(row.get('toiletdischarge', '')).strip().lower() == "yes") or
        (str(row.get('separatewaste', '')).strip().lower() == "no")
    )
    try:
        found = float(row.get("noncompliancesfound_Environmental", 0))
    except:
        found = 0
    if cond and found == 0:
        return "Environmental-Noncompliance-Mismatch"
    return None

def check_agronomic_mismatch(row):
    cond = (
        (str(row.get('pruning', '')).strip().lower() == "no") or
        (str(row.get('desuckering', '')).strip().lower() == "no") or
        (str(row.get('manageweeds', '')).strip().lower() == "no") or
        (str(row.get('knowledgeIPM', '')).strip().lower() == "no")
    )
    try:
        found = float(row.get("noncompliancesfound_Agronomic", 0))
    except:
        found = 0
    if cond and found == 0:
        return "Agronomic-Noncompliance-Mismatch"
    return None

def check_postharvest_mismatch(row):
    cond = (
        (str(row.get('ripepods', '')).strip().lower() == "no") or
        (str(row.get('storedrycocoa', '')).strip().lower() == "no") or
        (str(row.get('separatebasins', '')).strip().lower() == "no")
    )
    try:
        found = float(row.get("noncompliancesfound_Harvest_and_postharvestt", 0))
    except:
        found = 0
    if cond and found == 0:
        return "PostHarvest-Noncompliance-Mismatch"
    return None

# (b) Phone mismatch check – using the 'phone_match' column
def check_phone_mismatch(row):
    pm = str(row.get('phone_match', "")).strip().lower()
    if pm != "match":
        return "Phone number mismatch"
    return None

# (c) Productive plants expected check
def compute_total_area(row):
    # Sum areas from these polygon columns.
    # Ensure that the polygon is reprojected to Uganda's CRS (EPSG:2109) before area calculation.
    polygon_cols = ['polygonplot', 'polygonplotnew_2', 'polygonplotnew_3', 'polygonplotnew_4']
    total_area = 0
    for col in polygon_cols:
        val = row.get(col)
        poly = None
        if isinstance(val, str):
            poly = parse_polygon_z(val)
        elif isinstance(val, Polygon):
            poly = val
        if poly and poly.is_valid:
            try:
                # Create a GeoSeries, assign original CRS EPSG:4326 then reproject to EPSG:2109
                gseries = gpd.GeoSeries([poly], crs="EPSG:4326").to_crs("EPSG:2109")
                total_area += gseries.iloc[0].area
            except Exception as e:
                pass
    return total_area

def compute_total_plants(row):
    # Sum productive plants from these columns
    plant_cols = ['Productiveplants', 'Productiveplantsnew_1', 'Productiveplantsnew_2', 'Productiveplantsnew_3', 'Productiveplantsnew_4']
    total = 0
    for col in plant_cols:
        try:
            total += float(row.get(col, 0))
        except:
            continue
    return total

def check_productive_plants(row):
    total_area = compute_total_area(row)  # in m², computed in Uganda's CRS (EPSG:2109)
    acres = total_area * 0.000247105  # conversion factor from m² to acres
    expected = acres * 450  # expected number of plants per acre
    total_plants = compute_total_plants(row)
    # Uncomment the line below for debugging if needed:
    # st.write(f"Farmer {row['Farmercode']}: Area: {total_area:.2f} m², Acres: {acres:.2f}, Expected: {expected:.2f}, Actual: {total_plants}")
    if expected > 0:
        if total_plants > expected * 1.25 or total_plants < expected * 0.5:
            return "Total productive plants expected inconsistency"
    return None

# (d) Time inconsistency check (High risk: duration < 15 mins)
def check_time_inconsistency(row):
    try:
        duration = float(row.get('duration', 0))
    except:
        duration = None
    if duration is not None and duration < 900:
        return "Time inconsistency: Inspection completed in less than 15 mins"
    return None

# ---------------------------
# INCONSISTENCY FLAGS PER ROW
# ---------------------------
def get_inconsistency_flags(row):
    flags = {}
    flags['Labour_Noncompliance'] = True if check_labour_mismatch(row) else False
    flags['Environmental_Noncompliance'] = True if check_environmental_mismatch(row) else False
    flags['Agronomic_Noncompliance'] = True if check_agronomic_mismatch(row) else False
    flags['PostHarvest_Noncompliance'] = True if check_postharvest_mismatch(row) else False
    flags['Phone_Mismatch'] = True if check_phone_mismatch(row) else False
    flags['Productive_Plants_Inconsistency'] = True if check_productive_plants(row) else False
    flags['Time_Inconsistency'] = True if check_time_inconsistency(row) else False
    overlaps, overall_pct = check_overlaps(gdf, row['Farmercode'])
    flags['Overlap_Inconsistency'] = True if overall_pct >= 5 else False
    return flags

# ---------------------------
# INCONSISTENCY DETECTION (Row-wise)
# ---------------------------
inconsistencies_list = []
for idx, row in df.iterrows():
    farmer = row['Farmercode']
    user = row.get('username', '')
    for check_fn in [check_labour_mismatch, check_environmental_mismatch,
                     check_agronomic_mismatch, check_postharvest_mismatch,
                     check_phone_mismatch, check_productive_plants, check_time_inconsistency]:
        msg = check_fn(row)
        if msg:
            inconsistencies_list.append({'Farmercode': farmer, 'username': user, 'inconsistency': msg})
# Overlap check (aggregated per farmer)
overlap_incons_list = []
for code in df['Farmercode'].unique():
    overlaps, overall_pct = check_overlaps(gdf, code)
    if overall_pct > 10:
        text = "Overlap > 10%"
    elif overall_pct >= 5:
        text = "Overlap 5-10%"
    else:
        text = None
    if text:
        target_username = df.loc[df['Farmercode'] == code, 'username'].iloc[0]
        overlap_incons_list.append({'Farmercode': code, 'username': target_username, 'inconsistency': text})

inconsistencies_df = pd.DataFrame(inconsistencies_list + overlap_incons_list)

# ---------------------------
# AGGREGATE INCONSISTENCIES PER RECORD
# ---------------------------
risk_order = {"High": 3, "Medium": 2, "Low": 1, "None": 0}
def get_risk_rating(inc_text):
    inc_text_lower = inc_text.lower()
    if ("noncompliance-mismatch" in inc_text_lower or 
        "total productive plants" in inc_text_lower or 
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
# GRAPH: Individual Inconsistency Occurrences
# ---------------------------
indiv_incons_counts = inconsistencies_df.groupby('inconsistency').size().reset_index(name='Count')
st.subheader("Individual Inconsistency Occurrences")
if not indiv_incons_counts.empty:
    fig2, ax2 = plt.subplots(figsize=(10,6))
    ax2.bar(indiv_incons_counts['inconsistency'], indiv_incons_counts['Count'], color='orange')
    ax2.set_xlabel("Inconsistency Type")
    ax2.set_ylabel("Number of Occurrences")
    ax2.set_title("Count of Individual Inconsistency Occurrences")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig2)
else:
    st.write("No inconsistency data available.")

# ---------------------------
# TOP 10 INSPECTORS BAR CHART
# ---------------------------
total_per_inspector = df.groupby('username')['Farmercode'].nunique().reset_index(name='TotalCodes')
high_risk_inspector = agg_incons[agg_incons['Risk Rating'] == "High"]
high_per_inspector = high_risk_inspector.groupby('username')['Farmercode'].nunique().reset_index(name='HighRiskCodes')
inspector_counts = total_per_inspector.merge(high_per_inspector, on='username', how='left').fillna(0)
inspector_counts['HighRiskCodes'] = inspector_counts['HighRiskCodes'].astype(int)
top10 = inspector_counts.sort_values(by='TotalCodes', ascending=False).head(10)

st.subheader("Suspicious Inspectors: High Risk Inspections vs Total Inspections")
if not top10.empty:
    fig, ax = plt.subplots(figsize=(10,6))
    x = range(len(top10))
    width = 0.35
    ax.bar(x, top10['TotalCodes'], width, label='Total Unique Codes', color='lightblue')
    ax.bar([p + width for p in x], top10['HighRiskCodes'], width, label='High Risk Codes', color='red')
    ax.set_xticks([p + width/2 for p in x])
    ax.set_xticklabels(top10['username'], rotation=45, ha='right')
    ax.set_ylabel("Number of Inspections")
    ax.set_title("Unique High Risk vs Total Codes per Inspector")
    ax.legend()
    st.pyplot(fig)
else:
    st.write("No inspector data available for the bar chart.")

# ---------------------------
# UI: Show Aggregated Inconsistencies for a Selected Code
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
# EXPORT (Merged with Aggregated Risk and Boolean Inconsistency Flags)
# ---------------------------
def export_with_inconsistencies_merged(main_gdf, agg_incons_df):
    # Create a copy for export
    export_df = df.copy()
    # Compute individual inconsistency Boolean flags for each row
    flags = export_df.apply(lambda row: pd.Series(get_inconsistency_flags(row)), axis=1)
    export_df = pd.concat([export_df, flags], axis=1)
    # Compute area in acres using proper GeoSeries conversion (using Uganda's CRS EPSG:2109)
    export_df['Acres'] = gpd.GeoSeries(export_df['geometry'], crs=gdf.crs).area * 0.000247105
    # Convert geometry to WKT for export
    export_df['geometry'] = export_df['geometry'].apply(lambda geom: geom.wkt)
    # Merge with aggregated risk and overall inconsistency messages
    merged_df = export_df.merge(
        agg_incons_df[['Farmercode', 'username', 'inconsistency', 'Risk Rating', 'Trust Responses']],
        on=['Farmercode', 'username'], how='left'
    )
    merged_df['inconsistency'] = merged_df['inconsistency'].fillna("No Inconsistency")
    merged_df['Risk Rating'] = merged_df['Risk Rating'].fillna("None")
    merged_df['Trust Responses'] = merged_df['Trust Responses'].fillna("Yes")
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
        merged_df.to_excel(writer, index=False, sheet_name="Updated Form")
    towrite.seek(0)
    st.download_button(
        label="Download Updated Form with Inconsistency Columns",
        data=towrite,
        file_name="updated_inspection_form_merged.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if st.button("Export Updated Form to Excel"):
    export_with_inconsistencies_merged(gdf, agg_incons)
