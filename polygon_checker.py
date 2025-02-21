import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import io

st.title("Latitude Polygon Overlap & Inconsistency Checker (Optimized & Extended)")

# ---------------------------
# FILE UPLOAD
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Main Inspection File")
    main_file = st.file_uploader("Upload Main Inspection Form (CSV or Excel)",
                                 type=["xlsx", "csv"], key="main_upload")
with col2:
    st.subheader("Redo Polygon File")
    redo_file = st.file_uploader("Upload Redo Polygon Form (CSV or Excel)",
                                 type=["xlsx", "csv"], key="redo_upload")

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
    df = df.merge(df_redo[['Farmercode', 'redo_selectplot', 'redo_polygonplot']],
                  on='Farmercode', how='left')
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
    return valid_polys[0] if len(valid_polys)==1 else unary_union(valid_polys)

df['geometry'] = df.apply(combine_polygons, axis=1)
df = df[df['geometry'].notna()]

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

# ---------------------------
# PLOTTING HELPER
# ---------------------------
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
# UPDATED RISK RATING FUNCTION
# ---------------------------
def get_risk_rating(inc_text):
    txt = inc_text.lower()
    # High risk conditions: time issues, overlap >10%, productive plants issues, more than 12 codes in a day
    if ("time < 15min" in txt or 
        "overlap > 10%" in txt or 
        "more than 12 codes" in txt or 
        "productiveplants" in txt):
        return "High"
    # Medium risk: overlap between 5-10% or GPS distance issues
    if ("overlap 5-10%" in txt or 
        "gps is more than 100m" in txt):
        return "Medium"
    if any(kw in txt for kw in ["phone mismatch", "duplicate phone", "duplicate farmer"]):
        return "Medium"
    return "Low"

# ---------------------------
# INCONSISTENCY DETECTION (Vectorized)
# ---------------------------
# 1. Time Inconsistency
df_time_incons = df.loc[(df['duration'] < 900) & (df['Registered'].str.lower()=='yes'), ['Farmercode','username']]
df_time_incons = df_time_incons.assign(inconsistency="Time < 15min but Registered == Yes")

# 2. Phone Mismatch
df['Phone'] = pd.to_numeric(df['Phone'], errors='coerce').fillna(0).astype(int).astype(str)
df['Phone_hidden'] = pd.to_numeric(df['Phone_hidden'], errors='coerce').fillna(0).astype(int).astype(str)
df_phone_incons = df.loc[df['Phone'] != df['Phone_hidden'], ['Farmercode','username']]
df_phone_incons = df_phone_incons.assign(inconsistency="Phone mismatch (Phone != Phone_hidden)")

# 3. Duplicate Phone Entries & Duplicate Farmer Codes
df_dup_phones = df[df.duplicated(subset=['Phone'], keep=False)][['Farmercode','username']]
df_dup_phones = df_dup_phones.assign(inconsistency="Duplicate phone entry")
df_dup_codes = df[df.duplicated(subset=['Farmercode'], keep=False)][['Farmercode','username']]
df_dup_codes = df_dup_codes.assign(inconsistency="Duplicate Farmer code")

# 4. Productive Plants (flagged as high risk)
if 'Productiveplants' in df.columns:
    gdf_plants = gdf.copy()
    gdf_plants['acres'] = gdf_plants.geometry.area * 0.000247105
    gdf_plants['expected_plants'] = gdf_plants['acres'] * 450
    gdf_plants['productiveplants'] = pd.to_numeric(gdf_plants['Productiveplants'], errors='coerce')
    df_prod_high = gdf_plants[gdf_plants['productiveplants'] > gdf_plants['expected_plants']]
    df_prod_high = pd.DataFrame(df_prod_high[['Farmercode','username']])
    df_prod_high = df_prod_high.assign(inconsistency="Productiveplants exceed expected per acre")
    df_prod_low = gdf_plants[gdf_plants['productiveplants'] < (gdf_plants['expected_plants'] / 2)]
    df_prod_low = pd.DataFrame(df_prod_low[['Farmercode','username']])
    df_prod_low = df_prod_low.assign(inconsistency="Productiveplants less than half expected per acre")
else:
    df_prod_high = pd.DataFrame(columns=['Farmercode','username','inconsistency'])
    df_prod_low = pd.DataFrame(columns=['Farmercode','username','inconsistency'])

# 5. Uganda Boundary
uganda_coords = [
    (30.471786, -1.066837), (30.460829, -1.063428), (30.445614, -1.058694),
    # ... (complete with your full list) ...
    (33.904214, -1.002573), (33.822255, -1.002573)
]
uganda_poly = Polygon(uganda_coords)
gdf['in_uganda'] = gdf.geometry.within(uganda_poly)
df_uganda_incons = gdf.loc[gdf['in_uganda'], ['Farmercode','username']]
df_uganda_incons = pd.DataFrame(df_uganda_incons).assign(inconsistency="Plot lies within Uganda boundary")

# 6. Overlap (using updated thresholds)
overlap_list = []
for code in df['Farmercode'].unique():
    overlaps, overall_pct = check_overlaps(gdf, code)
    if overall_pct > 10:
        text = "Overlap > 10%"
    elif overall_pct >= 5:
        text = "Overlap 5-10%"
    else:
        text = None
    if text:
        overlap_list.append({'Farmercode': code, 'username': "", 'inconsistency': text})
df_overlap_incons = pd.DataFrame(overlap_list)

# 7. More Than 12 Codes Collected in a Day (High risk)
if 'SubmissionDate' in df.columns:
    df['SubmissionDate'] = pd.to_datetime(df['SubmissionDate'], errors='coerce').dt.date
    group = df.groupby(['username', 'SubmissionDate']).size().reset_index(name='count')
    high_code_groups = group[group['count'] > 12]
    df_codes_collected = df.merge(high_code_groups[['username','SubmissionDate']], on=['username','SubmissionDate'], how='inner')
    df_codes_collected = df_codes_collected[['Farmercode','username']].drop_duplicates()
    df_codes_collected = df_codes_collected.assign(inconsistency="More than 12 codes collected in a day")
else:
    df_codes_collected = pd.DataFrame(columns=['Farmercode','username','inconsistency'])

# 8. GPS Distance Check (if gps columns exist)
if 'gps-Latitude' in df.columns and 'gps-Longitude' in df.columns:
    def make_point(row):
        try:
            return Point(float(row['gps-Latitude']), float(row['gps-Longitude']))
        except:
            return None
    gdf['gps_point'] = gdf.apply(make_point, axis=1)
    if gdf['gps_point'].notna().sum() > 0:
        gps_series = gpd.GeoSeries(gdf['gps_point'].dropna(), crs="EPSG:4326").to_crs('EPSG:2109')
        gdf.loc[gps_series.index, 'gps_point_proj'] = gps_series
        gdf['gps_distance'] = gdf.apply(lambda row: row['gps_point_proj'].distance(row['geometry']) 
                                        if pd.notnull(row.get('gps_point_proj')) else None, axis=1)
        df_gps_incons = gdf.loc[gdf['gps_distance'] > 100, ['Farmercode','username']]
        df_gps_incons = pd.DataFrame(df_gps_incons).assign(inconsistency="GPS is more than 100m from polygon")
    else:
        df_gps_incons = pd.DataFrame(columns=['Farmercode','username','inconsistency'])
else:
    df_gps_incons = pd.DataFrame(columns=['Farmercode','username','inconsistency'])

# Concatenate all inconsistencies
inconsistencies_df = pd.concat([
    df_time_incons, df_phone_incons, df_dup_phones, df_dup_codes,
    df_prod_high, df_prod_low, df_uganda_incons, df_overlap_incons,
    df_codes_collected, df_gps_incons
], ignore_index=True)

if not inconsistencies_df.empty:
    inconsistencies_df['Risk Rating'] = inconsistencies_df['inconsistency'].apply(get_risk_rating)
    inconsistencies_df['Trust Responses'] = inconsistencies_df['Risk Rating'].apply(lambda x: "No" if x=="High" else "Yes")
else:
    inconsistencies_df = pd.DataFrame(columns=['Farmercode','username','inconsistency','Risk Rating','Trust Responses'])

# ---------------------------
# TOP 10 INSPECTORS (Bar Chart)
# ---------------------------
if not inconsistencies_df.empty:
    high_risks = inconsistencies_df[inconsistencies_df['Risk Rating'] == "High"]
    high_risks = high_risks.dropna(subset=['username'])
    top10 = high_risks.groupby('username').size().reset_index(name='HighRiskCount').sort_values(by='HighRiskCount', ascending=False).head(10)
    st.subheader("Top 10 Inspectors with Most High Risks")
    if not top10.empty:
        st.bar_chart(top10.set_index('username'))
    else:
        st.write("No high risk records found for any inspector.")
else:
    st.write("No inconsistencies detected.")

# ---------------------------
# EXPORT (MERGED WITH RISK COLUMNS)
# ---------------------------
def export_with_inconsistencies_merged(main_gdf, inconsistencies_df):
    export_gdf = main_gdf.to_crs("EPSG:4326").copy()
    export_gdf['geometry'] = export_gdf['geometry'].apply(lambda geom: geom.wkt)
    merged_df = export_gdf.merge(
        inconsistencies_df[['Farmercode','username','inconsistency','Risk Rating','Trust Responses']],
        on=['Farmercode','username'], how='left'
    )
    merged_df['inconsistency'] = merged_df['inconsistency'].fillna("No Inconsistency")
    merged_df['Risk Rating'] = merged_df['Risk Rating'].fillna("None")
    merged_df['Trust Responses'] = merged_df['Trust Responses'].fillna("Yes")
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
        merged_df.to_excel(writer, index=False, sheet_name="Updated Form")
    towrite.seek(0)
    st.download_button(
        label="Download Updated Form with Risk Columns",
        data=towrite,
        file_name="updated_inspection_form_merged.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ---------------------------
# UI: Inspector-Specific Overlap Map & Area for Selected Farmer
# ---------------------------
farmer_list = gdf['Farmercode'].dropna().unique().tolist()
selected_code = st.selectbox("Select Farmer Code", farmer_list)

target_row = gdf[gdf['Farmercode'] == selected_code]
if not target_row.empty:
    area = target_row.geometry.iloc[0].area
    st.subheader("Target Polygon Area:")
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
# EXPORT BUTTON (Merged)
# ---------------------------
if st.button("Export Updated Form to Excel (Merged with Risk Columns)"):
    export_with_inconsistencies_merged(gdf, inconsistencies_df)
