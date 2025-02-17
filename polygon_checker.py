import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Latitude Polygon Overlap Checker", layout="wide")

# --- CUSTOM CSS FOR STYLING ---
custom_css = """
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .section-box {
        background-color: #ecf0f1;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-size: 1rem;
        margin: 0.5rem 0;
    }
    .stExpander .streamlit-expanderHeader {
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- APP TITLE ---
st.markdown('<div class="main-title">Latitude Polygon Overlap Checker</div>', unsafe_allow_html=True)

# --- UPLOAD SECTIONS ---
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown("<strong>Upload Inspection Files</strong>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.subheader("Main Inspection File")
    main_file = st.file_uploader("Upload Main Inspection Form (CSV or Excel)",
                                 type=["xlsx", "csv"], key="main_upload")
with col2:
    st.subheader("Redo Polygon File")
    redo_file = st.file_uploader("Upload Redo Polygon Form (CSV or Excel)",
                                 type=["xlsx","csv"], key="redo_upload")
st.markdown('</div>', unsafe_allow_html=True)

# --- PROCESS MAIN FILE ---
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

# Ensure required columns exist in the main file
if 'Farmercode' not in df.columns or 'polygonplot' not in df.columns:
    st.error("The Main Inspection Form must contain 'Farmercode' and 'polygonplot' columns.")
    st.stop()

# --- PROCESS REDO FILE ---
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

    # Rename column if necessary
    if "farmer_code" in df_redo.columns:
        df_redo = df_redo.rename(columns={'farmer_code': 'Farmercode'})

    required_redo_cols = ['Farmercode', 'selectplot', 'polygonplot']
    if not all(col in df_redo.columns for col in required_redo_cols):
        st.error("Redo polygon file must contain 'Farmercode', 'selectplot', and 'polygonplot' columns.")
        st.stop()

    # Use SubmissionDate and endtime to select the latest submission if available
    if 'SubmissionDate' in df_redo.columns and 'endtime' in df_redo.columns:
        df_redo['SubmissionDate'] = pd.to_datetime(df_redo['SubmissionDate'], errors='coerce')
        df_redo['endtime'] = pd.to_datetime(df_redo['endtime'], errors='coerce')
        df_redo = df_redo.sort_values(by=['SubmissionDate', 'endtime'])
        df_redo = df_redo.groupby('Farmercode', as_index=False).last()

    # Rename columns to avoid conflict
    df_redo = df_redo.rename(columns={'selectplot': 'redo_selectplot',
                                        'polygonplot': 'redo_polygonplot'})

    # Merge the redo data with the main file on Farmercode
    df = df.merge(df_redo[['Farmercode', 'redo_selectplot', 'redo_polygonplot']],
                  on='Farmercode', how='left')

    # --- CONDITIONS TO UPDATE POLYGONS ---
    cond1 = df['polygonplot'].notna() & (df['redo_selectplot'] == 'Plot1')
    df.loc[cond1, 'polygonplot'] = df.loc[cond1, 'redo_polygonplot']

    if 'polygonplotnew_1' in df.columns:
        cond2 = df['polygonplotnew_1'].notna() & (df['redo_selectplot'] == 'Plot2')
        df.loc[cond2, 'polygonplot'] = df.loc[cond2, 'redo_polygonplot']

    if 'polygonplotnew_2' in df.columns:
        cond3 = df['polygonplotnew_2'].notna() & (df['redo_selectplot'] == 'Plot3')
        df.loc[cond3, 'polygonplotnew_2'] = df.loc[cond3, 'redo_polygonplot']

    if 'polygonplotnew_3' in df.columns:
        cond4 = df['polygonplotnew_3'].notna() & (df['redo_selectplot'] == 'Plot4')
        df.loc[cond4, 'polygonplotnew_3'] = df.loc[cond4, 'redo_polygonplot']

    if 'polygonplotnew_4' in df.columns:
        cond5 = df['polygonplotnew_4'].notna() & (df['redo_selectplot'] == 'Plot5')
        df.loc[cond5, 'polygonplotnew_4'] = df.loc[cond5, 'redo_polygonplot']

    df = df.drop(columns=['redo_selectplot', 'redo_polygonplot'])

# --- FUNCTIONS FOR POLYGON HANDLING ---
def parse_polygon_z(polygon_str):
    if not isinstance(polygon_str, str):
        return None
    vertices = []
    for point in polygon_str.split(';'):
        point = point.strip()
        if not point:
            continue
        coords = point.split()
        if len(coords) < 3:
            continue
        try:
            x, y, _ = map(float, coords[:3])
            vertices.append((x, y))
        except ValueError:
            continue
    return Polygon(vertices) if len(vertices) >= 3 else None

def combine_polygons(row):
    poly_list = []
    for col in ['polygonplot', 'polygonplotnew_1', 'polygonplotnew_2', 'polygonplotnew_3', 'polygonplotnew_4']:
        if col in row and pd.notna(row[col]):
            poly = parse_polygon_z(row[col])
            if poly is not None:
                if not poly.is_valid:
                    poly = poly.buffer(0)
                poly_list.append(poly)
    if not poly_list:
        return None
    valid_polys = [p for p in poly_list if p.is_valid]
    if not valid_polys:
        return None
    if len(valid_polys) == 1:
        return valid_polys[0]
    try:
        return unary_union(valid_polys)
    except Exception as e:
        st.error("Error during union: " + str(e))
        return None

def check_overlaps(gdf, target_code):
    target_row = gdf[gdf['Farmercode'] == target_code]
    if target_row.empty:
        return [], 0
    target_poly = target_row.geometry.iloc[0]
    total_target_area = target_poly.area
    overlaps = []
    union_overlap = None
    for _, row in gdf.iterrows():
        if row['Farmercode'] == target_code:
            continue
        other_poly = row.geometry
        if other_poly and target_poly.intersects(other_poly):
            intersection = target_poly.intersection(other_poly)
            overlap_area = intersection.area if not intersection.is_empty else 0
            if overlap_area > 1e-6:
                overlaps.append({
                    'Farmercode': row['Farmercode'],
                    'overlap_area': overlap_area,
                    'total_area': total_target_area,
                    'intersection': intersection
                })
                union_overlap = intersection if union_overlap is None else union_overlap.union(intersection)
    overall_overlap_area = union_overlap.area if union_overlap else 0
    overall_overlap_percentage = (overall_overlap_area / total_target_area) * 100 if total_target_area else 0
    return overlaps, overall_overlap_percentage

# --- CREATE GEOMETRY BY COMBINING POLYGON COLUMNS ---
df['geometry'] = df.apply(combine_polygons, axis=1)
df = df[df['geometry'].notna()]

# Convert to GeoDataFrame (assumed CRS EPSG:4326) and then to a local CRS
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
gdf = gdf.to_crs('EPSG:2109')
gdf['geometry'] = gdf['geometry'].buffer(0)
gdf = gdf[gdf.is_valid]

# --- SELECT FARMER CODE ---
farmer_codes = gdf['Farmercode'].dropna().unique().tolist()
if not farmer_codes:
    st.error("No Farmer codes found in the processed data.")
    st.stop()
selected_code = st.selectbox("Select Farmer Code", farmer_codes)

# --- CHECK OVERLAPS SECTION ---
with st.expander("Check Overlap Results", expanded=True):
    if st.button("Check Overlaps"):
        results, overall_percentage = check_overlaps(gdf, selected_code)
        if results:
            st.markdown("<div class='subheader'>Overlap Details:</div>", unsafe_allow_html=True)
            for result in results:
                percentage = (result['overlap_area'] / result['total_area']) * 100 if result['total_area'] else 0
                st.write(f"**Farmer {result['Farmercode']}**:")
                st.write(f"- Overlap Area: {result['overlap_area']:.2f} m²")
                st.write(f"- Percentage of Target Area: {percentage:.2f}%")
                st.markdown("---")
            st.markdown("<div class='subheader'>Overall Overlap Summary:</div>", unsafe_allow_html=True)
            st.write(f"🔹 **Total Overlap Percentage (Union): {overall_percentage:.2f}%**")
        else:
            st.success("No overlaps found!")

# --- PLOT THE OVERLAP MAP ---
with st.expander("Show Overlap Map", expanded=True):
    if st.button("Display Overlap Map"):
        target_row = gdf[gdf['Farmercode'] == selected_code]
        if target_row.empty:
            st.error("Selected Farmer not found.")
        else:
            target_poly = target_row.geometry.iloc[0]
            overlaps, _ = check_overlaps(gdf, selected_code)
            fig, ax = plt.subplots(figsize=(8, 8))
            # Plot target polygon
            x, y = target_poly.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='blue', ec='black', label=f"Target: {selected_code}")
            # Plot overlaps
            for overlap in overlaps:
                inter_geom = overlap['intersection']
                percent = (overlap['overlap_area'] / overlap['total_area']) * 100 if overlap['total_area'] else 0
                if inter_geom.geom_type == 'Polygon':
                    ix, iy = inter_geom.exterior.xy
                    ax.fill(ix, iy, alpha=0.5, fc='red', ec='darkred', label=f"Overlap {overlap['Farmercode']}")
                    cx, cy = inter_geom.centroid.x, inter_geom.centroid.y
                    ax.text(cx, cy, f"{percent:.1f}%", fontsize=10, color='white', ha='center', va='center')
                elif inter_geom.geom_type == 'MultiPolygon':
                    for geom in inter_geom.geoms:
                        ix, iy = geom.exterior.xy
                        ax.fill(ix, iy, alpha=0.5, fc='red', ec='darkred', label=f"Overlap {overlap['Farmercode']}")
                        cx, cy = geom.centroid.x, geom.centroid.y
                        ax.text(cx, cy, f"{percent:.1f}%", fontsize=10, color='white', ha='center', va='center')
            ax.set_title(f"Overlap Map for Farmer {selected_code}")
            ax.set_xlabel("Easting")
            ax.set_ylabel("Northing")
            ax.legend(loc='upper right', fontsize='small')
            st.pyplot(fig)

# --- DISPLAY TARGET POLYGON AREA ---
with st.expander("Target Polygon Area", expanded=True):
    target_row = gdf[gdf['Farmercode'] == selected_code]
    if not target_row.empty:
        target_area_m2 = target_row.geometry.iloc[0].area
        target_area_acres = target_area_m2 * 0.000247105
        st.write(f"**Total Area:** {target_area_m2:.2f} m² ({target_area_acres:.4f} acres)")

# --- EXPORT UPDATED FORM ---
with st.expander("Export Updated Form", expanded=True):
    if st.button("Export Updated Form to Excel"):
        export_df = gdf.copy()
        export_df['geometry'] = export_df['geometry'].apply(lambda geom: geom.wkt)
        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Updated Form")
        towrite.seek(0)
        st.download_button(
            label="Download Updated Inspection Form",
            data=towrite,
            file_name="updated_inspection_form.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# --- DATA INCONSISTENCY CHECKS ---
with st.expander("Data Inconsistency Checks", expanded=True):
    if st.button("Check Data Inconsistencies"):
        required_inconsistency_cols = ['Farmercode', 'username', 'duration', 'Registered', 'Phone', 'Phone_hidden']
        missing_cols = [col for col in required_inconsistency_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns for inconsistency checks: {', '.join(missing_cols)}")
        else:
            time_inconsistency = df[(df['duration'] < 900) & (df['Registered'].str.lower()=='yes')]
            phone_mismatch = df[df['Phone'] != df['Phone_hidden']]
            duplicate_phones = df[df.duplicated(subset=['Phone'], keep=False)]
            duplicate_farmercodes = df[df.duplicated(subset=['Farmercode'], keep=False)]
            
            st.markdown("<div class='subheader'>Time Inconsistencies</div>", unsafe_allow_html=True)
            with st.expander("Duration < 15 mins but Registered == Yes"):
                if not time_inconsistency.empty:
                    st.write(time_inconsistency[['Farmercode', 'username', 'duration', 'Registered']])
                else:
                    st.write("No time inconsistencies found.")
            
            st.markdown("<div class='subheader'>Phone Mismatches</div>", unsafe_allow_html=True)
            with st.expander("Phone != Phone_hidden"):
                if not phone_mismatch.empty:
                    st.write(phone_mismatch[['Farmercode', 'username', 'Phone', 'Phone_hidden']])
                else:
                    st.write("No phone mismatches found.")
            
            st.markdown("<div class='subheader'>Duplicate Phone Entries</div>", unsafe_allow_html=True)
            with st.expander("Duplicate Phone Entries"):
                if not duplicate_phones.empty:
                    st.write(duplicate_phones[['Farmercode', 'username', 'Phone']])
                else:
                    st.write("No duplicate phone entries found.")
            
            st.markdown("<div class='subheader'>Duplicate Farmer Codes</div>", unsafe_allow_html=True)
            with st.expander("Duplicate Farmer Codes"):
                if not duplicate_farmercodes.empty:
                    st.write(duplicate_farmercodes[['Farmercode', 'username']])
                else:
                    st.write("No duplicate Farmer codes found.")
