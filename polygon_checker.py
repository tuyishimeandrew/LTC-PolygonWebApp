import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import io

st.title("Polygon Overlap Checker")

# --- FUNCTIONS ---

def parse_polygon_z(polygon_str):
    """
    Parses a string with points in the format "x y z; x y z; ..." 
    and returns a Shapely Polygon using only the x and y coordinates.
    """
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
    """
    For a given row, parse any available polygon string from the following columns:
    'polygonplot', 'polygonplotnew_1', 'polygonplotnew_2', 'polygonplotnew_3', 'polygonplotnew_4'
    and return their union (or the single polygon if only one is present).
    """
    poly_list = []
    for col in ['polygonplot', 'polygonplotnew_1', 'polygonplotnew_2', 'polygonplotnew_3', 'polygonplotnew_4']:
        if col in row and pd.notna(row[col]):
            poly = parse_polygon_z(row[col])
            if poly is not None:
                poly_list.append(poly)
    if not poly_list:
        return None
    if len(poly_list) == 1:
        return poly_list[0]
    return unary_union(poly_list)

def check_overlaps(gdf, target_code):
    """
    Finds all polygons in the GeoDataFrame that overlap with the target farmer’s geometry.
    Returns details of individual overlaps and the overall union overlap percentage.
    """
    target_row = gdf[gdf['Farmercode'] == target_code]
    if target_row.empty:
        return [], 0
    
    target_poly = target_row.geometry.iloc[0]
    total_target_area = target_poly.area  # Area in square meters

    overlaps = []
    union_overlap = None  # To store the union of all intersections

    for _, row in gdf.iterrows():
        if row['Farmercode'] == target_code:
            continue
        other_poly = row.geometry

        if other_poly and target_poly.intersects(other_poly):
            intersection = target_poly.intersection(other_poly)
            overlap_area = intersection.area if not intersection.is_empty else 0

            if overlap_area > 1e-6:  # Ignore tiny overlaps
                overlaps.append({
                    'Farmercode': row['Farmercode'],
                    'overlap_area': overlap_area,
                    'total_area': total_target_area
                })
                
                # Compute union of intersections
                union_overlap = intersection if union_overlap is None else union_overlap.union(intersection)

    # Calculate overall overlap percentage
    overall_overlap_area = union_overlap.area if union_overlap else 0
    overall_overlap_percentage = (overall_overlap_area / total_target_area) * 100 if total_target_area else 0

    return overlaps, overall_overlap_percentage

# --- UPLOAD INSPECTION FILE ---

uploaded_file = st.file_uploader("Upload Inspection Form CSV or Excel File", type=["xlsx", "csv"])
if uploaded_file is None:
    st.info("Please upload the Inspection Form file.")
    st.stop()

try:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error("Error loading file: " + str(e))
    st.stop()

# Ensure required columns exist
if 'Farmercode' not in df.columns or 'polygonplot' not in df.columns:
    st.error("The uploaded file must contain 'Farmercode' and 'polygonplot' columns.")
    st.stop()

# --- OPTIONAL: UPLOAD REDO POLYGON FILE ---

redo_file = st.file_uploader("Upload Redo Polygon Excel File (Optional)", type=["xlsx"])
if redo_file is not None:
    try:
        df_redo = pd.read_excel(redo_file, engine='openpyxl')
    except Exception as e:
        st.error("Error loading redo polygon file: " + str(e))
        st.stop()

    # Check for required columns in redo file
    required_redo_cols = ['Farmercode', 'selectplot', 'polygonplot']
    if not all(col in df_redo.columns for col in required_redo_cols):
        st.error("Redo polygon file must contain 'Farmercode', 'selectplot', and 'polygonplot' columns.")
        st.stop()
        
    # Rename redo file columns to avoid conflict
    df_redo = df_redo.rename(columns={'selectplot': 'redo_selectplot', 'polygonplot': 'redo_polygonplot'})
    
    # Merge the redo data with the inspection form data on Farmercode
    df = df.merge(df_redo[['Farmercode', 'redo_selectplot', 'redo_polygonplot']], 
                  on='Farmercode', how='left')
    
    # --- Condition 1 ---
    cond1 = df['polygonplot'].notna() & (df['redo_selectplot'] == 'plot_1')
    df.loc[cond1, 'polygonplot'] = df.loc[cond1, 'redo_polygonplot']
    
    # --- Condition 2 ---
    if 'polygonplotnew_1' in df.columns:
        cond2 = df['polygonplotnew_1'].notna() & (df['redo_selectplot'] == 'plot_2')
        df.loc[cond2, 'polygonplot'] = df.loc[cond2, 'redo_polygonplot']
    
    # --- Condition 3 ---
    if 'polygonplotnew_2' in df.columns:
        cond3 = df['polygonplotnew_2'].notna() & (df['redo_selectplot'] == 'plot_3')
        df.loc[cond3, 'polygonplotnew_2'] = df.loc[cond3, 'redo_polygonplot']
    
    # --- Condition 4 ---
    if 'polygonplotnew_3' in df.columns:
        cond4 = df['polygonplotnew_3'].notna() & (df['redo_selectplot'] == 'plot_4')
        df.loc[cond4, 'polygonplotnew_3'] = df.loc[cond4, 'redo_polygonplot']
    
    # --- Condition 5 ---
    if 'polygonplotnew_4' in df.columns:
        cond5 = df['polygonplotnew_4'].notna() & (df['redo_selectplot'] == 'plot_5')
        df.loc[cond5, 'polygonplotnew_4'] = df.loc[cond5, 'redo_polygonplot']
    
    # Optionally, drop the redo columns after updating
    df = df.drop(columns=['redo_selectplot', 'redo_polygonplot'])

# --- CREATE GEOMETRY BY COMBINING POLYGON COLUMNS ---

# For each row, combine the polygons available from the possible columns.
df['geometry'] = df.apply(combine_polygons, axis=1)

# Remove rows where no valid geometry could be created
df = df[df['geometry'].notna()]

# Convert to GeoDataFrame (assuming the coordinates are in EPSG:4326)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

# Reproject to Uganda's National Grid (EPSG:2109)
gdf = gdf.to_crs('EPSG:2109')

# Fix invalid geometries
gdf['geometry'] = gdf['geometry'].buffer(0)
gdf = gdf[gdf.is_valid]

# --- SELECT FARMER CODE AND CHECK OVERLAPS ---

farmer_codes = gdf['Farmercode'].dropna().unique().tolist()
if not farmer_codes:
    st.error("No Farmer codes found in the uploaded file.")
    st.stop()

selected_code = st.selectbox("Select Farmer Code", farmer_codes)

if st.button("Check Overlaps"):
    results, overall_percentage = check_overlaps(gdf, selected_code)
    
    if results:
        st.subheader("Overlap Results:")
        for result in results:
            percentage = (result['overlap_area'] / result['total_area']) * 100 if result['total_area'] else 0
            st.write(f"**Farmer {result['Farmercode']}**:")
            st.write(f"- Overlap Area: {result['overlap_area']:.2f} m²")
            st.write(f"- Percentage of Target Area: {percentage:.2f}%")
            st.write("---")
        
        st.subheader("Overall Overlap Summary:")
        st.write(f"🔹 **Total Overlap Percentage (Union): {overall_percentage:.2f}%**")
    else:
        st.success("No overlaps found!")

# --- DISPLAY TARGET POLYGON AREA ---

target_row = gdf[gdf['Farmercode'] == selected_code]
if not target_row.empty:
    target_area_m2 = target_row.geometry.iloc[0].area
    target_area_acres = target_area_m2 * 0.000247105  # Conversion factor from m² to acres
    st.subheader("Target Polygon Area:")
    st.write(f"Total Area: {target_area_m2:.2f} m² ({target_area_acres:.4f} acres)")

# --- EXPORT UPDATED FORM AS EXCEL ---

if st.button("Export Updated Form to Excel"):
    # Create a copy for export
    export_df = gdf.copy()
    # Convert the geometry column to WKT strings for export readability
    export_df['geometry'] = export_df['geometry'].apply(lambda geom: geom.wkt)
    
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Updated Form")
        writer.save()
    towrite.seek(0)
    
    st.download_button(
        label="Download Updated Inspection Form",
        data=towrite,
        file_name="updated_inspection_form.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
