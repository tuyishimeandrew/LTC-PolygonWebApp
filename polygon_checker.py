import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.wkt import loads
from rtree import index

# ----------------------------------------------------
# Helper: Load a file with polygon geometries from Excel/CSV/GeoJSON
# ----------------------------------------------------
def load_polygon_file(uploaded_file, geometry_column='geometry', default_crs="EPSG:4326"):
    """
    Loads an Excel, CSV, or GeoJSON file that contains polygon geometries (as WKT strings)
    in the specified column.
    """
    file_name = uploaded_file.name.lower()
    if file_name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
        if geometry_column not in df.columns:
            st.error(f"Excel file must have a '{geometry_column}' column containing WKT geometries.")
            return None
        # Convert the WKT string to Shapely geometries
        df[geometry_column] = df[geometry_column].apply(lambda x: loads(x) if isinstance(x, str) else None)
        return gpd.GeoDataFrame(df, geometry=geometry_column, crs=default_crs)
    elif file_name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        if geometry_column not in df.columns:
            st.error(f"CSV file must have a '{geometry_column}' column containing WKT geometries.")
            return None
        df[geometry_column] = df[geometry_column].apply(lambda x: loads(x) if isinstance(x, str) else None)
        return gpd.GeoDataFrame(df, geometry=geometry_column, crs=default_crs)
    else:
        # Assume it's a GeoJSON file
        return gpd.read_file(uploaded_file).to_crs(default_crs)

# ----------------------------------------------------
# Helper: Parse the farmer polygon (from the "polygonplot" column)
# ----------------------------------------------------
def parse_farmer_polygon(polygon_str):
    """Parse a polygon string (assumed to be in WKT format) into a Shapely Polygon."""
    try:
        return loads(polygon_str) if isinstance(polygon_str, str) else None
    except Exception as e:
        return None

# ----------------------------------------------------
# Helper: Check for overlaps between a target polygon and all others
# ----------------------------------------------------
def check_overlaps(gdf, target_code, code_column='Farmercode'):
    target_poly = gdf[gdf[code_column] == target_code]['geometry'].iloc[0]
    if not target_poly:
        return []
    overlaps = []
    for _, row in gdf.iterrows():
        if row[code_column] == target_code:
            continue
        other_poly = row['geometry']
        if other_poly and target_poly.intersects(other_poly):
            overlap_area = target_poly.intersection(other_poly).area
            overlaps.append({
                code_column: row[code_column],
                'overlap_area': overlap_area,
                'total_area': target_poly.area
            })
    return overlaps

# ----------------------------------------------------
# Helper: Check if a polygon lies within Uganda
# ----------------------------------------------------
def check_within_uganda(farmer_poly, uganda_gdf):
    """
    Returns True if farmer_poly is within any of the Uganda polygons.
    """
    return any(farmer_poly.within(poly) for poly in uganda_gdf.geometry)

# ----------------------------------------------------
# Helper: Check if a polygon overlaps any protected area
# ----------------------------------------------------
def check_protected_areas(farmer_poly, protected_gdf):
    """
    Returns True if farmer_poly intersects any protected area.
    """
    return any(farmer_poly.intersects(poly) for poly in protected_gdf.geometry)

# ----------------------------------------------------
# Streamlit App Layout
# ----------------------------------------------------
st.title("LTC Polygon Checker")

st.sidebar.header("Data Uploads")

# --- Uganda Boundary File (Excel/CSV/GeoJSON) ---
uganda_file = st.sidebar.file_uploader(
    "Upload Uganda Boundary (Excel/CSV/GeoJSON)",
    type=["xlsx", "csv", "geojson"],
    key="uganda"
)
uganda_gdf = None
if uganda_file is not None:
    uganda_gdf = load_polygon_file(uganda_file, geometry_column='geometry', default_crs="EPSG:4326")
    if uganda_gdf is not None:
        st.sidebar.success("Uganda boundary loaded successfully.")

# --- Protected Areas File (Excel/CSV/GeoJSON) ---
protected_file = st.sidebar.file_uploader(
    "Upload Protected Areas (Excel/CSV/GeoJSON)",
    type=["xlsx", "csv", "geojson"],
    key="protected"
)
protected_gdf = None
if protected_file is not None:
    protected_gdf = load_polygon_file(protected_file, geometry_column='geometry', default_crs="EPSG:4326")
    if protected_gdf is not None:
        st.sidebar.success("Protected areas loaded successfully.")

# --- Farmer Polygon Data (Excel/CSV) ---
st.subheader("Upload Farmer Polygon Data")
uploaded_farmer = st.file_uploader(
    "Upload Farmer Data (Excel/CSV)",
    type=["xlsx", "csv"],
    key="farmer"
)

if uploaded_farmer:
    file_name = uploaded_farmer.name.lower()
    if file_name.endswith('.xlsx'):
        df_farmer = pd.read_excel(uploaded_farmer, engine='openpyxl')
    else:
        df_farmer = pd.read_csv(uploaded_farmer)
    
    # Expect these columns: "polygonplot" (with WKT) and "Farmercode" (unique identifier)
    if 'polygonplot' not in df_farmer.columns:
        st.error("The farmer data must have a 'polygonplot' column with polygon WKT strings.")
    elif 'Farmercode' not in df_farmer.columns:
        st.error("The farmer data must have a 'Farmercode' column.")
    else:
        df_farmer['geometry'] = df_farmer['polygonplot'].apply(parse_farmer_polygon)
        gdf_farmer = gpd.GeoDataFrame(df_farmer, geometry='geometry', crs="EPSG:4326")
        
        st.write("### Farmer Data Preview:")
        st.write(gdf_farmer.head())
        
        # If available, add flags for being inside Uganda and for protected areas
        if uganda_gdf is not None:
            gdf_farmer['inside_uganda'] = gdf_farmer['geometry'].apply(lambda x: check_within_uganda(x, uganda_gdf))
        if protected_gdf is not None:
            gdf_farmer['in_protected_area'] = gdf_farmer['geometry'].apply(lambda x: check_protected_areas(x, protected_gdf))
        
        # Select a farmer code for analysis
        farmer_codes = gdf_farmer['Farmercode'].unique().tolist()
        selected_code = st.selectbox("Select Farmer Code", farmer_codes)
        
        if st.button("Run Analysis"):
            results = check_overlaps(gdf_farmer, selected_code, code_column='Farmercode')
            target_poly = gdf_farmer[gdf_farmer['Farmercode'] == selected_code]['geometry'].iloc[0]
            
            st.write(f"## Analysis for Farmer Code: {selected_code}")
            # Report if the polygon is inside Uganda
            if uganda_gdf is not None:
                inside = check_within_uganda(target_poly, uganda_gdf)
                st.write(f"**Inside Uganda:** {'Yes' if inside else 'No'}")
            
            # Report if the polygon overlaps a protected area
            if protected_gdf is not None:
                in_protected = check_protected_areas(target_poly, protected_gdf)
                st.write(f"**Overlaps Protected Area:** {'Yes' if in_protected else 'No'}")
            
            # Report overlaps with other farmer polygons
            if results:
                st.write("### Overlap Results:")
                for res in results:
                    perc = (res['overlap_area'] / target_poly.area) * 100 if target_poly.area else 0
                    st.write(f"Farmer {res['Farmercode']}: Overlap Area = {res['overlap_area']:.2f} m², which is {perc:.2f}% of the target area.")
                    st.write("---")
            else:
                st.success("No overlaps found!")
