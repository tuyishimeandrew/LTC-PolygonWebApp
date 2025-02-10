import requests
from io import BytesIO
import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.wkt import loads as wkt_loads
from shapely.ops import unary_union

# -----------------------------------------------------------------------------
# Global Debug Flag & Helper
# -----------------------------------------------------------------------------
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)

def debug_log(msg):
    if debug_mode:
        st.write("[DEBUG]", msg)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def parse_polygon_z(polygon_str):
    """
    Parse a polygon string into a Shapely Polygon.
    Supports either a custom semicolon-delimited "x y z" format or a standard WKT string.
    """
    if not isinstance(polygon_str, str):
        debug_log("Non-string polygon encountered.")
        return None
    polygon_str = polygon_str.strip()
    if polygon_str.upper().startswith("POLYGON"):
        try:
            poly = wkt_loads(polygon_str)
            debug_log("Parsed WKT polygon successfully.")
            return poly
        except Exception as e:
            debug_log(f"Failed to parse WKT polygon: {e}")
            return None
    else:
        vertices = []
        for point in polygon_str.split(';'):
            point = point.strip()
            if not point:
                continue
            coords = point.split()
            if len(coords) < 3:
                debug_log("Skipping point with insufficient coordinates: " + point)
                continue
            try:
                x, y, z = map(float, coords[:3])
                vertices.append((x, y))
            except ValueError as e:
                debug_log(f"Value error for point {point}: {e}")
                continue
        if len(vertices) >= 3:
            debug_log("Parsed custom polygon successfully with vertices: " + str(vertices))
            return Polygon(vertices)
        else:
            debug_log("Not enough vertices to form a polygon.")
            return None

@st.cache_data(show_spinner=False)
def load_github_data(url):
    """
    Load a file (Excel or CSV) from a GitHub raw URL.
    The function detects the file type by the URL extension.
    """
    debug_log(f"Downloading data from URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Will raise an error if the response is not OK
        debug_log("Download successful.")
    except Exception as e:
        debug_log(f"Error downloading file: {e}")
        st.error(f"Error downloading file from {url}: {e}")
        return None

    try:
        if url.lower().endswith('.csv'):
            debug_log("Detected CSV file.")
            df = pd.read_csv(BytesIO(response.content))
        else:
            debug_log("Assuming Excel file.")
            df = pd.read_excel(BytesIO(response.content), engine='openpyxl')
        debug_log("File loaded into DataFrame successfully.")
        return df
    except Exception as e:
        debug_log(f"Error reading file: {e}")
        st.error(f"Error reading file from {url}: {e}")
        return None

def load_main_data(uploaded_file):
    """
    Load the main dataset (Excel or CSV) uploaded by the user.
    The file should have a 'Farmercode' column (unique identifier) and a 'polygonplot' column
    containing the polygon data (in custom "x y z" format or as a WKT string).
    """
    try:
        if uploaded_file.name.lower().endswith('.xlsx'):
            debug_log("Uploaded file is an Excel file.")
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            debug_log("Uploaded file is assumed to be CSV.")
            df = pd.read_csv(uploaded_file)
        debug_log("Main dataset loaded successfully.")
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        debug_log(f"Error loading uploaded file: {e}")
        return None

    df['polygon_z'] = df['polygonplot'].apply(parse_polygon_z)
    df = df[df['polygon_z'].notnull()].copy()
    debug_log(f"Filtered main dataset to {len(df)} valid polygons.")
    # Assume input coordinates are in EPSG:4326 (WGS84)
    gdf = gpd.GeoDataFrame(df, geometry='polygon_z', crs="EPSG:4326")
    # Reproject to a projected coordinate system for accurate area/overlap calculations
    gdf = gdf.to_crs(epsg=32636)
    return gdf

def check_overlaps(gdf, target_code):
    """
    For a given Farmercode (target_code), check overlaps between its polygon and all other polygons.
    Returns the target polygon, a list of overlap details, and the cumulative overlap percentage.
    """
    target_row = gdf[gdf['Farmercode'].str.lower() == target_code.lower()]
    if target_row.empty:
        debug_log("Target Farmercode not found.")
        return None, None, None
    target_poly = target_row.iloc[0].geometry
    target_area = target_poly.area
    overlaps = []
    intersection_geoms = []
    debug_log("Checking overlaps with other polygons...")
    for _, row in gdf.iterrows():
        if row['Farmercode'].lower() == target_code.lower():
            continue
        other_poly = row.geometry
        if other_poly and target_poly.intersects(other_poly):
            intersection = target_poly.intersection(other_poly)
            if not intersection.is_empty:
                intersection_geoms.append(intersection)
                overlap_area = intersection.area
                overlaps.append({
                    'Farmercode': row['Farmercode'],
                    'overlap_area': overlap_area,
                    'overlap_percentage': (overlap_area / target_area) * 100 if target_area > 0 else 0
                })
    cumulative_overlap_area = 0.0
    if intersection_geoms:
        union_intersection = unary_union(intersection_geoms)
        cumulative_overlap_area = union_intersection.area
    cumulative_overlap_percentage = (cumulative_overlap_area / target_area) * 100 if target_area > 0 else 0
    debug_log(f"Found {len(overlaps)} overlapping polygons; cumulative overlap = {cumulative_overlap_percentage:.2f}%.")
    return target_poly, overlaps, cumulative_overlap_percentage

# -----------------------------------------------------------------------------
# Streamlit App Layout
# -----------------------------------------------------------------------------
st.title("Polygon Overlap Checker")

st.markdown("""
This app allows you to:
- **Upload your main dataset** (Excel or CSV) containing polygon data.
- **Select a Farmercode** to check:
  - Detailed overlapping information with other polygons.
  - The cumulative percentage of the target polygon's area that is overlapped.

**Dataset requirements:**  
• A column named **Farmercode**.  
• A column named **polygonplot** containing polygon data (in a custom "x y z" format or as a WKT string starting with "POLYGON").
""")

# File uploader for the main dataset
uploaded_file = st.file_uploader("Upload Main Dataset (CSV or Excel)", type=["xlsx", "csv"])

if uploaded_file is not None:
    gdf_main = load_main_data(uploaded_file)
    if gdf_main is not None:
        st.success("Main dataset loaded successfully!")
        debug_log(f"Loaded main dataset with {len(gdf_main)} entries.")

        # List available Farmercodes (case-insensitive)
        farmer_codes = gdf_main['Farmercode'].unique().tolist()
        selected_code = st.selectbox("Select Farmercode to Check", farmer_codes)
        
        if st.button("Run Checks"):
            target_poly, overlaps, cumulative_overlap_pct = check_overlaps(gdf_main, selected_code)
            if target_poly is None:
                st.error("The selected Farmercode was not found in the dataset.")
            else:
                # Report target polygon area
                target_area_sqm = target_poly.area
                target_area_acres = target_area_sqm * 0.000247105
                st.write(f"**Target Polygon Area:** {target_area_sqm:.2f} m² ({target_area_acres:.2f} acres)")
                
                # Report cumulative overlap percentage
                st.subheader("Overall Overlap:")
                st.write(f"**Cumulative Overlap:** {cumulative_overlap_pct:.2f}% of the target polygon's area is overlapped.")
                
                # Report individual overlaps
                if overlaps:
                    st.subheader("Detailed Overlap with Other Polygons:")
                    for overlap in overlaps:
                        st.write(f"**Farmercode:** {overlap['Farmercode']}")
                        st.write(f"Overlap Area: {overlap['overlap_area']:.2f} m²")
                        st.write(f"Overlap Percentage: {overlap['overlap_percentage']:.2f}%")
                        st.markdown("---")
                else:
                    st.info("No overlaps found with other polygons.")
    else:
        st.error("Error processing the uploaded main dataset.")
else:
    st.info("Please upload your main dataset file.")
