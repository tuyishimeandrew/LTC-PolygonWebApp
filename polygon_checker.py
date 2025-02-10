import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

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
            # Extract x, y (ignore z) from the point string.
            x, y, _ = map(float, coords[:3])
            vertices.append((x, y))
        except ValueError:
            continue
    return Polygon(vertices) if len(vertices) >= 3 else None

def check_overlaps(gdf, target_code):
    """
    For the given target_code, find all other polygons in the GeoDataFrame
    that overlap with the target polygon. Returns a list of dictionaries
    containing the overlapping farmer code, the overlap area, and the
    total area of the target polygon.
    """
    target_row = gdf[gdf['Farmercode'] == target_code]
    if target_row.empty:
        return []
    # Use the active geometry column.
    target_poly = target_row.geometry.iloc[0]
    overlaps = []
    for _, row in gdf.iterrows():
        if row['Farmercode'] == target_code:
            continue
        other_poly = row.geometry
        if other_poly and target_poly.intersects(other_poly):
            intersection = target_poly.intersection(other_poly)
            overlap_area = intersection.area if not intersection.is_empty else 0
            overlaps.append({
                'Farmercode': row['Farmercode'],
                'overlap_area': overlap_area,
                'total_area': target_poly.area
            })
    return overlaps

st.title("Polygon Overlap Checker")

uploaded_file = st.file_uploader("Upload CSV or Excel File", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error("Error loading file: " + str(e))
        st.stop()
    
    # Validate that required columns exist.
    if 'polygonplot' not in df.columns or 'Farmercode' not in df.columns:
        st.error("The uploaded file must contain 'polygonplot' and 'Farmercode' columns.")
        st.stop()
    
    # Parse the polygon data from the 'polygonplot' column.
    df['polygon_z'] = df['polygonplot'].apply(parse_polygon_z)
    
    # Create a GeoDataFrame using the parsed polygons.
    # Here we assume the input coordinates are in lat/lon (EPSG:4326).
    gdf = gpd.GeoDataFrame(df, geometry='polygon_z', crs='EPSG:4326')
    
    # Rename the geometry column to "geometry" to ensure that gdf.geometry works.
    gdf = gdf.rename_geometry('geometry')
    
    # Reproject the GeoDataFrame to Uganda's National Grid (EPSG:2109)
    # so that area calculations (in m²) are correct.
    gdf = gdf.to_crs('EPSG:2109')
    
    # Get the list of unique farmer codes.
    farmer_codes = gdf['Farmercode'].dropna().unique().tolist()
    if not farmer_codes:
        st.error("No Farmer codes found in the uploaded file.")
        st.stop()
    
    selected_code = st.selectbox("Select Farmer Code", farmer_codes)
    
    if st.button("Check Overlaps"):
        results = check_overlaps(gdf, selected_code)
        
        if results:
            st.subheader("Overlap Results:")
            for result in results:
                percentage = (result['overlap_area'] / result['total_area']) * 100 if result['total_area'] else 0
                st.write(f"**Farmer {result['Farmercode']}**:")
                st.write(f"- Overlap Area: {result['overlap_area']:.2f} m²")
                st.write(f"- Percentage of Target Area: {percentage:.2f}%")
                st.write("---")
        else:
            st.success("No overlaps found!")
