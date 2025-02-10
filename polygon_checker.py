import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

# Function to parse polygon data (removes Z-coordinates)
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

# Function to check overlaps and calculate overall percentage
def check_overlaps(gdf, target_code):
    """
    Finds all polygons in the GeoDataFrame that overlap with the target polygon.
    Returns details of individual overlaps, and computes the overall overlap percentage.
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

# Streamlit App
st.title("Polygon Overlap Checker")

# File Upload
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

    # Ensure required columns exist
    if 'polygonplot' not in df.columns or 'Farmercode' not in df.columns:
        st.error("The uploaded file must contain 'polygonplot' and 'Farmercode' columns.")
        st.stop()

    # Convert polygon strings to Shapely Polygons
    df['polygon_z'] = df['polygonplot'].apply(parse_polygon_z)

    # Remove rows with invalid geometries
    df = df[df['polygon_z'].notna()]

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='polygon_z', crs='EPSG:4326')
    gdf = gdf.rename_geometry('geometry')

    # Reproject to Uganda's National Grid (EPSG:2109)
    gdf = gdf.to_crs('EPSG:2109')

    # Fix invalid geometries
    gdf['geometry'] = gdf['geometry'].buffer(0)

    # Validate geometries
    gdf = gdf[gdf.is_valid]

    # Get unique farmer codes
    farmer_codes = gdf['Farmercode'].dropna().unique().tolist()
    if not farmer_codes:
        st.error("No Farmer codes found in the uploaded file.")
        st.stop()

    # Farmer selection dropdown
    selected_code = st.selectbox("Select Farmer Code", farmer_codes)

    # Button to check overlaps
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

            # Display overall overlap percentage
            st.subheader("Overall Overlap Summary:")
            st.write(f"🔹 **Total Overlap Percentage (Union): {overall_percentage:.2f}%**")
        else:
            st.success("No overlaps found!")

    # Calculate total area in acres when a code is entered
    if selected_code:
        target_row = gdf[gdf['Farmercode'] == selected_code]
        if not target_row.empty:
            target_area_m2 = target_row.geometry.iloc[0].area
            target_area_acres = target_area_m2 * 0.000247105  # Convert square meters to acres
            st.subheader("Target Polygon Area:")
            st.write(f"Total Area: {target_area_m2:.2f} m² ({target_area_acres:.4f} acres)")
