import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
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

# Function to check overlaps with ALL polygons
def check_overlaps(gdf, target_code):
    """
    Computes the total overlap area of the target polygon with all other polygons.
    Returns individual overlaps and the overall overlap percentage.
    """
    target_row = gdf[gdf['Farmercode'] == target_code]
    if target_row.empty:
        return [], 0
    
    target_poly = target_row.geometry.iloc[0]
    total_overlap_area = 0  # Tracks the total combined overlap area
    individual_overlaps = []
    overlapping_polygons = []

    for _, row in gdf.iterrows():
        if row['Farmercode'] == target_code:
            continue
        other_poly = row.geometry

        if other_poly and target_poly.intersects(other_poly):
            intersection = target_poly.intersection(other_poly)
            overlap_area = intersection.area if not intersection.is_empty else 0

            if overlap_area > 1e-6:  # Ignore tiny overlaps
                total_overlap_area += overlap_area
                overlapping_polygons.append(intersection)  # Store overlapping polygons

                individual_overlaps.append({
                    'Farmercode': row['Farmercode'],
                    'overlap_area': overlap_area
                })

    # Compute the total unique overlap by merging overlapping polygons
    if overlapping_polygons:
        unioned_overlap = gpd.GeoSeries(overlapping_polygons).unary_union
        total_overlap_area = unioned_overlap.area

    # Compute overall overlap percentage
    overall_percentage = (total_overlap_area / target_poly.area) * 100 if target_poly.area else 0

    return individual_overlaps, overall_percentage

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

    # Debugging: Check the CRS
    st.write(f"Coordinate Reference System: {gdf.crs}")

    # Debugging: Display some geometries
    st.write("Sample Data:", gdf[['Farmercode', 'geometry']].head())

    # Get unique farmer codes
    farmer_codes = gdf['Farmercode'].dropna().unique().tolist()
    if not farmer_codes:
        st.error("No Farmer codes found in the uploaded file.")
        st.stop()

    # Farmer selection dropdown
    selected_code = st.selectbox("Select Farmer Code", farmer_codes)

    # Visualize polygons
    st.subheader("Visualizing Farmer Polygons")
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(ax=ax, color='blue', alpha=0.5, edgecolor='black')
    plt.title("Farmer Plots")
    st.pyplot(fig)

    # Button to check overlaps
    if st.button("Check Overlaps"):
        results, overall_percentage = check_overlaps(gdf, selected_code)

        if results:
            st.subheader("Overlap Results:")
            for result in results:
                st.write(f"**Farmer {result['Farmercode']}**:")
                st.write(f"- Overlap Area: {result['overlap_area']:.2f} m²")
                st.write("---")
            
            # Show the overall overlap percentage
            st.subheader("Overall Overlap Summary")
            st.write(f"🔴 **Total Overlap Percentage:** {overall_percentage:.2f}% of the target area")
        else:
            st.success("No overlaps found!")
