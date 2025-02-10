#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.wkt import loads as wkt_loads
from shapely.ops import unary_union

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def parse_polygon_z(polygon_str):
    """
    Parse a polygon string into a Shapely Polygon.
    Supports either a custom semicolon-delimited "x y z" format or a standard WKT string.
    """
    if not isinstance(polygon_str, str):
        return None
    polygon_str = polygon_str.strip()
    if polygon_str.upper().startswith("POLYGON"):
        try:
            return wkt_loads(polygon_str)
        except Exception:
            return None
    else:
        vertices = []
        for point in polygon_str.split(';'):
            point = point.strip()
            if not point:
                continue
            coords = point.split()
            if len(coords) < 3:
                continue
            try:
                x, y, z = map(float, coords[:3])
                vertices.append((x, y))
            except ValueError:
                continue
        return Polygon(vertices) if len(vertices) >= 3 else None

@st.cache_data
def load_github_excel(url):
    """Load an Excel file from a GitHub raw URL."""
    df = pd.read_excel(url)
    return df

@st.cache_data
def load_uganda_boundary():
    """
    Load the Uganda boundary from an Excel file hosted on GitHub.
    The file must have a column named 'geometry' with WKT strings.
    Update the URL with your actual GitHub raw link.
    """
    url = "https://raw.githubusercontent.com/yourusername/yourrepository/main/UGANDA%20MAP.xlsx"  # UPDATE THIS URL
    df = load_github_excel(url)
    df['geometry'] = df['geometry'].apply(lambda x: wkt_loads(x) if isinstance(x, str) else None)
    df = df[df['geometry'].notnull()].copy()
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    # Reproject to Uganda UTM zone (EPSG:32636)
    gdf = gdf.to_crs(epsg=32636)
    # Combine all features into a single boundary if needed
    uganda_boundary = unary_union(gdf.geometry.tolist())
    return uganda_boundary

@st.cache_data
def load_protected_areas():
    """
    Load the protected areas (Game Parks) from an Excel file hosted on GitHub.
    The file must have a column named 'geometry' with WKT strings.
    Update the URL with your actual GitHub raw link.
    """
    url = "https://raw.githubusercontent.com/yourusername/yourrepository/main/Uganda%20Game%20Parks.xlsx"  # UPDATE THIS URL
    df = load_github_excel(url)
    df['geometry'] = df['geometry'].apply(lambda x: wkt_loads(x) if isinstance(x, str) else None)
    df = df[df['geometry'].notnull()].copy()
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=32636)
    return gdf

def load_main_data(uploaded_file):
    """
    Load the main dataset (Excel or CSV) uploaded by the user.
    The file should have a 'Farmercode' column (unique identifier) and a 'polygonplot' column
    containing the polygon data (in custom "x y z" format or as a WKT string).
    """
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        df = pd.read_csv(uploaded_file)
    df['polygon_z'] = df['polygonplot'].apply(parse_polygon_z)
    df = df[df['polygon_z'].notnull()].copy()
    # Assume input coordinates are in EPSG:4326 (WGS84)
    gdf = gpd.GeoDataFrame(df, geometry='polygon_z', crs="EPSG:4326")
    # Reproject to Uganda UTM zone (EPSG:32636) for accurate area and overlap computations
    gdf = gdf.to_crs(epsg=32636)
    return gdf

def check_in_uganda(polygon, uganda_boundary):
    """Check whether the given polygon is within or intersects the Uganda boundary."""
    return polygon.within(uganda_boundary) or polygon.intersects(uganda_boundary)

def check_overlaps(gdf, target_code):
    """
    For a given Farmercode (target_code), check overlaps between its polygon and all other polygons.
    Returns the target polygon, a list of overlap details for each overlapping polygon, and
    the cumulative overlap percentage (union of all overlaps).
    """
    target_row = gdf[gdf['Farmercode'].str.lower() == target_code.lower()]
    if target_row.empty:
        return None, None, None
    target_poly = target_row.iloc[0].geometry
    target_area = target_poly.area
    overlaps = []
    intersection_geoms = []
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
    return target_poly, overlaps, cumulative_overlap_percentage

def check_protected_area_overlap(polygon, protected_gdf):
    """
    Check if the given polygon overlaps any protected area (Game Park).
    Returns a list of protected areas with their names and overlap details.
    """
    overlapping_parks = []
    for idx, park in protected_gdf.iterrows():
        park_poly = park.geometry
        if polygon.intersects(park_poly):
            intersection = polygon.intersection(park_poly)
            park_name = park.get('Name', f'Park {idx}')
            overlapping_parks.append({
                'park_name': park_name,
                'overlap_area': intersection.area
            })
    return overlapping_parks

# -----------------------------------------------------------------------------
# Streamlit App Layout
# -----------------------------------------------------------------------------
st.title("Polygon Overlap & Uganda Boundary Checker")

st.markdown("""
This app allows you to:
- **Upload your main dataset** (Excel or CSV) containing polygon data.
- **Select a Farmercode** to check:
  - Whether the polygon is within Uganda.
  - The detailed list of overlapping polygons (by Farmercode) with their overlap percentages.
  - The cumulative percentage of the target polygon that is overlapped.
  - Whether it overlaps any protected areas (Game Parks).

**Dataset requirements:**  
• A column named **Farmercode** (unique identifier for each polygon).  
• A column named **polygonplot** containing polygon data (either as a custom "x y z" string or as a WKT string starting with "POLYGON").
""")

# Upload Main Dataset
uploaded_file = st.file_uploader("Upload Main Dataset", type=["xlsx", "csv"])

if uploaded_file is not None:
    gdf_main = load_main_data(uploaded_file)
    st.success("Main dataset loaded successfully!")
    
    # Load Uganda boundary and protected areas from GitHub
    uganda_boundary = load_uganda_boundary()
    protected_gdf = load_protected_areas()
    
    # List available Farmercodes (case-insensitive search)
    farmer_codes = gdf_main['Farmercode'].unique().tolist()
    selected_code = st.selectbox("Select Farmercode to Check", farmer_codes)
    
    if st.button("Run Checks"):
        target_poly, overlaps, cumulative_overlap_pct = check_overlaps(gdf_main, selected_code)
        if target_poly is None:
            st.error("The selected Farmercode was not found in the dataset.")
        else:
            # Check if the target polygon is within Uganda
            in_uganda = check_in_uganda(target_poly, uganda_boundary)
            st.write(f"**Is the polygon within Uganda?** {'Yes' if in_uganda else 'No'}")
            
            # Report target polygon area
            target_area_sqm = target_poly.area
            target_area_acres = target_area_sqm * 0.000247105
            st.write(f"**Target Polygon Area:** {target_area_sqm:.2f} m² ({target_area_acres:.2f} acres)")
            
            # Report cumulative overlap percentage
            st.subheader("Overall Overlap:")
            st.write(f"**Cumulative Overlap:** {cumulative_overlap_pct:.2f}% of the target polygon's area is overlapped.")
            
            # Report overlaps with other polygons individually
            if overlaps:
                st.subheader("Detailed Overlap with Other Polygons:")
                for overlap in overlaps:
                    st.write(f"**Farmercode:** {overlap['Farmercode']}")
                    st.write(f"Overlap Area: {overlap['overlap_area']:.2f} m²")
                    st.write(f"Overlap Percentage: {overlap['overlap_percentage']:.2f}%")
                    st.markdown("---")
            else:
                st.info("No overlaps found with other polygons.")
            
            # Check for overlaps with protected areas (Game Parks)
            protected_overlaps = check_protected_area_overlap(target_poly, protected_gdf)
            if protected_overlaps:
                st.subheader("Overlaps with Protected Areas (Game Parks):")
                for park in protected_overlaps:
                    percentage = (park['overlap_area'] / target_area_sqm) * 100 if target_area_sqm > 0 else 0
                    st.write(f"**Park:** {park['park_name']}")
                    st.write(f"Overlap Area: {park['overlap_area']:.2f} m²")
                    st.write(f"Overlap Percentage of Target: {percentage:.2f}%")
                    st.markdown("---")
            else:
                st.info("No overlaps with protected areas found.")
else:
    st.info("Please upload your main dataset file.")
