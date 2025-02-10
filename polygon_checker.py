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

# Function to check overlaps with all polygons
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

    return individual_overl
