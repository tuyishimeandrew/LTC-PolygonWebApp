#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from shapely.geometry import Polygon

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
            x, y, z = map(float, coords[:3])
            vertices.append((x, y))
        except ValueError:
            continue
    return Polygon(vertices) if len(vertices) >= 3 else None

def check_overlaps(df, target_code):
    target_poly = df[df['farmer_code'] == target_code]['polygon_z'].iloc[0]
    if not target_poly:
        return []
    
    overlaps = []
    for _, row in df.iterrows():
        if row['farmer_code'] == target_code:
            continue
        other_poly = row['polygon_z']
        if other_poly and target_poly.intersects(other_poly):
            overlap_area = target_poly.intersection(other_poly).area
            overlaps.append({
                'farmer_code': row['farmer_code'],
                'overlap_area': overlap_area,
                'total_area': target_poly.area
            })
    return overlaps

# Streamlit App
st.title("Polygon Overlap Checker")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        df = pd.read_csv(uploaded_file)
    
    df['polygon_z'] = df['polygonplot'].apply(parse_polygon_z)
    farmer_codes = df['farmer_code'].unique().tolist()
    
    selected_code = st.selectbox("Select Farmer Code", farmer_codes)
    
    if st.button("Check Overlaps"):
        results = check_overlaps(df, selected_code)
        
        if results:
            st.subheader("Overlap Results:")
            for result in results:
                st.write(f"Farmer {result['farmer_code']}:")
                st.write(f"Overlap Area: {result['overlap_area']:.2f} m²")
                st.write(f"Percentage of Total Area: {(result['overlap_area']/result['total_area'])*100:.2f}%")
                st.write("---")
        else:
            st.success("No overlaps found!")

