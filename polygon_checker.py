import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
import io

st.title("Latitude Polygon Inconsistency Checker")

# --- Display Two Upload Areas Side-by-Side ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Main Inspection File")
    main_file = st.file_uploader("Upload Main Inspection Form (CSV or Excel)",
                                 type=["xlsx", "csv"], key="main_upload")
with col2:
    st.subheader("Redo Polygon File")
    redo_file = st.file_uploader("Upload Redo Polygon Form (CSV or Excel)",
                                 type=["xlsx","csv"], key="redo_upload")

# --- Process Main File Only If Provided ---
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

# --- Process Redo File (MANDATORY) ---
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

    # If the redo file uses a different column name for farmer code, rename it.
    if "farmer_code" in df_redo.columns:
        df_redo = df_redo.rename(columns={'farmer_code': 'Farmercode'})

    # Check for required columns in the redo file
    required_redo_cols = ['Farmercode', 'selectplot', 'polygonplot']
    if not all(col in df_redo.columns for col in required_redo_cols):
        st.error("Redo polygon file must contain 'Farmercode', 'selectplot', and 'polygonplot' columns.")
        st.stop()

    # --- Use SubmissionDate and endtime to select latest submission ---
    if 'SubmissionDate' in df_redo.columns and 'endtime' in df_redo.columns:
        df_redo['SubmissionDate'] = pd.to_datetime(df_redo['SubmissionDate'], errors='coerce')
        df_redo['endtime'] = pd.to_datetime(df_redo['endtime'], errors='coerce')
        df_redo = df_redo.sort_values(by=['SubmissionDate', 'endtime'])
        df_redo = df_redo.groupby('Farmercode', as_index=False).last()

    # Rename redo file columns to avoid conflict
    df_redo = df_redo.rename(columns={'selectplot': 'redo_selectplot',
                                      'polygonplot': 'redo_polygonplot'})

    # Merge the redo data with the main file on Farmercode
    df = df.merge(df_redo[['Farmercode', 'redo_selectplot', 'redo_polygonplot']],
                  on='Farmercode', how='left')

    # --- Conditions to replace polygons ---
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

    # Drop the redo columns after updating
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

# --- CREATE GEOMETRY BY COMBINING POLYGON COLUMNS ---
df['geometry'] = df.apply(combine_polygons, axis=1)
df = df[df['geometry'].notna()]

# Convert to GeoDataFrame (assumed CRS EPSG:4326)
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

# --- DISPLAY TARGET POLYGON AREA ---
target_row = gdf[gdf['Farmercode'] == selected_code]
if not target_row.empty:
    target_area_m2 = target_row.geometry.iloc[0].area
    target_area_acres = target_area_m2 * 0.000247105
    st.subheader("Target Polygon Area:")
    st.write(f"Total Area: {target_area_m2:.2f} m² ({target_area_acres:.4f} acres)")

# --- EXPORT UPDATED FORM AS EXCEL ---
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

# ----------------------------
# SINGLE BUTTON: CHECK INCONSISTENCIES
# ----------------------------
if st.button("Check Inconsistencies"):
    # Filter the main df to the selected code
    df_code = df[df['Farmercode'] == selected_code].copy()
    gdf_code = gdf[gdf['Farmercode'] == selected_code].copy()

    # Required columns
    required_inconsistency_cols = ['Farmercode', 'username', 'duration', 'Registered', 'Phone', 'Phone_hidden']
    missing_cols = [col for col in required_inconsistency_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing columns for inconsistency checks: {', '.join(missing_cols)}")
    else:
        st.subheader(f"Inconsistencies for Farmer Code: {selected_code}")

        # Check 1: Duration < 15 mins (900s) but Registered == 'Yes'
        time_inconsistency = df_code[(df_code['duration'] < 900) & (df_code['Registered'].str.lower()=='yes')]
        
        # Check 2: Phone != Phone_hidden
        phone_mismatch = df_code[df_code['Phone'] != df_code['Phone_hidden']]
        
        # Check 3: Duplicate Phone entries (only if multiple rows exist for the same phone & code)
        # For duplicates, we can check among all data but filter for rows that have this code
        # or just check duplicates within df_code. Usually, duplicates are identified in the entire dataset,
        # but you asked for per code. We'll do it within df_code to keep it simple.
        duplicate_phones = df_code[df_code.duplicated(subset=['Phone'], keep=False)]
        
        # Check 4: Duplicate Farmer codes (again, typically across entire dataset,
        # but we'll do it within df_code for this example).
        duplicate_farmercodes = df_code[df_code.duplicated(subset=['Farmercode'], keep=False)]
        
        # Display the results:
        with st.expander("Time Inconsistencies (Duration < 15 mins but Registered == Yes)"):
            if not time_inconsistency.empty:
                st.write(time_inconsistency[['Farmercode', 'username', 'duration', 'Registered']])
            else:
                st.write("No time inconsistencies found.")
                
        with st.expander("Phone Mismatches (Phone != Phone_hidden)"):
            if not phone_mismatch.empty:
                st.write(phone_mismatch[['Farmercode', 'username', 'Phone', 'Phone_hidden']])
            else:
                st.write("No phone mismatches found.")
                
        with st.expander("Duplicate Phone Entries"):
            if not duplicate_phones.empty:
                st.write(duplicate_phones[['Farmercode', 'username', 'Phone']])
            else:
                st.write("No duplicate phone entries found.")
                
        with st.expander("Duplicate Farmer Codes"):
            if not duplicate_farmercodes.empty:
                st.write(duplicate_farmercodes[['Farmercode', 'username']])
            else:
                st.write("No duplicate Farmer codes found.")

    # --- ADDITIONAL INCONSISTENCY CHECKS ---

    # 1) Productive Plants vs. Expected
    if 'Productiveplants' in df.columns:
        if not gdf_code.empty:
            # Calculate acreage for this code's geometry
            gdf_code['acres'] = gdf_code['geometry'].area * 0.000247105
            gdf_code['expected_plants'] = gdf_code['acres'] * 450
            # Convert the Productiveplants to numeric
            gdf_code['productiveplants'] = pd.to_numeric(gdf_code['Productiveplants'], errors='coerce')
            
            # Check if actual plants exceed expected
            plants_inconsistency = gdf_code[gdf_code['productiveplants'] > gdf_code['expected_plants']]
            
            with st.expander("Productive Plants Inconsistency (Plants exceed expected per acre)"):
                if not plants_inconsistency.empty:
                    st.write(plants_inconsistency[[
                        'Farmercode', 'productiveplants', 'expected_plants', 'acres'
                    ]])
                else:
                    st.write("No productive plants inconsistencies found.")
        else:
            st.info("No geometry found for this code. Skipping Productiveplants check.")
    else:
        st.info("Column 'Productiveplants' not found; skipping productive plants check.")
    
    # 2) Uganda Polygon Check
    # Hard-code the Uganda polygon (example polygon from your request)
    uganda_coords = [
            (30.471786, -1.066837), (30.460829, -1.063428), (30.445614, -1.058694),
            (30.432023, -1.060554), (30.418897, -1.066445), (30.403188, -1.070373),
            (30.386341, -1.068202), (30.369288, -1.063241), (30.352752, -1.060761),
            (30.337455, -1.066239), (30.329032, -1.080501), (30.322572, -1.121843),
            (30.317405, -1.137035), (30.311307, -1.1421), (30.294564, -1.149644),
            (30.29095, -1.152621), (30.287536, -1.155432), (30.284642, -1.161427),
            (30.28242, -1.175793), (30.280508, -1.182407), (30.269759, -1.200494),
            (30.256685, -1.217237), (30.212192, -1.259509), (30.19674, -1.268707),
            (30.189351, -1.270877), (30.181392, -1.271497), (30.173434, -1.272841),
            (30.165579, -1.277492), (30.158448, -1.291135), (30.15235, -1.329892),
            (30.147183, -1.345085), (30.136331, -1.355213), (30.095506, -1.37113),
            (30.065984, -1.386945), (30.06078, -1.389733), (30.047757, -1.403169),
            (30.038662, -1.424977), (30.028327, -1.427147), (29.960424, -1.464767),
            (29.938462, -1.472932), (29.917429, -1.475206), (29.897896, -1.469625),
            (29.880739, -1.453605), (29.871024, -1.432418), (29.868647, -1.391283),
            (29.864203, -1.370303), (29.836091, -1.329478), (29.825135, -1.323897),
            (29.825024, -1.323881), (29.816143, -1.322554), (29.807462, -1.325138),
            (29.798212, -1.330925), (29.789168, -1.341674), (29.783174, -1.361414),
            (29.774906, -1.366272), (29.767981, -1.363792), (29.74669, -1.350872),
            (29.734805, -1.348185), (29.710517, -1.352526), (29.693774, -1.361208),
            (29.678322, -1.37237), (29.657703, -1.383945), (29.6391, -1.38901),
            (29.618119, -1.39056), (29.577915, -1.38839), (29.58732, -1.329685),
            (29.587113, -1.310565), (29.583186, -1.299299), (29.571507, -1.279249),
            (29.57099, -1.268604), (29.580447, -1.243282), (29.581429, -1.234807),
            (29.575538, -1.213206), (29.565254, -1.19791), (29.557038, -1.181787),
            (29.556934, -1.157706), (29.56913, -1.095901), (29.57006, -1.077918),
            (29.565926, -1.058591), (29.551198, -1.020143), (29.54846, -1.002573),
            (29.551353, -0.990584), (29.551518, -0.990305), (29.556521, -0.981799),
            (29.560345, -0.972084), (29.559157, -0.957305), (29.555281, -0.938495),
            (29.554661, -0.928573), (29.556004, -0.919478), (29.567166, -0.901908),
            (29.596725, -0.891882), (29.607991, -0.878447), (29.610781, -0.863977),
            (29.613365, -0.803826), (29.611091, -0.782742), (29.602203, -0.743778),
            (29.60303, -0.7229), (29.615536, -0.644146), (29.618843, -0.638978),
            (29.624424, -0.634844), (29.629281, -0.62978), (29.630573, -0.622028),
            (29.628248, -0.616034), (29.620496, -0.605388), (29.618843, -0.599394),
            (29.622874, -0.588438), (29.632175, -0.585751), (29.642821, -0.585028),
            (29.650882, -0.57955), (29.653053, -0.565597), (29.649745, -0.504309),
            (29.645094, -0.48891), (29.634615, -0.466968), (29.631865, -0.461211),
            (29.629385, -0.442401), (29.650981, -0.316455), (29.653983, -0.298947),
            (29.670569, -0.200867), (29.676617, -0.165105), (29.694032, -0.063096),
            (29.709018, -0.026302), (29.714341, -0.007492), (29.713462, 0.011628),
            (29.701628, 0.055243), (29.703075, 0.072503), (29.711809, 0.099582),
            (29.755785, 0.16087), (29.761263, 0.172135), (29.773045, 0.167484),
            (29.780383, 0.16118), (29.787205, 0.158493), (29.797127, 0.164797),
            (29.800641, 0.172445), (29.832628, 0.336983), (29.839605, 0.358481),
            (29.851077, 0.377187), (29.903909, 0.438229), (29.922907, 0.46018),
            (29.926083, 0.467073), (29.940477, 0.498317), (29.937996, 0.537281),
            (29.919496, 0.618103), (29.920013, 0.63867), (29.932312, 0.723161),
            (29.926834, 0.774889), (29.928281, 0.785018), (29.947298, 0.824602),
            (29.960217, 0.832095), (29.982194, 0.849017), (29.996391, 0.859949),
            (30.038352, 0.878914), (30.145116, 0.90315), (30.154831, 0.90868),
            (30.165683, 0.921444), (30.18377, 0.955137), (30.18687, 0.958754),
            (30.191521, 0.974877), (30.214052, 0.998545), (30.220977, 1.017097),
            (30.215602, 1.057766), (30.215292, 1.077093), (30.228005, 1.08903),
            (30.231519, 1.097764), (30.234102, 1.108099), (30.236169, 1.129493),
            (30.238857, 1.136004), (30.269346, 1.167268), (30.277976, 1.171609),
            (30.286502, 1.174296), (30.295701, 1.172643), (30.306553, 1.163909),
            (30.323813, 1.155848), (30.33606, 1.16887), (30.348204, 1.189024),
            (30.364947, 1.202047), (30.376626, 1.20308), (30.39926, 1.2006),
            (30.412696, 1.202047), (30.43161, 1.207008), (30.445562, 1.212795),
            (30.458275, 1.221684), (30.478274, 1.238634), (30.553704, 1.335632),
            (30.597284, 1.391673), (30.681724, 1.500349), (30.817013, 1.609515),
            (30.95442, 1.720671), (31.025837, 1.778239), (31.096018, 1.866363),
            (31.119113, 1.895363), (31.183295, 1.976211), (31.242826, 2.051168),
            (31.271455, 2.102999), (31.280096, 2.151441), (31.280447, 2.15341),
            (31.27874, 2.156021), (31.267476, 2.173253), (31.21089, 2.205345),
            (31.190116, 2.221519), (31.182468, 2.238728), (31.179058, 2.25976),
            (31.177559, 2.30291), (31.129242, 2.284668), (31.112602, 2.282084),
            (31.099063, 2.282704), (31.055241, 2.290249), (31.040668, 2.297897),
            (31.035501, 2.30694), (31.038808, 2.310971), (31.044389, 2.313917),
            (31.045939, 2.319653), (31.043562, 2.327043), (31.041288, 2.331228),
            (30.984858, 2.394635), (30.968011, 2.405436), (30.930907, 2.405591),
            (30.914474, 2.378151), (30.900573, 2.345956), (30.87179, 2.332055),
            (30.854943, 2.339703), (30.83634, 2.356343), (30.819803, 2.37598),
            (30.809675, 2.392362), (30.806987, 2.406986), (30.807142, 2.422179),
            (30.80492, 2.434374), (30.794998, 2.440111), (30.724925, 2.440782),
            (30.710559, 2.445123), (30.707665, 2.46228), (30.71676, 2.483105),
            (30.729369, 2.503466), (30.737017, 2.519537), (30.737844, 2.537469),
            (30.73402, 2.574469), (30.73526, 2.593073), (30.739395, 2.603305),
            (30.758928, 2.633794), (30.761254, 2.641597), (30.762013, 2.645781),
            (30.764303, 2.658392), (30.797686, 2.74836), (30.798926, 2.753528),
            (30.799133, 2.76345), (30.801613, 2.769083), (30.80585, 2.771873),
            (30.817943, 2.774199), (30.82187, 2.776162), (30.828691, 2.786084),
            (30.853393, 2.853367), (30.85484, 2.89321), (30.843988, 2.932794),
            (30.82094, 2.973153), (30.803628, 2.989069), (30.757068, 3.02147),
            (30.745079, 3.036302), (30.743839, 3.055474), (30.74787, 3.076713),
            (30.763416, 3.123437), (30.804197, 3.246005), (30.822129, 3.281403),
            (30.825746, 3.283677), (30.837786, 3.286416), (30.842437, 3.288741),
            (30.845641, 3.293909), (30.846882, 3.304399), (30.84869, 3.309256),
            (30.868482, 3.343337), (30.897318, 3.375015), (30.904933, 3.386035),
            (30.91003, 3.393412), (30.916335, 3.414806), (30.914474, 3.426484),
            (30.904449, 3.447465), (30.902899, 3.458911), (30.909617, 3.487178),
            (30.909307, 3.496144), (30.896388, 3.519967), (30.880161, 3.514412),
            (30.861248, 3.498211), (30.839543, 3.490202), (30.843781, 3.505911),
            (30.865692, 3.548958), (30.931734, 3.645102), (30.936282, 3.656858),
            (30.939382, 3.668408), (30.944447, 3.679286), (30.955505, 3.689001),
            (30.965996, 3.692618), (30.985788, 3.692127), (30.995968, 3.693522),
            (31.009559, 3.699775), (31.040358, 3.724218), (31.04966, 3.727991),
            (31.068677, 3.731737), (31.077668, 3.735303), (31.104408, 3.756175),
            (31.141489, 3.785119), (31.167585, 3.792405), (31.214818, 3.792354),
            (31.255745, 3.786669), (31.29502, 3.774189), (31.377702, 3.729308),
            (31.505446, 3.659855), (31.523739, 3.656083), (31.53452, 3.665563),
            (31.535522, 3.666444), (31.547201, 3.680991), (31.564771, 3.689802),
            (31.668537, 3.70502), (31.68683, 3.712824), (31.696132, 3.721273),
            (31.775714, 3.810699), (31.777884, 3.816435), (31.780985, 3.815815),
            (31.801107, 3.806472), (31.80703, 3.803722), (31.830697, 3.783879),
            (31.901908, 3.704297), (31.916274, 3.680267), (31.920046, 3.661302),
            (31.923095, 3.615284), (31.93002, 3.59836), (31.943662, 3.591255),
            (32.022236, 3.58642), (32.030168, 3.585932), (32.041537, 3.57986),
            (32.054226, 3.559572), (32.060864, 3.548958), (32.076161, 3.53317),
            (32.093007, 3.524463), (32.155846, 3.511776), (32.16799, 3.512242),
            (32.174552, 3.520897), (32.175929, 3.527234), (32.178997, 3.541361),
            (32.179203, 3.556864), (32.174863, 3.59265), (32.175948, 3.605595),
            (32.187782, 3.61916), (32.371801, 3.731065), (32.415571, 3.741297),
            (32.599281, 3.756283), (32.756429, 3.769022), (32.840352, 3.794291),
            (32.9189, 3.83416), (32.979568, 3.879196), (32.997241, 3.885526),
            (33.01724, 3.87718), (33.143486, 3.774086), (33.164466, 3.763053),
            (33.19542, 3.757059), (33.286526, 3.752537), (33.447343, 3.744372),
            (33.490648, 3.749746), (33.527716, 3.771431), (33.532609, 3.774293),
            (33.606196, 3.848087), (33.701901, 3.944076), (33.813677, 4.056033),
            (33.896049, 4.138353), (33.977078, 4.219692), (34.006017, 4.205713),
            (34.028548, 4.188014), (34.041054, 4.164812), (34.03971, 4.134038),
            (34.04095, 4.120421), (34.049735, 4.109466), (34.060897, 4.09926),
            (34.069269, 4.08802), (34.072369, 4.076419), (34.072679, 4.064663),
            (34.069786, 4.041357), (34.065858, 4.02743), (34.061517, 4.017921),
            (34.061104, 4.007767), (34.068959, 3.991876), (34.080844, 3.980792),
            (34.09552, 3.970999), (34.107096, 3.959501), (34.109576, 3.943352),
            (34.098931, 3.917721), (34.086219, 3.894673), (34.084772, 3.877336),
            (34.108026, 3.868938), (34.123736, 3.872039), (34.163423, 3.886198),
            (34.182854, 3.886095), (34.204971, 3.874545), (34.207555, 3.860877),
            (34.196703, 3.847389), (34.148954, 3.822688), (34.150814, 3.81721),
            (34.166524, 3.811319), (34.17903, 3.7961), (34.159082, 3.783362),
            (34.152468, 3.775636), (34.16518, 3.77083), (34.173655, 3.771399),
            (34.190915, 3.775843), (34.210552, 3.777135), (34.230396, 3.782742),
            (34.240938, 3.78362), (34.241558, 3.778737), (34.263779, 3.750056),
            (34.278868, 3.709723), (34.290444, 3.70316), (34.295095, 3.707346),
            (34.299436, 3.716932), (34.309874, 3.726699), (34.337056, 3.734579),
            (34.35504, 3.727371), (34.387492, 3.692747), (34.406303, 3.682929),
            (34.425009, 3.677193), (34.439686, 3.667736), (34.446403, 3.646704),
            (34.443923, 3.566295), (34.434518, 3.52622), (34.415501, 3.497023),
            (34.406199, 3.492294), (34.395967, 3.489607), (34.386872, 3.485628),
            (34.381291, 3.476869), (34.382118, 3.466043), (34.394417, 3.444623),
            (34.398551, 3.433642), (34.397001, 3.424159), (34.386976, 3.399664),
            (34.383978, 3.388347), (34.386562, 3.376384), (34.394417, 3.365661),
            (34.403822, 3.355404), (34.41116, 3.344707), (34.424079, 3.304812),
            (34.434001, 3.182029), (34.444853, 3.159136), (34.46573, 3.145856),
            (34.512859, 3.132368), (34.53384, 3.118467), (34.545556, 3.097501),
            (34.545622, 3.097383), (34.574664, 2.946126), (34.58469, 2.928711),
            (34.616419, 2.893416), (34.631405, 2.869542), (34.6405, 2.860137),
            (34.654143, 2.856519), (34.660344, 2.858638), (34.671506, 2.867785),
            (34.67564, 2.8698), (34.678121, 2.871557), (34.682565, 2.87936),
            (34.685045, 2.880962), (34.688869, 2.879154), (34.69104, 2.875691),
            (34.692693, 2.872229), (34.694657, 2.87042), (34.696517, 2.867733),
            (34.724836, 2.854142), (34.730107, 2.85285), (34.736928, 2.844375),
            (34.740959, 2.835539), (34.76132, 2.772493), (34.776512, 2.685625),
            (34.81868, 2.597982), (34.828086, 2.588784), (34.841832, 2.58775),
            (34.849893, 2.59514), (34.856508, 2.60346), (34.865603, 2.604855),
            (34.875628, 2.591212), (34.881312, 2.541448), (34.8871, 2.522379),
            (34.914385, 2.494371), (34.923584, 2.477318), (34.920896, 2.454632),
            (34.906117, 2.436907), (34.885963, 2.425434), (34.867876, 2.411482),
            (34.859091, 2.38678), (34.865603, 2.347506), (34.904284, 2.254255),
            (34.922343, 2.210719), (34.967509, 2.101914), (34.967612, 2.082561),
            (34.958207, 2.037965), (34.956657, 2.018508), (34.957897, 1.997838),
            (34.962031, 1.977813), (34.969162, 1.960295), (34.977947, 1.949908),
            (35.001408, 1.927997), (35.006473, 1.916861), (35.002752, 1.906138),
            (34.984045, 1.88247), (34.979704, 1.8703), (34.978671, 1.675945),
            (34.972676, 1.654241), (34.940947, 1.587036), (34.933712, 1.575693),
            (34.92348, 1.566107), (34.913558, 1.56156), (34.892578, 1.55727),
            (34.882553, 1.552232), (34.859505, 1.517919), (34.838421, 1.437174),
            (34.778993, 1.388547), (34.780747, 1.371789), (34.782714, 1.352993),
            (34.800077, 1.312324), (34.810516, 1.272533), (34.797907, 1.231916),
            (34.766074, 1.217498), (34.683805, 1.209075), (34.663031, 1.196362),
            (34.628925, 1.163599), (34.580452, 1.152541), (34.573631, 1.134247),
            (34.571977, 1.111716), (34.566883, 1.103022), (34.561228, 1.093371),
            (34.560065, 1.093149), (34.540351, 1.089392), (34.524021, 1.098642),
            (34.507485, 1.102518), (34.487021, 1.082467), (34.477203, 1.064587),
            (34.468831, 1.044227), (34.463663, 1.022936), (34.463043, 0.978081),
            (34.457152, 0.957617), (34.431211, 0.893538), (34.402375, 0.856073),
            (34.388009, 0.815817), (34.370852, 0.800211), (34.30667, 0.768068),
            (34.298402, 0.759283), (34.296252, 0.746746), (34.292304, 0.72373),
            (34.276181, 0.680838), (34.2521, 0.655), (34.219751, 0.638567),
            (34.179133, 0.623891), (34.149574, 0.603634), (34.132417, 0.572834),
            (34.131588, 0.569433), (34.107509, 0.470722), (34.104409, 0.462247),
            (34.097381, 0.451808), (34.080121, 0.431344), (34.075677, 0.422249),
            (34.07671, 0.403852), (34.087976, 0.367369), (34.085185, 0.347008),
            (34.076607, 0.333779), (34.04064, 0.305874), (33.960266, 0.198677),
            (33.951757, 0.187328), (33.893569, 0.109814), (33.890468, 0.090073),
            (33.921578, -0.01297), (33.95248, -0.115702), (33.953514, -0.154356),
            (33.935117, -0.313106), (33.911346, -0.519295), (33.894809, -0.662749),
            (33.89853, -0.799072), (33.904214, -1.002573), (33.822255, -1.002573)
        ]
    uganda_poly = Polygon(uganda_coords)
    
    if not gdf_code.empty:
        uganda_inconsistency = gdf_code[gdf_code['geometry'].within(uganda_poly)]
        with st.expander("Uganda Polygon Inconsistency"):
            if not uganda_inconsistency.empty:
                st.write("This code's plot lies within the specified Uganda boundary:")
                st.write(uganda_inconsistency[['Farmercode', 'geometry']])
            else:
                st.write("No Uganda polygon inconsistencies found for this code.")
    else:
        st.info("No geometry found for this code. Skipping Uganda polygon check.")
