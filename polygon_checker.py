import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
import io
import re
import numpy as np

###############################################################################
# ⚡  Latitude Inspections Inconsistency Checker — Optimised Edition          #
#                                                                             #
#  This rewrite focuses on raw‑speed without changing functional behaviour.    #
#  Key techniques                                                            
#   • Disk I/O & heavy transforms cached (`st.cache_data` / `st.cache_resource`)
#   • Vectorised numeric work (NumPy)                                          
#   • One‑time construction of spatial index + STRtree                         
#   • Batched polygon parsing                                                  
#   • Dropped repeated passes / groupbys                                       
###############################################################################

st.set_page_config(page_title="Latitude Inspections Inconsistency Checker", layout="wide")

st.title("Latitude Inspections Inconsistency Checker ⚡")

###############################################################################
# 1 ▸ FILE UPLOAD + CACHED LOADERS                                            #
###############################################################################
col1, col2 = st.columns(2)
with col1:
    st.subheader("Main Inspection File")
    main_file = st.file_uploader("Upload Main Inspection Form (CSV or Excel)", type=["xlsx", "csv"], key="main_upload")
with col2:
    st.subheader("Redo Polygon File")
    redo_file = st.file_uploader("Upload Redo Polygon Form (CSV or Excel)", type=["xlsx", "csv"], key="redo_upload")

if main_file is None or redo_file is None:
    st.info("Please upload both Main and Redo files to continue.")
    st.stop()

@st.cache_data(show_spinner=False)
def _read_tabular(file):
    """Load CSV/XLSX with the fastest backend available and standardise cols."""
    try:
        if file.name.endswith(".xlsx"):
            df = pd.read_excel(file, engine="openpyxl")  # openpyxl fastest for xlsx in-memory
        else:
            # pyarrow>3 is ~5‑10× faster than default Python engine, fallback graceful
            try:
                df = pd.read_csv(file, engine="pyarrow")
            except ValueError:
                df = pd.read_csv(file, low_memory=False)
    except Exception as exc:
        st.error(f"Error loading {file.name}: {exc}")
        st.stop()
    df.columns = df.columns.str.strip()  # cheap normalisation
    return df

main_df = _read_tabular(main_file)
redo_df = _read_tabular(redo_file)

###############################################################################
# 2 ▸ BASIC VALIDATION / NORMALISATION                                        #
###############################################################################
required_main = {"Farmercode", "polygonplot"}
required_redo = {"Farmercode", "selectplot", "polygonplot"}

if not required_main.issubset(main_df.columns):
    st.error("Main file must contain Farmercode & polygonplot columns.")
    st.stop()
if not required_redo.issubset(redo_df.columns):
    st.error("Redo file must contain Farmercode, selectplot & polygonplot columns.")
    st.stop()

redo_df = redo_df.rename(columns={"farmer_code": "Farmercode"})  # tolerant rename

###############################################################################
# 3 ▸ FAST REDO MERGE (keep latest per Farmer)                                #
###############################################################################
if {"SubmissionDate", "endtime"}.issubset(redo_df.columns):
    redo_df["SubmissionDate"] = pd.to_datetime(redo_df["SubmissionDate"], errors="coerce")
    redo_df["endtime"]         = pd.to_datetime(redo_df["endtime"], errors="coerce")
    # sort once, then drop_duplicates keeps last (vectorised, faster than groupby)
    redo_df = redo_df.sort_values(["Farmercode", "SubmissionDate", "endtime"]).drop_duplicates("Farmercode", keep="last")
    redo_df = redo_df.rename(columns={"selectplot": "redo_selectplot", "polygonplot": "redo_polygonplot"})
    main_df = main_df.merge(redo_df[["Farmercode", "redo_selectplot", "redo_polygonplot"]], on="Farmercode", how="left")

    # Vectorised update per redo_selectplot
    plot_cols = ["polygonplot", "polygonplotnew_1", "polygonplotnew_2", "polygonplotnew_3", "polygonplotnew_4"]
    plot_lookup = {"Plot1": "polygonplot",
                   "Plot2": "polygonplotnew_1",
                   "Plot3": "polygonplotnew_2",
                   "Plot4": "polygonplotnew_3",
                   "Plot5": "polygonplotnew_4"}
    for sel_plot, col in plot_lookup.items():
        mask = (main_df["redo_selectplot"] == sel_plot) & main_df[col].notna()
        main_df.loc[mask, col] = main_df.loc[mask, "redo_polygonplot"].astype(str)

    main_df = main_df.drop(columns=["redo_selectplot", "redo_polygonplot"])  # tidy

###############################################################################
# 4 ▸ DATE SLIDER (vectorised)                                                #
###############################################################################
sub_date_col = "Submissiondate"
if "Submissiondate" in main_df.columns:
    main_df[sub_date_col] = pd.to_datetime(main_df["Submissiondate"], errors="coerce")
elif "SubmissionDate" in main_df.columns:
    main_df[sub_date_col] = pd.to_datetime(main_df["SubmissionDate"], errors="coerce")

if main_df[sub_date_col].notna().any():
    dmin, dmax = main_df[sub_date_col].min().date(), main_df[sub_date_col].max().date()
    dsel = st.slider("Select Submission Date Range", dmin, dmax, (dmin, dmax))
    main_df = main_df[(main_df[sub_date_col].dt.date >= dsel[0]) & (main_df[sub_date_col].dt.date <= dsel[1])]
else:
    st.warning("Submission date not available – skipping date filter.")

###############################################################################
# 5 ▸ POLYGON PARSE + GDF BUILD (cached)                                      #
###############################################################################
POLY_COLS = [c for c in main_df.columns if c.startswith("polygonplot")]

@st.cache_resource(show_spinner=True)
def _build_gdf(df: pd.DataFrame):
    """Parse all polygons once & construct spatial index."""

    def _str_to_poly(poly_str: str | float):
        if not isinstance(poly_str, str):
            return None
        # split quickly with regex – up to 2× faster than Python loops for long strings
        pts = np.fromstring(re.sub(r"[;,]", " ", poly_str), sep=" ")
        # expect triples (x y z), we want pairs – reshape then discard z
        if pts.size < 6 or pts.size % 3:
            return None
        xy = pts.reshape(-1, 3)[:, :2]
        try:
            return Polygon(xy)
        except ValueError:
            return None

    # vectorised parse into list-of-polygons columns
    geom_series = []
    for _, row in df[POLY_COLS].iterrows():
        polys = [ _str_to_poly(row[col]) for col in POLY_COLS if pd.notna(row[col]) ]
        # clean invalids
        polys = [ p.buffer(0) for p in polys if p is not None and not p.is_valid ] or polys
        if not polys:
            geom_series.append(None)
        else:
            geom_series.append(polys[0] if len(polys) == 1 else unary_union(polys))

    gdf = gpd.GeoDataFrame(df.copy(), geometry=geom_series, crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notna()].to_crs("EPSG:2109")
    gdf["geometry"] = gdf["geometry"].buffer(0)  # clean self‑intersections
    # STRtree (shapely >=2) ~4‑5× faster than geopandas R‑tree for many queries
    str_tree = STRtree(gdf.geometry)
    return gdf, str_tree

GDF, STR_TREE = _build_gdf(main_df)

###############################################################################
# 6 ▸ FAST OVERLAP LOOKUP                                                     #
###############################################################################
@st.cache_resource(show_spinner=False)
def _precompute_overlaps(gdf: gpd.GeoDataFrame, tree: STRtree):
    """Return dict Farmercode → (list[dict], pct_total_overlap)."""
    overlaps_dict = {}
    # map geometry to index for reverse lookup
    geom_to_idx = {geom: idx for idx, geom in enumerate(gdf.geometry.values)}
    for idx, row in gdf.iterrows():
        farmer = row["Farmercode"]
        target_geom = row.geometry
        total_area = target_geom.area
        # candidate indices via STRtree
        candidates = [geom_to_idx[g] for g in tree.query(target_geom) if geom_to_idx[g] != idx]
        union_int = None
        overlaps = []
        for c_idx in candidates:
            other = gdf.iloc[c_idx]
            inter = target_geom.intersection(other.geometry)
            if inter.is_empty:
                continue
            area = inter.area
            overlaps.append({
                "Farmercode": other["Farmercode"],
                "overlap_area": area,
                "total_area": total_area,
                "intersection": inter,
            })
            union_int = inter if union_int is None else union_int.union(inter)
        pct = (union_int.area / total_area * 100) if union_int else 0
        overlaps_dict[farmer] = (overlaps, pct)
    return overlaps_dict

OVERLAPS_CACHE = _precompute_overlaps(GDF, STR_TREE)

def check_overlaps(farmer_code: str):
    return OVERLAPS_CACHE.get(farmer_code, ([], 0))

###############################################################################
# 7 ▸ PRODUCTIVE PLANTS + INCONSISTENCY FLAGS (vectorised)                    #
###############################################################################
PLANT_AREA_RE = re.compile(r"acres_polygonplot", re.I)
PRODUCTIVE_RE = re.compile(r"productiveplants", re.I)


def _productive_metrics(df: pd.DataFrame):
    area_cols = [c for c in df.columns if PLANT_AREA_RE.search(c)]
    prod_cols = [c for c in df.columns if PRODUCTIVE_RE.search(c)]
    area_total = df[area_cols].fillna(0).astype(float).sum(axis=1)
    plants_total = df[prod_cols].fillna(0).astype(float).sum(axis=1)
    expected = area_total * 450
    return pd.DataFrame({
        "Total_Area": area_total,
        "Total_Productive_Plants": plants_total,
        "Expected_Plants": expected,
        "Half_Expected_Plants": expected / 2,
        "Pct125_Expected_Plants": expected * 1.25,
        "Productive_Plants_Inconsistency": np.where(plants_total < expected / 2, "Less than Expected Productive Plants", np.where(plants_total > expected * 1.25, "More than expected Productive Plants", ""))
    }, index=df.index)

PROD_METRICS = _productive_metrics(main_df)

###############################################################################
# 8 ▸ FLAG CHECKS (vectorised where cheap)                                    #
###############################################################################

def _flag_yes(df_col, *yes_values):
    yes_set = set(v.lower() for v in yes_values)
    return df_col.astype(str).str.strip().str.lower().isin(yes_set)

LABOUR_FLAG = (
    _flag_yes(main_df["childrenworkingconditions"], "any_time_when_needed") |
    _flag_yes(main_df["prisoners"], "yes") |
    _flag_yes(main_df["contractsworkers"], "no") |
    _flag_yes(main_df["drinkingwaterworkers"], "no")
)

LABOUR_MISMATCH = LABOUR_FLAG & (main_df["noncompliancesfound_Labour"].fillna(0) == 0)

ENV_FLAG = (
    _flag_yes(main_df["cutnativetrees"], "yes") |
    _flag_yes(main_df["cutforests"], "yes") |
    _flag_yes(main_df["toiletdischarge"], "yes") |
    _flag_yes(main_df["separatewaste"], "no")
)
ENV_MISMATCH = ENV_FLAG & (main_df["noncompliancesfound_Environmental"].fillna(0) == 0)

# agrochemical columns numeric 1‑check (vectorised)
AGRO_COLS = ["methodspestdiseasemanagement_using_chemicals", "fertilizerchemicals_Pesticides", "fertilizerchemicals_Fungicides", "fertilizerchemicals_Herbicides", "childrenlabouractivities_spraying_of_chemicals", "typeworkvulnerable_Spraying_of_chemicals", "agriculturalinputs_synthetic_chemicals_or_fertilize"]
AGRO_FLAG = main_df[AGRO_COLS].fillna(0).astype(float).eq(1).any(axis=1)
AGRO_MISMATCH = AGRO_FLAG & (main_df["noncompliancesfound_Agro_chemical"].fillna(0) == 0)

AGRO_NO = _flag_yes(main_df["pruning"], "no") | _flag_yes(main_df["desuckering"], "no") | _flag_yes(main_df["manageweeds"], "no") | _flag_yes(main_df["knowledgeIPM"], "no")
AGRONOMIC_MISMATCH = AGRO_NO & (main_df["noncompliancesfound_Agronomic"].fillna(0) == 0)

POST_NO = _flag_yes(main_df["ripepods"], "no") | _flag_yes(main_df["storedrycocoa"], "no") | _flag_yes(main_df["separatebasins"], "no")
POST_MISMATCH = POST_NO & (main_df["noncompliancesfound_Harvest_and_postharvestt"].fillna(0) == 0)

PHONE_MISMATCH = ~_flag_yes(main_df["phone_match"], "match")
TIME_MISMATCH  = main_df["duration"].fillna(0).astype(float) < 900

# Build inconsistencies dataframe quickly
ALL_FLAGS = pd.DataFrame({
    "Farmercode": main_df["Farmercode"],
    "username":   main_df.get("username", ""),
    "Labour": LABOUR_MISMATCH,
    "Environmental": ENV_MISMATCH,
    "Agrochemical": AGRO_MISMATCH,
    "Agronomic": AGRONOMIC_MISMATCH,
    "PostHarvest": POST_MISMATCH,
    "Phone": PHONE_MISMATCH,
    "Time": TIME_MISMATCH,
})

FLAG_MAP = {
    "Labour": "Labour-Noncompliance-Mismatch",
    "Environmental": "Environmental-Noncompliance-Mismatch",
    "Agrochemical": "Agrochemical-Noncompliance-Mismatch",
    "Agronomic": "Agronomic-Noncompliance-Mismatch",
    "PostHarvest": "PostHarvest-Noncompliance-Mismatch",
    "Phone": "Phone number mismatch",
    "Time": "Time inconsistency: Inspection < 15 min",
}

incons_long = ALL_FLAGS.melt(id_vars=["Farmercode", "username"], var_name="flag", value_name="val")
INCONSISTENCIES_DF = inconsist_long = incons_long[incons_long["val"]].drop("val", axis=1)
INCONSISTENCIES_DF["inconsistency"] = INCONSISTENCIES_DF["flag"].map(FLAG_MAP)

# overlap inconsistencies from cache
olap_rows = []
for farmer, (_, pct) in OVERLAPS_CACHE.items():
    if pct >= 5:
        row = main_df.loc[main_df["Farmercode"] == farmer].iloc[0]
        text = "Overlap > 10%" if pct > 10 else "Overlap 5‑10%"
        olap_rows.append({"Farmercode": farmer, "username": row.get("username", ""), "inconsistency": text})

INCONSISTENCIES_DF = pd.concat([INCONSISTENCIES_DF, pd.DataFrame(olap_rows)], ignore_index=True)

###############################################################################
# 9 ▸ RISK RATING (vectorised)                                                #
###############################################################################
risk_order = {"High": 3, "Medium": 2, "Low": 1, "None": 0}

INCONSISTENCIES_DF["Risk Rating"] = np.select([
    INCONSISTENCIES_DF["inconsistency"].str.contains("noncompliance-mismatch|less than expected|more than expected|time inconsistency", case=False, regex=True),
    INCONSISTENCIES_DF["inconsistency"].str.contains("overlap >", case=False),
    INCONSISTENCIES_DF["inconsistency"].str.contains("overlap 5", case=False) | INCONSISTENCIES_DF["inconsistency"].str.contains("phone", case=False),
], ["High", "Medium", "Low"], default="Low")

agg_incons = INCONSISTENCIES_DF.groupby(["Farmercode", "username"], as_index=False).agg({
    "inconsistency": lambda s: ", ".join(sorted(s.unique())),
    "Risk Rating": lambda s: max(s, key=lambda k: risk_order[k])
})
agg_incons["Trust Responses"] = np.where(agg_incons["Risk Rating"] == "High", "No", "Yes")

###############################################################################
# 10 ▸ RATING SCORE (apply‑row still okay; cached)                            #
###############################################################################
@st.cache_resource(show_spinner=False)
def _compute_row_scores(df: pd.DataFrame) -> pd.Series:
    return df.apply(_row_score, axis=1)

def _row_score(row):
    score = 0
    if str(row.get("phone_match", "")).strip().lower() == "match":
        score += 1
    if row.get("duration", 0) > 900:
        score += 1
    if (pd.notnull(row.get("kgsold")) and pd.notnull(row.get("harvestflyseason")) and pd.notnull(row.get("totalharvest")) and row["kgsold"] <= (row["harvestflyseason"] + row["totalharvest"])):
        score += 1
    if (pd.notnull(row.get("acreagetotalplot")) and pd.notnull(row.get("cocoaacreage")) and row["acreagetotalplot"] < row["cocoaacreage"]):
        score += 1
    if pd.notnull(row.get("Productiveplants")):
        if row["Productiveplants"] >= (row.get("youngplants", 0) + row.get("stumpedplants", 0) + row.get("shadeplants", 0)):
            score += 1
    # labour
    if not LABOUR_MISMATCH.loc[row.name]:
        score += 1
    if not ENV_MISMATCH.loc[row.name]:
        score += 1
    if not AGRO_MISMATCH.loc[row.name]:
        score += 1
    if not AGRONOMIC_MISMATCH.loc[row.name]:
        score += 1
    if not POST_MISMATCH.loc[row.name]:
        score += 1
    # productive plants consistency
    if PROD_METRICS.loc[row.name, "Productive_Plants_Inconsistency"] == "":
        score += 1
    # overlap <5%
    _, pct = OVERLAPS_CACHE[row["Farmercode"]]
    if pct < 5:
        score += 1
    return score

main_df["total_rating"] = _compute_row_scores(main_df)

inspector_rating = main_df.groupby("username")["total_rating"].mean().reset_index()

###############################################################################
# 11 ▸ UI – TOP INSPECTORS                                                    #
###############################################################################
if "district" in main_df.columns:
    districts = sorted(main_df["district"].dropna().unique())
    sel_district = st.selectbox("Filter by District", ["All"] + districts)
    if sel_district != "All":
        filter_mask = main_df["district"] == sel_district
        inspector_rating = main_df[filter_mask].groupby("username")["total_rating"].mean().reset_index()

st.subheader("Best Inspectors by Average Rating")
if not inspector_rating.empty:
    best = inspector_rating.sort_values("total_rating", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(best["username"], best["total_rating"])
    ax.set_xlabel("Inspector")
    ax.set_ylabel("Average Rating")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)
else:
    st.write("No rating data available.")

###############################################################################
# 12 ▸ UI – INCONSISTENCIES FOR ONE FARMER                                    #
###############################################################################
sel_farmer = st.selectbox("Select Farmer Code", GDF["Farmercode"].unique())
sel_incon = agg_incons[agg_incons["Farmercode"] == sel_farmer]
st.subheader(f"Inconsistencies for Farmer {sel_farmer}")
if not sel_incon.empty:
    st.dataframe(sel_incon, hide_index=True)
else:
    st.success("No inconsistencies for this farmer.")

# Polygon + overlap map
row = GDF[GDF["Farmercode"] == sel_farmer].iloc[0]
area_m2 = row.geometry.area
st.write(f"**Union Area:** {area_m2:,.1f} m²")

ovlps, pct_total = check_overlaps(sel_farmer)
st.write(f"**Total Overlap:** {pct_total:.2f}%")

if ovlps:
    fig_map, ax_map = plt.subplots(figsize=(8, 8))
    # target polygon blue
    if row.geometry.geom_type == "MultiPolygon":
        for p in row.geometry.geoms:
            x, y = p.exterior.xy
            ax_map.fill(x, y, alpha=0.4, fc="blue", ec="black")
    else:
        x, y = row.geometry.exterior.xy
        ax_map.fill(x, y, alpha=0.4, fc="blue", ec="black")

    for ov in ovlps:
        inter = ov["intersection"]
        if inter.geom_type == "MultiPolygon":
            for p in inter.geoms:
                x, y = p.exterior.xy
                ax_map.fill(x, y, alpha=0.5, fc="red", ec="darkred")
        else:
            x, y = inter.exterior.xy
            ax_map.fill(x, y, alpha=0.5, fc="red", ec="darkred")
    ax_map.set_title(f"Overlap Map – {sel_farmer}")
    ax_map.set_xlabel("Easting")
    ax_map.set_ylabel("Northing")
    st.pyplot(fig_map)
else:
    st.success("No overlaps with other farmers.")

###############################################################################
# 13 ▸ EXPORT                                                                 #
###############################################################################

def _prepare_export(df_base: pd.DataFrame):
    df_out = df_base.copy()
    df_out = pd.concat([df_out, PROD_METRICS], axis=1)
    df_out["Acres"] = gpd.GeoSeries(df_out["geometry"], crs=GDF.crs).area * 0.000247105
    df_out["geometry"] = df_out["geometry"].apply(lambda g: g.wkt if g is not None else "")
    df_out = df_out.merge(agg_incons, on=["Farmercode", "username"], how="left")
    df_out[["inconsistency", "Risk Rating", "Trust Responses"]] = df_out[["inconsistency", "Risk Rating", "Trust Responses"]].fillna({
        "inconsistency": "No Inconsistency", "Risk Rating": "None", "Trust Responses": "Yes"})
    df_out["total_rating"] = _compute_row_scores(df_out)
    avg_rating = df_out.groupby("username")["total_rating"].mean().rename("average_rating_per_username").reset_index()
    return df_out.merge(avg_rating, on="username", how="left")

if st.button("Export Updated Form to Excel"):
    exp = _prepare_export(main_df)
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        exp.to_excel(writer, index=False, sheet_name="Updated Form")
    bio.seek(0)
    st.download_button("Download Updated Form with Ratings", data=bio, file_name="updated_inspection_form_merged.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
