import re
from io import BytesIO
from difflib import get_close_matches

import numpy as np
import pandas as pd
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

# =============================
# CONFIG
# =============================

SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQkb6qWg9Jk_eKnf3DtvajFfJROa4v7_m6muP5ZP_MgWy85dn4zSsjtZlG9yEhXZFzw_U5VHY8miSzH/"
    "pub?gid=1998354188&single=true&output=csv"
)

st.set_page_config(
    page_title="MLB Strikeout Decision Engine",
    layout="wide"
)

st.title("MLB Strikeout Evidence Engine")
st.caption("Paste pregame read + PP lines. App generates a full evidence PDF for every PP-line pitcher.")


# =============================
# HELPERS
# =============================

def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:.1%}"


def pc_tier(pc):
    if pd.isna(pc):
        return ""
    if pc <= 78:
        return "≤78 PC"
    elif pc <= 86:
        return "79-86 PC"
    elif pc <= 92:
        return "87-92 PC"
    elif pc <= 97:
        return "93-97 PC"
    else:
        return "98+ PC"


def mlk_tier(mlk):
    if pd.isna(mlk):
        return ""
    if mlk <= 3:
        return "Low MLK 2-3"
    elif mlk <= 5:
        return "Mid MLK 4-5"
    elif mlk <= 6:
        return "High MLK 6"
    else:
        return "Elite MLK 7+"


def fmt_num(x, digits=2):
    if pd.isna(x):
        return "N/A"
    return round(float(x), digits)


def safe_list(values):
    if values is None or len(values) == 0:
        return ""
    return " / ".join(str(x) for x in values)


# =============================
# DATA LOAD
# =============================

@st.cache_data(ttl=60)
def load_dataset():
    df = pd.read_csv(SHEET_URL)
    df.columns = df.columns.str.strip()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = [
        "Most Likely Ks", "PP Line", "Proj PC", "Proj IP", "Proj BF",
        "Actual Ks", "Actual IP", "Actual PC", "PC Error", "IP Error",
        "Proj Error", "L3 Avg PC", "90+ PC Rate", "Delta", "Edge_Result",
        "BF Estimate", "K/IP", "K Conversion", "Over Hit", "Under Hit", "Push Hit"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Pitcher_norm"] = df["Pitcher"].apply(norm_text)
    df["Opponent_norm"] = df["Normalized Opponent"].apply(
        lambda x: str(x).strip().upper() if pd.notna(x) else ""
    )
    df["Team_norm"] = df["Normalized Pitcher Team"].apply(
        lambda x: str(x).strip().upper() if pd.notna(x) else ""
    )

    return df


# =============================
# PARSERS
# =============================

def parse_pregame_rows(text):
    rows = []

    pattern = r"(20\d{2}-\d{2}-\d{2})\s+(.+?)(?=\s+20\d{2}-\d{2}-\d{2}|$)"
    matches = re.findall(pattern, text.strip(), flags=re.DOTALL)

    for date, rest in matches:
        tokens = rest.strip().split()

        try:
            venue = tokens[-1]
            opponent = tokens[-2]
            pitcher_team = tokens[-3]
            proj_bf = float(tokens[-4])
            proj_ip = float(tokens[-5])
            proj_pc = float(tokens[-6])
            leash = tokens[-7]
            prob = tokens[-8]
            mlk = float(tokens[-9])

            front = tokens[:-9]

            matchup_idx = None
            for i in range(len(front) - 2):
                if front[i + 1] in ["@", "vs"]:
                    matchup_idx = i
                    break

            if matchup_idx is None:
                pitcher = " ".join(front)
                matchup = ""
            else:
                pitcher = " ".join(front[:matchup_idx])
                matchup = " ".join(front[matchup_idx:matchup_idx + 3])

            rows.append({
                "Date": date,
                "Pitcher_Input": pitcher,
                "Pitcher_norm": norm_text(pitcher),
                "Matchup": matchup,
                "MLK": mlk,
                "Prob": prob,
                "Leash": leash,
                "Proj PC": proj_pc,
                "Proj IP": proj_ip,
                "Proj BF": proj_bf,
                "Pitcher Team": pitcher_team,
                "Pitcher Team Norm": str(pitcher_team).strip().upper(),
                "Opponent": opponent,
                "Opponent Norm": str(opponent).strip().upper(),
                "Venue": venue,
                "Proj PC Tier": pc_tier(proj_pc),
                "MLK Tier": mlk_tier(mlk),
            })

        except Exception as e:
            rows.append({
                "Date": date,
                "Pitcher_Input": "PARSE_ERROR",
                "Raw": rest,
                "Error": str(e)
            })

    return pd.DataFrame(rows)


def parse_pp_lines(text):
    items = []

    normal_lines = [x.strip() for x in text.strip().splitlines() if x.strip()]

    if len(normal_lines) == 1:
        pattern = r"([A-Za-zÀ-ÿ.'\- ]+?)\s+(-?\d+(?:\.\d+)?)"
        matches = re.findall(pattern, normal_lines[0])

        for name, line in matches:
            items.append({
                "Input Name": name.strip(),
                "Input Norm": norm_text(name),
                "PP Line": float(line),
            })

    else:
        for raw in normal_lines:
            m = re.search(r"(.+?)\s+(-?\d+(?:\.\d+)?)\s*$", raw)
            if not m:
                continue

            name = m.group(1).strip()
            line = float(m.group(2))

            items.append({
                "Input Name": name,
                "Input Norm": norm_text(name),
                "PP Line": line,
            })

    return pd.DataFrame(items)


# =============================
# MATCHING
# =============================

def match_to_pregame(input_name, pregame_df):
    """
    Match PP input to TODAY'S pregame pitcher pool first.
    Safer than matching directly to historical dataset.
    """

    raw = str(input_name).strip()
    raw_norm = norm_text(raw)

    if pregame_df.empty:
        return None, "no_pregame_rows"

    names = pregame_df["Pitcher_Input"].dropna().unique().tolist()

    exact = [n for n in names if norm_text(n) == raw_norm]
    if len(exact) == 1:
        return exact[0], "pregame_exact"

    last_matches = []
    for n in names:
        parts = str(n).strip().split()
        if parts and norm_text(parts[-1]) == raw_norm:
            last_matches.append(n)

    if len(last_matches) == 1:
        return last_matches[0], "pregame_last_name"
    elif len(last_matches) > 1:
        return last_matches, "manual_review_multiple_pregame_last"

    contains = [n for n in names if raw_norm in norm_text(n)]
    if len(contains) == 1:
        return contains[0], "pregame_contains"
    elif len(contains) > 1:
        return contains, "manual_review_multiple_pregame_contains"

    choices_norm = {norm_text(n): n for n in names}
    close = get_close_matches(raw_norm, list(choices_norm.keys()), n=5, cutoff=0.65)

    if len(close) > 0:
        return [choices_norm[c] for c in close], "manual_review_pregame_fuzzy"

    return None, "no_pregame_match"


def build_board(pregame_df, pp_df):
    board_rows = []

    for _, pp in pp_df.iterrows():
        input_name = pp["Input Name"]
        line = pp["PP Line"]

        matched, method = match_to_pregame(input_name, pregame_df)

        if isinstance(matched, list) or matched is None:
            board_rows.append({
                "Input Name": input_name,
                "Matched Pitcher": matched,
                "Match Status": method,
                "PP Line": line,
            })
            continue

        row = pregame_df[pregame_df["Pitcher_Input"].eq(matched)]

        if row.empty:
            board_rows.append({
                "Input Name": input_name,
                "Matched Pitcher": matched,
                "Match Status": "matched_but_missing_pregame_row",
                "PP Line": line,
            })
            continue

        r = row.iloc[0].to_dict()
        r["Input Name"] = input_name
        r["Matched Pitcher"] = matched
        r["Match Status"] = "ok"
        r["Match Method"] = method
        r["PP Line"] = line
        r["Delta"] = r["MLK"] - line

        board_rows.append(r)

    return pd.DataFrame(board_rows)


# =============================
# ANALYSIS
# =============================

def rate_summary(sub, line):
    if sub.empty:
        return {
            "n": 0,
            "avg_ks": np.nan,
            "median_ks": np.nan,
            "over_rate": np.nan,
            "under_rate": np.nan,
            "push_rate": np.nan,
            "avg_pc": np.nan,
            "avg_kip": np.nan,
        }

    return {
        "n": len(sub),
        "avg_ks": sub["Actual Ks"].mean(),
        "median_ks": sub["Actual Ks"].median(),
        "over_rate": (sub["Actual Ks"] > line).mean(),
        "under_rate": (sub["Actual Ks"] < line).mean(),
        "push_rate": (sub["Actual Ks"] == line).mean(),
        "avg_pc": sub["Actual PC"].mean(),
        "avg_kip": sub["K/IP"].mean(),
    }


def compact_rows(sub, max_rows=10):
    cols = [
        "Date", "Pitcher", "Matchup", "Actual Ks", "Actual IP", "Actual PC",
        "K/IP", "Pitch Count Tier", "MLK Tier", "PP Line", "Pick Outcome"
    ]

    available_cols = [c for c in cols if c in sub.columns]
    out = sub[available_cols].copy()

    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    return out.tail(max_rows)


def analyze_pitcher(row, df):
    pitcher = row["Matched Pitcher"]
    opponent = str(row["Opponent Norm"]).strip().upper()
    line = row["PP Line"]
    proj_pc_tier = row["Proj PC Tier"]
    mlk_tier_val = row["MLK Tier"]
    team = str(row["Pitcher Team Norm"]).strip().upper()

    # historical pitcher rows: use name + team when possible
    hist = df[
        (df["Pitcher"].eq(pitcher)) &
        (df["Team_norm"].eq(team))
    ].copy().sort_values("Date")

    if hist.empty:
        hist = df[df["Pitcher"].eq(pitcher)].copy().sort_values("Date")

    recent = hist.tail(5)

    opp_exact = df[
        (df["Opponent_norm"].eq(opponent)) &
        (df["Pitch Count Tier"].eq(proj_pc_tier)) &
        (df["MLK Tier"].eq(mlk_tier_val))
    ].copy().sort_values("Date")

    opp_pc = df[
        (df["Opponent_norm"].eq(opponent)) &
        (df["Pitch Count Tier"].eq(proj_pc_tier))
    ].copy().sort_values("Date")

    hist_sum = rate_summary(hist, line)
    recent_sum = rate_summary(recent, line)
    opp_exact_sum = rate_summary(opp_exact, line)
    opp_pc_sum = rate_summary(opp_pc, line)

    recent_pcs = recent["Actual PC"].dropna().astype(int).tolist()
    recent_ks = recent["Actual Ks"].dropna().astype(int).tolist()
    recent_proj_errors = recent["Proj Error"].dropna().tolist()

    recent_90_rate = np.nan
    if len(recent_pcs) > 0:
        recent_90_rate = sum(x >= 90 for x in recent_pcs) / len(recent_pcs)

    return {
        "pitcher": pitcher,
        "team": team,
        "opponent": opponent,
        "line": line,
        "mlk": row["MLK"],
        "delta": row["Delta"],
        "leash": row["Leash"],
        "proj_pc": row["Proj PC"],
        "proj_ip": row["Proj IP"],
        "proj_bf": row["Proj BF"],
        "proj_pc_tier": proj_pc_tier,
        "mlk_tier": mlk_tier_val,
        "recent_pcs": recent_pcs,
        "recent_ks": recent_ks,
        "recent_proj_errors": recent_proj_errors,
        "recent_90_rate": recent_90_rate,
        "hist_summary": hist_sum,
        "recent_summary": recent_sum,
        "opp_exact_summary": opp_exact_sum,
        "opp_pc_summary": opp_pc_sum,
        "hist_rows": hist,
        "recent_rows": recent,
        "opp_exact_rows": opp_exact,
        "opp_pc_rows": opp_pc,
    }


# =============================
# PDF
# =============================

def dataframe_table(df, max_rows=10, font_size=6):
    if df is None or df.empty:
        return None

    dfx = df.tail(max_rows).copy()
    data = [dfx.columns.tolist()] + dfx.astype(str).values.tolist()

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def sample_label(n):
    if n >= 6:
        return "Good"
    elif n >= 3:
        return "Medium"
    elif n >= 1:
        return "Small"
    return "None"


def build_master_board_table(results):

    rows = []
    scored_results = []

    for r in results:

        exact = r["opp_exact_summary"]
        broad = r["opp_pc_summary"]

        opp_under = (
            exact["under_rate"]
            if pd.notna(exact["under_rate"])
            else broad["under_rate"]
        )

        opp_over = (
            exact["over_rate"]
            if pd.notna(exact["over_rate"])
            else broad["over_rate"]
        )

        if pd.isna(opp_under):
            opp_under = 0

        if pd.isna(opp_over):
            opp_over = 0


        # ----------------
        # CONFLICT
        # ----------------

        conflict = "Aligned"

        if r["delta"] > 0 and opp_under >= .70:
            conflict = "Opp Suppression"

        elif r["delta"] < 0 and opp_over >= .70:
            conflict = "Pitcher Aggression"

        elif abs(r["delta"]) <= .5:
            conflict = "Coinflip"


        # ----------------
        # RISK
        # ----------------

        risk = 0

        if r["leash"] == "Fragile":
            risk += 2

        elif r["leash"] == "Moderate":
            risk += 1


        if exact["n"] < 3:
            risk += 1


        if conflict and conflict != "Coinflip":
            risk += 1


        if pd.notna(r["recent_90_rate"]):

            if r["recent_90_rate"] < .40:
                risk += 1


        risk = min(risk,5)


        # ----------------
        # EDGE SCORE
        # ----------------

        sample_penalty = 0

        if exact["n"] < 3:
            sample_penalty = -1.5

        elif exact["n"] < 6:
            sample_penalty = -.25


        edge = (

            abs(r["delta"])*2

            +

            abs(opp_under-opp_over)*2

            +

            (
                r["recent_90_rate"]
                if pd.notna(r["recent_90_rate"])
                else 0
            )

            +

            sample_penalty
        )

        if abs(r["delta"]) < .5:
            edge -= 1


        play = (
            "OVER"
            if r["delta"] > 0
            else "UNDER"
            if r["delta"] < 0
            else "PASS"
        )


        scored_results.append({

            "pitcher": r["pitcher"],
            "opp": r["opponent"],
            "line": r["line"],
            "mlk": r["mlk"],
            "delta": r["delta"],
            "play": play,
            "leash": r["leash"],
            "recent_ks": safe_list(r["recent_ks"]),
            "recent90": pct(r["recent_90_rate"]),
            "opp_under": pct(opp_under),
            "opp_over": pct(opp_over),
            "risk": risk,
            "conflict": conflict,
            "edge": round(edge,1)

        })


    scored_results = sorted(
        scored_results,
        key=lambda x:x["edge"],
        reverse=True
    )


    rows.append([

        "Pitcher",
        "Opp",
        "Line",
        "MLK",
        "Delta",
        "Play",
        "Leash",
        "Recent Ks",
        "90+",
        "Opp U%",
        "Opp O%",
        "Risk",
        "Conflict",
        "Edge"

    ])


    for r in scored_results:

        rows.append([

            r["pitcher"],
            r["opp"],
            r["line"],
            r["mlk"],
            r["delta"],
            r["play"],
            r["leash"],
            r["recent_ks"],
            r["recent90"],
            r["opp_under"],
            r["opp_over"],
            r["risk"],
            r["conflict"],
            r["edge"]

        ])


    table = Table(
        rows,
        repeatRows=1
    )

    table.setStyle(TableStyle([

        ("BACKGROUND",
         (0,0),
         (-1,0),
         colors.lightgrey),

        ("GRID",
         (0,0),
         (-1,-1),
         0.4,
         colors.black),

        ("FONTNAME",
         (0,0),
         (-1,0),
         "Helvetica-Bold"),

        ("FONTSIZE",
         (0,0),
         (-1,-1),
         6)

    ]))

    return table

def build_pdf(results, board, manual_review_rows, dataset_rows, latest_date):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=18,
        leftMargin=18,
        topMargin=18,
        bottomMargin=18
    )

    styles = getSampleStyleSheet()
    story = []

    # =========================
    # COVER / REPORT SUMMARY
    # =========================

    story.append(Paragraph("MLB Strikeout Evidence Report", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Dataset Rows: {dataset_rows}", styles["BodyText"]))
    story.append(Paragraph(f"Latest Dataset Date: {latest_date}", styles["BodyText"]))
    story.append(Paragraph(f"PP Lines Parsed: {len(board)}", styles["BodyText"]))
    story.append(Paragraph(f"Valid PP Pitchers: {len(results)}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    # =========================
    # MASTER BOARD SUMMARY
    # =========================

    story.append(Paragraph("MASTER BOARD SUMMARY", styles["Heading1"]))
    story.append(Paragraph(
        "Use this page as the triage board. Detailed evidence for each pitcher follows after the summary.",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 8))

    if len(results) > 0:
        story.append(build_master_board_table(results))
    else:
        story.append(Paragraph("No valid pitchers found for PDF.", styles["BodyText"]))

    story.append(PageBreak())

    # =========================
    # MANUAL REVIEW SECTION
    # =========================

    if manual_review_rows is not None and not manual_review_rows.empty:
        story.append(Paragraph("Manual Review / Unmatched Rows", styles["Heading2"]))
        story.append(Paragraph(
            "These rows were excluded from pitcher analysis because the app could not safely match them to today's pregame pitcher pool.",
            styles["BodyText"]
        ))
        story.append(Spacer(1, 8))

        mr = manual_review_rows[["Input Name", "PP Line", "Matched Pitcher", "Match Status"]].copy()
        table = dataframe_table(mr, max_rows=50, font_size=7)
        if table:
            story.append(table)

        story.append(PageBreak())

    # =========================
    # DETAILED EVIDENCE SECTIONS
    # =========================

    for result in results:
        story.append(Paragraph(
            f"{result['pitcher']} ({result['team']}) vs {result['opponent']} — Line {result['line']}",
            styles["Heading1"]
        ))

        proj_text = f"""
        <b>Pregame Projection</b><br/>
        MLK: {result['mlk']}<br/>
        Delta: {result['delta']}<br/>
        Leash: {result['leash']}<br/>
        Proj PC: {result['proj_pc']}<br/>
        Proj IP: {result['proj_ip']}<br/>
        Proj BF: {result['proj_bf']}<br/>
        Proj PC Tier: {result['proj_pc_tier']}<br/>
        MLK Tier: {result['mlk_tier']}<br/><br/>

        <b>Recent Validation Snapshot</b><br/>
        Recent PCs: {safe_list(result['recent_pcs'])}<br/>
        Recent Ks: {safe_list(result['recent_ks'])}<br/>
        Recent 90+ PC Rate: {pct(result['recent_90_rate'])}<br/>
        Recent Projection Errors: {safe_list(result['recent_proj_errors'])}
        """
        story.append(Paragraph(proj_text, styles["BodyText"]))
        story.append(Spacer(1, 8))

        story.append(Paragraph("Pitcher Recent Rows Used", styles["Heading2"]))
        recent_table = dataframe_table(
            compact_rows(result["recent_rows"], max_rows=5),
            max_rows=5,
            font_size=6
        )

        if recent_table:
            story.append(recent_table)
        else:
            story.append(Paragraph("No historical pitcher rows found.", styles["BodyText"]))

        story.append(Spacer(1, 10))

        exact = result["opp_exact_summary"]
        exact_text = f"""
        <b>Opponent Exact Archetype</b><br/>
        Filter: Opponent {result['opponent']} + PC Tier {result['proj_pc_tier']} + MLK Tier {result['mlk_tier']}<br/>
        Rows: {exact['n']} ({sample_label(exact['n'])} sample)<br/>
        Avg Ks: {fmt_num(exact['avg_ks'])}<br/>
        Over Rate vs Line: {pct(exact['over_rate'])}<br/>
        Under Rate vs Line: {pct(exact['under_rate'])}<br/>
        Push Rate vs Line: {pct(exact['push_rate'])}
        """
        story.append(Paragraph(exact_text, styles["BodyText"]))

        if exact["n"] > 0:
            exact_table = dataframe_table(
                compact_rows(result["opp_exact_rows"], max_rows=10),
                max_rows=10,
                font_size=6
            )
            if exact_table:
                story.append(exact_table)

        story.append(Spacer(1, 10))

        broad = result["opp_pc_summary"]
        broad_text = f"""
        <b>Opponent Broader Workload</b><br/>
        Filter: Opponent {result['opponent']} + PC Tier {result['proj_pc_tier']}<br/>
        Rows: {broad['n']} ({sample_label(broad['n'])} sample)<br/>
        Avg Ks: {fmt_num(broad['avg_ks'])}<br/>
        Over Rate vs Line: {pct(broad['over_rate'])}<br/>
        Under Rate vs Line: {pct(broad['under_rate'])}<br/>
        Push Rate vs Line: {pct(broad['push_rate'])}
        """
        story.append(Paragraph(broad_text, styles["BodyText"]))

        if broad["n"] > 0:
            broad_table = dataframe_table(
                compact_rows(result["opp_pc_rows"], max_rows=10),
                max_rows=10,
                font_size=6
            )
            if broad_table:
                story.append(broad_table)

        story.append(PageBreak())

    doc.build(story)

    buffer.seek(0)
    return buffer

# =============================
# APP UI
# =============================

pregame_text = st.text_area("Paste Pregame Read", height=260)
pp_lines_text = st.text_area("Paste PP Lines", height=180)

run = st.button("Run Engine + Generate PDF", type="primary")

if run:
    if not pregame_text.strip() or not pp_lines_text.strip():
        st.error("Paste both pregame read and PP lines.")
        st.stop()

    df = load_dataset()

    st.success(f"Loaded dataset: {len(df)} rows")
    st.write("Latest dataset date:", df["Date"].max())

    pregame_df = parse_pregame_rows(pregame_text)
    pp_df = parse_pp_lines(pp_lines_text)

    board = build_board(pregame_df, pp_df)

    st.subheader("Parsed Board")
    st.dataframe(board)

    manual_review = board[board["Match Status"].ne("ok")].copy()
    valid_board = board[board["Match Status"].eq("ok")].copy()

    if not manual_review.empty:
        st.warning("Some rows need manual review and were excluded from analysis.")
        st.dataframe(manual_review[["Input Name", "PP Line", "Matched Pitcher", "Match Status"]])

    results = []
    for _, row in valid_board.iterrows():
        results.append(analyze_pitcher(row, df))

    st.success(f"Valid pitchers included in PDF: {len(results)}")

    pdf_buffer = build_pdf(
        results=results,
        board=board,
        manual_review_rows=manual_review,
        dataset_rows=len(df),
        latest_date=str(df["Date"].max())
    )

    st.download_button(
        label="Download Full Evidence PDF",
        data=pdf_buffer,
        file_name="mlb_strikeout_evidence_report.pdf",
        mime="application/pdf"
    )
