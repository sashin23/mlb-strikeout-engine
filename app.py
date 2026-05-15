import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("MLB Strikeout Decision Engine")

sheet_url = st.text_input(
    "Google Sheet CSV URL",
    ""
)

pregame = st.text_area(
    "Paste Pregame Read",
    height=250
)

pp_lines = st.text_area(
    "Paste PP Lines",
    height=150
)

run = st.button("Run Engine")

if run:

    st.success("Inputs captured")

    st.write("Pregame chars:", len(pregame))
    st.write("PP chars:", len(pp_lines))

    try:

        df = pd.read_csv(sheet_url)

        st.write("Dataset rows:", len(df))

        st.dataframe(
            df.head()
        )

    except:

        st.warning(
            "Google sheet not loaded"
        )
