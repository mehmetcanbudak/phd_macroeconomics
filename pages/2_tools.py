import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from st_pages import show_pages_from_config

import src.scripts.plot_themes as thm
import src.scripts.utils as utl

### PAGE CONFIGS ###

st.set_page_config(
    page_title="PhD Macroeconomics - Top Tools",
    page_icon="🌐",
    layout="wide",
)

utl.local_css("src/styles/styles_pages.css")

random_seed = 0
s1, c1, s2 = utl.wide_col()

### PAGE START ###
# Dashboard header
with c1:
    st.title("Top 10 Theory Tools for Macroeconomics")
    st.divider()
    st.markdown(r"""<h3>According to TBD</h3>""", unsafe_allow_html=True)


s1, c2, s2 = utl.wide_col()

with c2:
    st.markdown(
        r"""
    <h5>1. Title TBD</h5>

    Description TBD:<br>
    $Formula_1\; TBD$<br>
    $Formula_2\; TBD$

    <h5>2. Title TBD</h5>

    Description TBD:<br>
    $Formula_1\; TBD$<br>
    $Formula_2\; TBD$

    <h5>3. Title TBD</h5>

    Description TBD:<br>
    $Formula_1\; TBD$<br>
    $Formula_2\; TBD$

    <h5>4. Title TBD</h5>

    Description TBD:<br>
    $Formula_1\; TBD$<br>
    $Formula_2\; TBD$

    <h5>5. Title TBD</h5>

    Description TBD:<br>
    $Formula_1\; TBD$<br>
    $Formula_2\; TBD$

    """,
        unsafe_allow_html=True,
    )
