import streamlit as st
from st_pages import Page, add_page_title, show_pages, show_pages_from_config

import src.scripts.plot_themes as thm
import src.scripts.utils as utl

st.set_page_config(
    page_title="PhD Macroeconomics",
    page_icon="🌐",
    layout="wide",
)

utl.local_css("src/styles/styles_home.css")
utl.external_css(
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"
)

show_pages_from_config()

s1, c1, c2 = utl.wide_col()

# my LinkedIn, GitHub, and email
linkedin_url = "https://www.linkedin.com/in/justinas-grigaitis/"
github_url = "https://github.com/justgri"
email_url = "mailto:justinas.grigaitis@econ.uzh.ch"

# Intro
with c1:
    st.title("Macroeconomics for PhD Students")
    st.sidebar.success("Select a page above.")

    st.markdown(
        "Trying to learn and enjoy the first year of Econ PhD. <br> Procrastinating productively. <br> All mistakes are my own.",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        **Disclaimer:** <br>
        This website does not represent the official curriculum taught at my university. <br>
        My goal is to cover fewer topics in greater depth rather than scratch the surface of many. <br>
        Visuals are meant to capture the key concepts, which could be helpful to undergraduate students and industry professionals, too. <br>
        """,
        # Main difference is matrix algebra and proving everything along the way, which might not always be included here.
        # Hopefully it will give insights to both PhD students, undergrads, and others.
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        Please send me corections and suggestions: 
    <a href="{linkedin_url}" target="_blank">
        <i class="fab fa-linkedin fa-lg"></i>
    </a>
    <a href="{email_url}" target="_blank">
        <i class="fas fa-envelope fa-lg"></i>
    </a>
    <a href="{github_url}" target="_blank">
        <i class="fab fa-github fa-lg"></i>
    </a>
    """,
        unsafe_allow_html=True,
    )


s1, c2, s2 = utl.narrow_col()

# Textbooks
with c2:
    st.markdown(
        "<h3 style='text-align: center'>Reference Textbooks</h3>",
        unsafe_allow_html=True,
    )

    c2_1, s2_1, c2_2 = st.columns((1, 0.05, 1))

    with c2_1:
        st.image("src/images/intro_slp.jpg", width=350)

    with c2_2:
        st.image("src/images/intro_sargent_ljungqvist.jpg", width=420)


# Other references
with c2:
    st.markdown(
        "<h3 style='text-align: center'>Other References</h3>",
        unsafe_allow_html=True,
    )
    st.link_button(
        "Lecture Notes by Florian Scheuer (U of Zurich)",
        "https://www.econ.uzh.ch/en/people/faculty/scheuer/teaching/Fall-2017.html",
        type="secondary",
    )
    st.link_button(
        "Lecture Notes by Jennifer La'O (Columbia)",
        "https://www.jennifer-la-o.com/macroeconomic-analysis-i",
        type="secondary",
    )

    st.link_button(
        "Lecture notes by Eric Sims (Notre Dame)",
        "https://sites.nd.edu/esims/courses/ph-d-macro-theory-ii/",
        type="secondary",
    )

    st.link_button(
        "Advanced Quantitative Economics with Python by QuantEcon (Sargent and Stachurski)",
        "https://python-advanced.quantecon.org/intro.html",
        type="secondary",
    )

    st.link_button(
        "PhD Macroeconomics YouTube playlist by Thomas Sargent",
        "https://youtube.com/playlist?list=PLLAPgKPWbsiSGvDwJN87YuXEPcf6B0JHd&si=Tjrk6g3ZgP1bBhKz",
        type="secondary",
    )

    st.link_button(
        "Intro to Avanced Macroeconomics (Master's) by Michael C. Burda (HU Berlin)",
        "https://youtube.com/playlist?list=PLJZlW3ik4xixAhVnY0aaTrz72XCZsygEA&si=_j3pktBu8xdRvEHS",
        type="secondary",
    )


with c2:
    st.markdown(
        "<h3 style='text-align: center'>What is Macroeconomics?</h3>",
        unsafe_allow_html=True,
    )

    econometrica_public = "https://www.sv.uio.no/econ/om/tall-og-fakta/nobelprisvinnere/ragnar-frisch/published-scientific-work/rf-published-scientific-works/rf1933c.pdf"

    st.markdown(
        r"""
        Find a quote
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""Find a quote (free access [link]({econometrica_public}))""",
        unsafe_allow_html=True,
    )

# Preliminary ToC

with c2:
    st.markdown(
        "<h3 style='text-align: center'>Tentative Table of Contents</h3>",
        unsafe_allow_html=True,
    )

    with st.expander("Click to expand", expanded=False):
        st.write(
            "Chapters follow Stokey, Lucas, Prescott *Recursive Methods in Economic Dynamics* (1989). Chapters from Ljungqvist and Sargent *Recursive Macroeconomic Theory* (2012) are added were relevant."
        )
        st.write(
            "Subsections are likely to change depending on which topics I find most interesting or challenging."
        )

        st.markdown(
            r"""
    <div class="numbered-header">
        <b>Section 1: Endowment Economy</b><br>
    </div>
        
    <div class="numbered">
        1. Topic 1 (SLP Chapters) <br>
        2. Topic 2 (SLP Chapters) <br>
    </div>

    <br>

    <div class="numbered-header">
        <b>Section 2: Dynamic Programming</b><br>
    </div>
        
    <div class="numbered">
        3. Topic 3 (SLP Chapters) <br>
        4. Topic 4 (SLP Chapters) <br>
    </div>

    <br>

    Next semester - TBD. <br>
    Bonus if time permits (it never does) - TBD.
    """,
            unsafe_allow_html=True,
        )

# Wooldridge Top 10
with c2:
    st.markdown(
        "<h3 style='text-align: center'>10 Tools for Macroeconomics</h3>",
        unsafe_allow_html=True,
    )

    st.write(f"Find a good source for top 10 things to know in macroeconomics.")
    st.write(f"Whom to follow on Twitter?")

    with st.expander("Click to expand", expanded=True):
        st.markdown(
            r"""
        1. **Top 1 thing to know** <br>
        Some fancy formula: $E[x] = E[E[x|y]]$ <br>
        
        2. **Top 2 thing to know** <br>
        Some other fancy formula: $E[x] = E[E[x|y]]$ <br>
        
        """,
            unsafe_allow_html=True,
        )
