import streamlit as st
from st_pages import Page, show_pages_from_config

import src.scripts.plot_themes
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

utl.local_css("src/styles/styles_home.css")
utl.external_css(
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"
)

s1, c1, c2 = utl.wide_col()

# my LinkedIn, GitHub, and email
linkedin_url = "https://www.linkedin.com/in/justinas-grigaitis/"
github_url = "https://github.com/justgri"
email_url = "mailto:justinas.grigaitis@econ.uzh.ch"

# Intro
with c1:
    # Title
    st.title("PhD for All: Macroeconomics")

    # Header
    st.markdown(
        '<span style="font-size: 28px; display: block; margin-bottom: 5px;">*Interactive visuals. Rigorous theory. Simple code.*</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "<hr style='margin-top: 0; margin-bottom: 5px;'>",
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""Learning and helping others learn along the way.<br>
            Explaining PhD concepts intuitively.<br>
            Procrastinating productively.""",
        unsafe_allow_html=True,
    )
    st.markdown(
        r"""Targeted at **grad students**, but useful for **professionals** and **undergrads** alike.""",
        unsafe_allow_html=True,
    )

    st.markdown("My other Streamlit apps for PhD students:")

    but_col1, but_col2, _ = st.columns((1, 1, 2))

    but_col1.link_button(
        "PhD Econometrics",
        "https://phd-econometrics.streamlit.app/",
        type="secondary",
    )

    but_col2.link_button(
        "PhD Microeconomics",
        "https://phd-microeconomics.streamlit.app/",
        type="secondary",
    )

    st.markdown(
        f"""
        Please send me feedback:<br>
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

    st.markdown(
        r"""<u>**Disclaimer:**</u> <br>
        This website does not represent the official curriculum taught at my university. <br>
        My goal is to cover fewer topics in greater depth rather than scratch the surface of many. <br>
        All mistakes are my own.
        """,
        unsafe_allow_html=True,
    )


_, note_col, _ = st.columns((0.02, 1, 1.5))
with note_col:
    with st.expander("For mobile users:"):
        st.write("Sidebar leads to other pages within this app.")
        st.image("src/images/intro_mobile_sidebar.png", width=200)

s1, c2, s2 = utl.wide_col()


# Preliminary ToC
with c2:
    st.markdown(
        "<h3 style='text-align: center'>Table of Contents</h3>",
        unsafe_allow_html=True,
    )

    with st.expander("Click to expand"):
        # Page links - potentially hrefs with st.experimental_set_query_params()
        path_must = "https://phd-macroeconomics.streamlit.app/Must-know"
        path_glossary = "https://phd-macroeconomics.streamlit.app/Glossary"

        st.markdown(
            r"""
            Hyperlinks lead to the corresponding pages on this website.""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""[**Glossary of definitions**]({path_glossary})"""
            + r""" **with notation used across the book**
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""[**Top 10 theory tools**]({path_must})"""
            + r""" **that everyone should know**
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            r"""

    <h3>Part I</h3>

    <div class="numbered-header">
        <b>Section 1: Intro to Dymamic Programming</b><br>
    </div>
        
    <div class="numbered">
        1. Cake eating problem - sequential optimization <br>
        2. Recursive formulation and Bellman equation <br>
        3. Principle of Optimality
    </div>

    <br>

    <div class="numbered-header">
        <b>Section 2: Optimal Growth - Neoclassical Model</b><br>
    </div>
        
    <div class="numbered">
        4. Deterministic Consumption-Investment problem <br>
        5. Social planner's problem <br>
    </div>

    <br>

    <div class="numbered-header">
        <b>Section 3: Markets and Competitive Equilbrium</b><br>
    </div>
        
    <div class="numbered">
        6. Consumption-Savings problem with trading (Arrow-Debreu) <br>
        7. First Fundamental Theory of Welfare Economics <br>
    </div>

    <br>

    <div class="numbered-header">
        <b>Section 4: Math of Stochastic Dynamic Programming</b><br>
    </div>
        
    <div class="numbered">
        8. Markov processes<br>
        9. Linear state space models <br>
    </div>

    <br>

    
    <div class="numbered-header">
        <b>Section 5: Stochastic Consumption-Savings Decisions</b><br>
    </div>
        
    <div class="numbered">
        10. Permanent income hypothesis <br>
        11. Precautionary savings <br>
        12. Income fluctuations problem (Aiyagari) <br>
        
    </div>

    <br>

    <div class="numbered-header">
    <b>Section 6: Dynamic Programming Squared </b><br>
    </div>
        
    <div class="numbered">
        13. Recursive contracts - frictionless benchmark <br>
        14. Recursive contracts - moral hazard <br>
        
    </div>

    <br>
    
     <h3>Part II</h3>
    

     <div class="numbered-header">
        <b>Section X: TBD</b><br>
    </div>
        
    <div class="numbered">
        1. TBD <br>
        2. TBD <br>
    </div>

    <br>


    Next semester - TBD. <br>

    Chapters follow Stokey, Lucas, Prescott *Recursive Methods in Economic Dynamics* (1989).<br>
    Chapters from Ljungqvist and Sargent *Recursive Macroeconomic Theory* (2012) are added were relevant.<br>
    Subsections are likely to change depending on which topics I find most interesting or challenging.

    """,
            unsafe_allow_html=True,
        )

# What is Macroeconomics?
with c2:
    st.markdown(
        "<h3 style='text-align: center'>Micro Foundations for Macroeconomics</h3>",
        unsafe_allow_html=True,
    )

    with st.expander("Click to expand"):
        st.markdown(
            r"""
            "This book is about micro foundations for macroeconomics. Browning, Hansen,
                and Heckman (1999) identify two possible justifications for putting microfoundations
                underneath macroeconomic models. The first is aesthetic and preempirical:
                models with micro foundations are by construction coherent and explicit.
                And because they contain descriptions of agents' purposes, they allow us to analyze
                policy interventions using standard methods of welfare economics. Lucas
                (1987) gives a distinct second reason: a model with micro foundations broadens
                the sources of empirical evidence that can be used to assign numerical values
                to the model’s parameters. Lucas endorses Kydland and Prescott's (1982) procedure
                of borrowing parameter values from micro studies. Browning, Hansen,
                and Heckman (1999) describe some challenges to Lucas's recommendation for
                an empirical strategy. Most seriously, they point out that in many contexts the
                specifications underlying the microeconomic studies cited by a calibrator conflict
                with those of the macroeconomic model being “calibrated.” It is typically not
                obvious how to transfer parameters from one data set and model specification
                to another data set, especially if the theoretical and econometric specification
                differs.<br>
                Although we take seriously the doubts about Lucas’s justification for microeconomic
                foundations that Browning, Hansen and Heckman raise, we remain
                strongly attached to micro foundations. For us, it remains enough to appeal to
                the first justification mentioned, the coherence provided by micro foundations
                and the virtues that come from having the ability to “see the agents” in the
                artificial economy. We see Browning, Hansen, and Heckman as raising many
                legitimate questions about empirical strategies for implementing macro models
                with micro foundations. We don’t think that the clock will soon be turned back
                to a time when macroeconomics was done without micro foundations."
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""Ljungqvist, Lars, and Thomas J. Sargent. “Preface to the Third Edition.”
            In Recursive Macroeconomic Theory, xx–xxxvi. The MIT Press, 2012. http://www.jstor.org/stable/j.ctt5vjq05.4.""",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<h3 style='text-align: center'>Recursive Approach</h3>",
        unsafe_allow_html=True,
    )

    with st.expander("Click to expand"):
        st.markdown(
            r"""
            "Recursive methods constitute a powerful approach to dynamic economics due
                to their described focus on a tradeoff between the current period's utility and a
                continuation value for utility in all future periods. As mentioned, the simplification
                arises from dealing with the evolution of state variables that capture the
                consequences of today's actions and events for all future periods, and in the case
                of uncertainty, for all possible realizations in those future periods. Not only is
                this a powerful approach to characterizing and solving complicated problems,
                but it also helps us to develop intuition, conceptualize, and think about dynamic
                economics. Students often find that half of the job in understanding how
                a complex economic model works is done once they understand what the set of
                state variables is. Thereafter, the students are soon on their way to formulating
                optimization problems and transition equations. Only experience from solving
                practical problems fully conveys the power of the recursive approach. This book
                provides many applications.<br>
                Still another reason for learning about the recursive approach is the increased
                importance of numerical simulations in macroeconomics, and most computational
                algorithms rely on recursive methods. When such numerical simulations
                are called for in this book, we give some suggestions for how to proceed
                but without saying too much on numerical methods."
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""Ljungqvist, Lars, and Thomas J. Sargent. “Preface to the Third Edition.”
            In Recursive Macroeconomic Theory, xx–xxxvi. The MIT Press, 2012. http://www.jstor.org/stable/j.ctt5vjq05.4.""",
            unsafe_allow_html=True,
        )


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
        "Intermediate Quantitative Economics with Python by QuantEcon (Sargent and Stachurski)",
        "https://python.quantecon.org/intro.html",
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
