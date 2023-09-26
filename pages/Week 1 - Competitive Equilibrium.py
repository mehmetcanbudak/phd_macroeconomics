import datetime
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

import plotly_themes as thm

st.title("1. Competitive Equilibrium")

# Dashboard header
st.header("Learn about Competitive Equilibrium Visually")
# Dashboard description

if "my_input" not in st.session_state:
    st.session_state["my_input"] = ""

st.markdown(
    "This is where the project description will go. "
    "This is a long description. "
    "<br>"
    "And one more line"
    "<br>"
    f"And here goes text from last page: {st.session_state['my_input']}",
    unsafe_allow_html=True,
)


# Functions to make columns with consistent width
def two_cols():
    return st.columns((0.1, 1, 0.1, 1, 0.1))


def one_col():
    return st.columns((0.5, 1, 0.5))


# Create a simple chart with a slider
# Row 0 with the title
r0_s1, r0_c1, r0_s2 = one_col()


def u_ln(c):
    return np.log(c)


def u_crra(c, sigma=0.5):
    if sigma == 0:
        return np.log(c)
    else:
        return (c ** (1 - sigma)) / (1 - sigma)


def plot_utility(sigma=None):
    x = np.linspace(0.1, 3, 100)

    fig = plt.figure()
    if u_func == "Natural Log":
        plt.plot(x, u_ln(x), color="red")
    elif u_func == "CRRA":
        plt.plot(x, u_crra(x, sigma), color="blue")

    plt.xlim([0, 3])
    # plt.ylim([-1, 2])

    return fig


def optimal_cons(beta=None):
    t = [i for i in range(1, 101)]

    T = len(t)

    def cons_1(t, beta):
        if beta == 1:
            return [0.5 for i in range(T)]
        else:
            return [(beta * (1 - beta)) / (1 - beta**2) for i in range(T)]

    def cons_2(t, beta):
        if beta == 1:
            return [0.5 for i in range(T)]
        else:
            return [(1 - beta) / (1 - beta**2) for i in range(T)]

    fig = plt.figure()

    plt.plot(t, cons_1(t, beta), color="red", label="consumer 1")
    plt.plot(t, cons_2(t, beta), color="blue", label="consumer 2")

    plt.xlim([0, len(t)])
    plt.ylim([0, 1])

    return fig


with r0_c1:
    st.markdown(
        "<h3 style='text-align: center'>Utility Function</h3>",
        unsafe_allow_html=True,
    )

    u_func = st.selectbox(
        "Select your prefered utility function", ("CRRA", "Natural Log")
    )

    st.write("Your select utility function:", u_func)

    if u_func == "CRRA":
        sigma = st.slider("Choose your sigma", 0.0, 5.0, 0.01)
        st.write("Sigma:", sigma)
        st.pyplot(plot_utility(sigma))

    else:
        st.pyplot(plot_utility())

    st.markdown(
        "<h3 style='text-align: center'>Optimal Consumption at t</h3>",
        unsafe_allow_html=True,
    )

    beta = st.slider("Choose your Beta", 0.0, 1.0, 0.01)

    st.pyplot(optimal_cons(beta))
