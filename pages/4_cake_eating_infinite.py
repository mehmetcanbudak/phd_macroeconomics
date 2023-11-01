import os
import random
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import src.scripts.plot_themes as thm
import src.scripts.utils as utl
import statsmodels.api as sm
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from plotly.subplots import make_subplots
from scipy.stats import t
from st_pages import show_pages_from_config

### PAGE CONFIGS ###
st.set_page_config(
    page_title="PhD Macroeconomics - Cake Eating Inf",
    page_icon="🌐",
    layout="wide",
)

utl.local_css("src/styles/styles_pages.css")

random_seed = 0


#### Cake eating problem
# Sources
comp_econ = "https://juejung.github.io/jdocs/Comp/html/Slides_Optimization_2_Cake.html"  # with ln(c); note a typo in the c_0 formula (no parenthesis needed in the denom)
quant_econ = "https://python.quantecon.org/cake_eating_problem.html"  # with CRRA

uni_basel = "https://wwz.unibas.ch/fileadmin/user_upload/wwz/00_Professuren/Berentsen_Wirtschaftstheorie/Lecture_Material/Monetary_Theory/FS_2018/Dynamic_Programming.pdf"  # with ln(c) - infinite and finite
dynamic_econ = "https://ifs.org.uk/sites/default/files/output_url_files/Dynamic%252520Economics%252520slides%252520-%252520handout.pdf"  # analytical with CRRA


video = "https://www.youtube.com/watch?v=UCbTlB-4dcA&list=PLLAPgKPWbsiQ0Ejh-twYC3Fr8_WA9BKCc&index=1"  # decent explanation of backwards steps, but wrong conclusion - doesn't show the analytical solution


## Data viz part
# Show consumption each period, cumulative utility
# Show cumulative consumption against cake left
# Play animation of pie shrinking with each period

## Theory part
# All about sequential optimization

# create one column with consistent width
s1, c01, s2 = utl.wide_col()

### PAGE START ###
# Dashboard header

with c01:
    st.title("Cake Eating Problem")
    st.header("Infinite Time Horizon")
    st.divider()
    st.header("1. Visualizing Optimal Consumption")
    st.markdown(
        r"""
    Let's suppose that the cake never goes bad and you have an infinite amount of time to finish it.<br>
    How would you choose an optimal consumption path in that case?<br>
    This might sound unrealistic, but speaking more broadly,
    a lot of optimization problems don't have a specific end date
    (e.g., imagine that you pass the leftovers of the imperishable cake onto your children and they can pass it onto their children, etc.).
    That's how economists often think about consumption-savings problems and other similar models.
    Therefore, it's important to get familiar with the cake eating problem in an infinite horizon case.<br>
""",
        unsafe_allow_html=True,
    )
    st.markdown(r"""The problem then becomes:""", unsafe_allow_html=True)

    st.latex(
        r""" \max_{\{c_t\}_{t=0}^{\infty}} \sum_{t=0}^{\infty} \beta^t u(c_t) \\
        \\[5pt]
        \text{s.t.}w_{t+1} = w_t - c_t,\; c_t\geq 0 \; \forall t\\
             """
    )

    st.markdown(
        r"""In an infinite horizon case, we cannot find the solution by starting at the end and solving it backwards as we did before.
        However, there are other methods to get an analytical or at least a numerical solution.
        The main solution method in dynamic programming is using the **Bellman equation**. We will talk about it more in the next section, 
        but for now, let's just look at the optimal consumption rule (policy) for the cake eating problem with CRRA utility:""",
        unsafe_allow_html=True,
    )

    st.latex(r"""c_t^*(w_t) = (1-\beta^{1/\gamma})w_t""")

    st.markdown(
        r"""
        As in the finite horizon case, the optimal consumption in each period depends on $\beta$ and $\gamma$.<br>
        Let's compare how they change the optimal consumption path and conduct a comparative statics analysis
        for three sets of parameters.

        """,
        unsafe_allow_html=True,
    )

# Using columns to separate the inputs
param_col_1, param_col_2, param_col_3 = st.columns(3)

# Inputs for first set
with param_col_1:
    st.markdown("#### :green[Line 1]")
    beta_1 = st.number_input(
        "β (patience):",
        min_value=0.0,
        max_value=0.99,
        value=0.8,
        step=0.1,
        key="beta1",
    )
    gamma_1 = st.number_input(
        "γ (consumption smoothing):",
        min_value=0.05,
        max_value=5.0,
        value=0.95,
        step=0.1,
        key="gamma1",
    )

# Inputs for second Line
with param_col_2:
    st.markdown("#### :blue[Line 2]")
    beta_2 = st.number_input(
        "β (patience):",
        min_value=0.0,
        max_value=0.99,
        value=0.95,
        step=0.1,
        key="beta2",
    )
    gamma_2 = st.number_input(
        "γ (consumption smoothing):",
        min_value=0.05,
        max_value=5.0,
        value=0.95,
        step=0.1,
        key="gamma2",
    )

# Inputs for third Line
with param_col_3:
    st.markdown("#### :orange[Line 3]")
    beta_3 = st.number_input(
        "β (patience):",
        min_value=0.0,
        max_value=0.99,
        value=0.95,
        step=0.1,
        key="beta3",
    )
    gamma_3 = st.number_input(
        "γ (consumption smoothing):",
        min_value=0.05,
        max_value=5.0,
        value=0.8,
        step=0.1,
        key="gamma3",
    )


def solve_cake_infinite(W, max_t, beta, gamma):
    consumption = np.zeros(max_t)
    remaining_cake = np.zeros(max_t)
    utility = np.zeros(max_t)

    remaining_cake[0] = W  # Initial cake size
    target_cons = W * 0.8  # 80% of initial cake size

    for t in range(max_t):
        consumption[t] = (1 - beta ** (1 / gamma)) * remaining_cake[t]

        if t < 99:
            remaining_cake[t + 1] = remaining_cake[t] - consumption[t]

        if consumption[t] > 0:
            utility[t] = utl.crra_utility(consumption[t], gamma) * (beta**t)
        else:
            utility[t] = 0

    # Calculate cumulative consumption
    cum_consumption = np.cumsum(consumption)

    # Find the period when 80% of the cake is eaten
    if cum_consumption[-1] < target_cons:
        periods_eat_80_pct = ">1000"
    else:
        # argmax finds first max value of the array and returns its index
        periods_eat_80_pct = np.argmax(cum_consumption >= target_cons) + 1

    pct_eaten_in_5 = (np.sum(consumption[:5]) / W) * 100

    tot_utility_1000 = np.sum(utility[:1000])
    cum_utility = np.cumsum(utility)

    return (
        consumption[:max_t],
        cum_consumption[:max_t],
        cum_utility[:max_t],
        periods_eat_80_pct,
        pct_eaten_in_5,
        tot_utility_1000,
    )


W_inf, max_t = 100, 1000
# First scenario
(
    cons_1,
    cum_cons_1,
    cum_ut_1,
    periods_80_1,
    pct_5_1,
    cum_util_1,
) = solve_cake_infinite(W_inf, max_t, beta_1, gamma_1)
# Second scenario
(
    cons_2,
    cum_cons_2,
    cum_ut_2,
    periods_80_2,
    pct_5_2,
    cum_util_2,
) = solve_cake_infinite(W_inf, max_t, beta_2, gamma_2)
# Third scenario
(
    cons_3,
    cum_cons_3,
    cum_ut_3,
    periods_80_3,
    pct_5_3,
    cum_util_3,
) = solve_cake_infinite(W_inf, max_t, beta_3, gamma_3)


def plot_inf_horizon(W_inf, max_t, title, y_title, legend_dict, line_1, line_2, line_3):
    # Creating the plot
    fig = go.Figure()

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=list(range(max_t + 1)),
            y=line_1,
            mode="lines",
            name=f"β={beta_1:.2f}, γ={gamma_1:.2f}",
            line=dict(color="green", width=1.5, dash="solid"),
            hovertemplate="c<sub>t</sub> = %{y:.1f}<extra></extra>",
            hoverlabel=dict(bgcolor="green", font=dict(color="white")),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(max_t + 1)),
            y=line_2,
            mode="lines",
            name=f"β={beta_2:.2f}, γ={gamma_2:.2f}",
            line=dict(color="blue", width=1.5, dash="dash"),
            hovertemplate="c<sub>t</sub> = %{y:.1f}<extra></extra>",
            hoverlabel=dict(bgcolor="blue", font=dict(color="white")),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(max_t + 1)),
            y=line_3,
            mode="lines",
            name=f"β={beta_3:.2f}, γ={gamma_3:.2f}",
            line=dict(color="orange", width=1.5, dash="dot"),
            hovertemplate="c<sub>t</sub> = %{y:.1f}<extra></extra>",
            hoverlabel=dict(bgcolor="orange", font=dict(color="black")),
        )
    )

    # Update layout
    fig.update_layout(
        width=600,  # Width in pixels
        height=400,
        margin=dict(autoexpand=False, l=50, r=30, t=25, b=105, pad=0),
        font=dict(family="Sans-Serif", color="black"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=legend_dict,
        # title="",  # set title with st.markdown()
        title=dict(
            text=title,
            font=dict(size=22),
            x=0.5,
            xanchor="center",
            xref="paper",
            y=0.98,
            yanchor="top",
            yref="container",
        ),
        hovermode="x",
        hoverlabel=dict(
            font=dict(family="Sans-Serif", color="black"),
            bgcolor="white",
            bordercolor="black",
            namelength=-1,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            autorange=False,
            title=dict(text="<b>period t</b>", font=dict(size=16)),
            tickformat=".0f",
            rangeslider=dict(
                visible=True, range=[0, 100], autorange=False, thickness=0.1
            ),
            range=[0, 20],
            fixedrange=True,
        ),
        yaxis=dict(
            autorange=True,
            showgrid=False,
            showspikes=False,
            title=dict(
                text=y_title,
                standoff=0.2,
                font=dict(size=16),
            ),
        ),
    )

    return fig


col_inf_1, col_inf_2 = st.columns(2)
col_inf_3, col_inf_4 = st.columns(2)

with col_inf_1:
    st.markdown(
        "<div style='height: 23px;'></div>", unsafe_allow_html=True
    )  # add a spacer row

    st.plotly_chart(
        plot_inf_horizon(
            W_inf,
            max_t,
            title="<b>Optimal Consumption Path</b>",
            y_title="<b>Consumption (cₜ)</b>",
            legend_dict=dict(
                x=1,
                y=1,
                xanchor="right",
                yanchor="top",
                orientation="v",
            ),
            line_1=cons_1,
            line_2=cons_2,
            line_3=cons_3,
        ),
        theme=None,
        use_container_width=True,
    )


with col_inf_3:
    st.markdown(
        "<div style='height: 23px;'></div>", unsafe_allow_html=True
    )  # add a spacer row

    st.plotly_chart(
        plot_inf_horizon(
            W_inf,
            max_t,
            title="<b>Cake Eaten Until t</b>",
            y_title="<b>Cumulative Consumption (Σcₜ)</b>",
            legend_dict=dict(
                x=1,
                y=0,
                xanchor="right",
                yanchor="bottom",
                orientation="v",
            ),
            line_1=cum_cons_1,
            line_2=cum_cons_2,
            line_3=cum_cons_3,
        ),
        theme=None,
        use_container_width=True,
    )


with col_inf_4:
    st.markdown(
        "<div style='height: 23px;'></div>", unsafe_allow_html=True
    )  # add a spacer row

    st.plotly_chart(
        plot_inf_horizon(
            W_inf,
            max_t,
            title="<b>Disc. Cumulative Utility Until t</b>",
            y_title="<b>Cumulative Utility (Σβ<sup>t</sup>u(cₜ))</b>",
            legend_dict=dict(
                x=0.01,
                y=1,
                xanchor="left",
                yanchor="top",
                orientation="v",
            ),
            line_1=cum_ut_1,
            line_2=cum_ut_2,
            line_3=cum_ut_3,
        ),
        theme=None,
        use_container_width=True,
    )


def comp_stat_table():
    set_1 = {
        "name": "Line 1",
        "gamma": gamma_1,
        "beta": beta_1,
        "periods_80": periods_80_1,
        "pct_5": pct_5_1,
        "cum_util": cum_util_1,
    }
    set_2 = {
        "name": "Line 2",
        "gamma": gamma_2,
        "beta": beta_2,
        "periods_80": periods_80_2,
        "pct_5": pct_5_2,
        "cum_util": cum_util_2,
    }
    set_3 = {
        "name": "Line 3",
        "gamma": gamma_3,
        "beta": beta_3,
        "periods_80": periods_80_3,
        "pct_5": pct_5_3,
        "cum_util": cum_util_3,
    }

    sets = [set_1, set_2, set_3]
    # Start the table and add headers
    table_html = "<table>\n"
    table_html += "    <tr><th>Name</th>"

    # Add column headers
    for s in sets:
        table_html += f"<th>{s['name']}</th>"
    table_html += "</tr>\n"

    # List of attributes to display
    attributes = ["gamma", "beta", "periods_80", "pct_5", "cum_util"]
    attributes_clean = [
        "Gamma",
        "Beta",
        "Number of periods to eat 80% of cake",
        "Cake eaten in first 5 periods",
        "Sum β<sup>t</sup>u(c) for first 1000 periods",
    ]

    # Loop through each attribute
    for attr, attr_clean in zip(attributes, attributes_clean):
        table_html += f"<tr><td style='text-align: center;'>{attr_clean}</td>"
        for s in sets:
            # Format the value based on the type
            if attr in ["gamma", "beta"]:
                val = f"{s[attr]:.2f}"
            elif attr == "pct_5":
                val = f"{s[attr]:.2f}%"
            elif attr == "periods_80":
                if s[attr] == ">1000":
                    val = ">1000"
                else:
                    val = f"{s[attr]:.0f}"
            else:
                val = f"{s[attr]:.2f}"

            table_html += f"<td style='text-align: center;'>{val}</td>\n"

        table_html += "</tr>\n"

    # Close the table
    table_html += "</table>"

    return table_html


with col_inf_2:
    st.markdown("#### Comparative Statics", unsafe_allow_html=True)
    st.markdown(comp_stat_table(), unsafe_allow_html=True)
    st.markdown("")
    st.markdown(
        r"""<div style="font-size: small;">
                NB: Don't be confused by negative utility when γ > 1, since units of utility don't have a real meaning.
                What we found is the optimal consumption path that leads to the highest cumulative utility possible for a given set of parameters
                given the infinite horizon, even if the utility is negative.
                Comparison of cumulative utilities is also not meaningful per se and is only shown to illustrate the relative changes in curvuture.
                </div>
                """,
        unsafe_allow_html=True,
    )


s0, c02, s1 = utl.wide_col()

with c02:
    st.header(r"2. Cake eating problem extensions")
    st.markdown(
        r"""The cake eating problem can be extended to include many additional features:""",
        unsafe_allow_html=True,
    )
    # NB timing is important - covered in finite case section
    # remaining cake indicates cake in the morning; then consumption indicates consumption throughout the day, and remaining cake next morning is remaining cake previous morning - consumption during the day

    st.markdown(
        r"""
        1. Cake can shrink over time, e.g., your friend might be secretely eating it at night.<br>
        Let's call this *depreciation rate* $\delta$.<br>
        $w_{t+1} = (1-\delta)w_t - c_t$ (if cake depreciates in the morning)<br>
        $w_{t+1} = (1-\delta)(w_t - c_t)$ (if cake depreciates over night)<br>
        In the latter case, solution with CRRA utility becomes:<br>
        $c_{t+1} = (\beta (1-\delta))^{1/\gamma}c_t$ (Euler equation)<br>
        $c_t^*(w_t) = (1-\beta^{1/\gamma}(1-\delta)^{\frac{1-\gamma}{\gamma}})w_t$<br>
        $w_{t+1}^*(w_t) = (\beta^{1/\gamma})(1-\delta)^{\frac{1-\gamma}{\gamma}} w_t$<br>
        
        2. Cake can increase over time, e.g., like in the marshmallow experiment, your friend is rewarding your patience proportionally to the cake left.<br>
        Let's call this *interest rate* $R$.<br>
        $w_{t+1} = R w_t - c_t$ (if interest is paid in the morning)<br>
        $w_{t+1} = R(w_t - c_t)$ (if interested is paid at night)<br>
        Solution is the same as in the case of depreciation, just replace ($1-\delta$) with $R$.<br>
        
        3. New cake can also be produced, e.g., your friend is using cake leftovers to bake something additional - you at least get what you had before and some new cake,
        depending on how productive your friend is.<br>
        Let's call this *production function* $w^\alpha$.<br>
        $w_{t+1} = w_t + w_t^\alpha - c_t$<br>
        """,
        unsafe_allow_html=True,
    )

    # Solve part 1 both cases via guess-and-verify
    # Check why optimal policy doesn't change with log utility but does change with CRRA utility

    st.markdown(
        r"""
        You can also receive additional cake of random size (*income shocks*), your tastes can change (*utility shocks*), you can sell some cake today and buy some tomorrow (*exchange economy*),
        you can sell some cake and invest the money in cake production to get even more cake in the future (*consumption-investment problem*), etc. etc.<br>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""
        All these features will eventually be studied in the following sections.
        Let's keep this in mind and see if you can relate them back to the cake eating problem!""",
        unsafe_allow_html=True,
    )

    # for now, let's focus on the infinite horizon case with CRRA utility
    # and see how depreciation $\delta$, interest rate $R$, and additional cake production $w^\alpha$ affect the optimal consumption in this toy model - can you spot the pattern?""",

s0, c03, s1 = utl.wide_col()

with c03:
    st.header("3. Theory references")
    st.write("Main reference:")
    st.link_button(
        "QuantEcon with Python - Cake Eating",
        "https://python.quantecon.org/cake_eating_problem.html",
        type="primary",
    )

    st.write(
        "Finite and infinite case (guess-and-verify) with log utility; taste shocks:"
    )
    st.link_button(
        "Dynamic Programming: An overview (Prof. Russell Cooper)",
        "https://www2.econ.iastate.edu/tesfatsi/dpintro.cooper.pdf",
        type="primary",
    )

    st.write("Infinite case with CRRA utility and interest rate; technical notes:")

    st.link_button(
        "Notes on Dynamic Programming (Prof. Lutz Hendricks)",
        "https://lhendricks.org/econ720/ih1/Dp_ln.pdf",
        type="primary",
    )

    st.write(
        "Video explanation finite and infite horizon with log utility; depreciation:"
    )
    st.link_button(
        "Dynamic Programming Economics YouTube (by EconJohn)",
        "https://youtube.com/playlist?list=PLLAPgKPWbsiQ0Ejh-twYC3Fr8_WA9BKCc&si=jSGmuYGe_MtnvtTP",
        type="secondary",
    )
