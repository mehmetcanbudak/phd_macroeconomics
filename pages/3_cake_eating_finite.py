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
    page_title="PhD Macroeconomics - Cake Eating Fin",
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
    st.header("Finite Time Horizon")
    st.divider()
    st.header("1. Visualizing Optimal Consumption")

    st.markdown(
        r"""Suppose you get a cake of size $W$ and you want to eat it over ($T+1$) day.
        You want to be smart about it and ask yourself: what is the **optimal consumption** each day to maximize your happiness derived from eating the entire cake?
        We formulate this problem as follows:<br>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        r""" \max_{\{c_t\}_{t=0}^T} \sum_{t=0}^{T} \beta^t u(c_t) \\
        \\[5pt]
        \text{s.t.} \sum_{t=0}^{T} c_t \leq W\\
        \\[5pt]
          w_{t+1} = w_t - c_t,\; c_t\geq 0 \; \forall t\\
             """
    )
    st.markdown(
        r"""Given a specific utility function, an analytical solution can be found by setting $c_T = w_T$, using Euler equation from FOC, and solving backwards.
        The solution is given below with $u(c) = \text{ln}(c)$. If utility function is CRRA, then $\beta$ has to be replaced with $\beta^{\frac{1}{\gamma}}$.
        """,
        unsafe_allow_html=True,
    )
    st.latex(
        r""" c_0 = \dfrac{W}{\sum_{t=0}^{T} \beta^t} = W \dfrac{1-\beta}{1-\beta^{T+1}} \\
        \\[5pt]
             c_t = \beta c_{t-1} = \beta^t c_0 \; \forall t \\"""
    )

    st.markdown(
        r"""
        Let's see how you should optimally eat the cake, given different $\beta$, $u(c)$, and $T$. Cake size is fixed to $W=100$.
       
        """,
        unsafe_allow_html=True,
    )

input_col, _, chart_col = st.columns((0.5, 0.05, 1))

### WIDGETS
### Input widgets
with input_col:
    beta = st.number_input(
        r"Select $\beta$ (patience factor):",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.01,
    )
    T = st.number_input(
        r"Select $T$ (last period):",
        min_value=2,
        max_value=100,
        value=15,
        step=1,
    )
    utility_function = st.selectbox(r"Select utility function:", ["Log", "CRRA"])

    if utility_function == "Log":
        st.latex(r"u(c) = \text{ln}(c)")

    if utility_function == "CRRA":
        gamma = st.number_input(
            r"Select $\gamma$ (preference for smoothing):",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
        )
        st.latex(
            r"""u(c) = \begin{cases}
\frac{c^{1-\gamma}}{1-\gamma} & \gamma > 0, \gamma \neq 1 \\
\ln(c) & \gamma = 1
\end{cases}"""
        )
    else:
        gamma = None


### DEFINE CAKE EATING SOLVER
def solve_cake_finite(W, T, beta, utility_fn, gamma=None):
    V = np.zeros(T + 2)
    consumption = np.zeros(
        T + 2
    )  # add additional period to show when consumption drops to 0
    remaining_cake = np.zeros(T + 2)  # cake at the start of each period

    # using analytical solution
    if utility_fn == "Log":
        disc = beta

    elif utility_fn == "CRRA":
        if gamma == 1 or gamma == 0:  # By definition gamma > 0 and gamma != 1
            disc = beta
        else:
            disc = beta ** (1 / gamma)

    if beta < 1:
        numerator = W
        denominator = (1 - disc ** (T + 1)) / (1 - disc)
        c_0 = numerator / denominator
    else:
        c_0 = W / (T + 1)

    consumption[0] = c_0
    remaining_cake[0] = W  # initial cake size
    remaining_cake[1] = W - consumption[0]

    for t in range(1, T + 2):  # 1 through T+2 periods, but T+2 should be 0
        if t <= T:
            consumption[t] = disc * consumption[t - 1]
            remaining_cake[t + 1] = remaining_cake[t] - consumption[t]
        elif t == T + 1:
            # remaining cake should at T+1 should be 0, so cons also 0.
            consumption[t] = 0
            if -0.001 < remaining_cake[t] < 0.001:
                remaining_cake[t] = 0
            else:
                # Display an error message if final remaining cake calculation out of expected range.
                st.error(
                    "Error: Final remaining cake calculation out of expected range."
                )

    return consumption, remaining_cake


consumption, remaining_cake = solve_cake_finite(100, T, beta, utility_function, gamma)


pio.templates.default = "my_streamlit"


def plot_consumption_plotly(
    consumption, remaining_cake, T, beta, utility_fn, gamma=None
):
    # Create subplots and specify we want two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add consumption series
    fig.add_trace(
        go.Scatter(
            x=list(range(T + 2)),
            y=consumption,
            mode="lines",
            name="Consumption",
            line=dict(color="green", width=2),
            hovertemplate="t = %{x}<br>c<sub>t</sub> = %{y:.1f}<extra></extra>",
            showlegend=False,
        ),
        secondary_y=False,
    )

    # Add remaining cake series
    fig.add_trace(
        go.Scatter(
            x=list(range(T + 2)),
            y=remaining_cake,
            mode="lines",
            name="Remaining Cake",
            line=dict(color="blue", width=2, dash="dash"),
            hovertemplate="t = %{x}<br>w<sub>t</sub> = %{y:.1f}<extra></extra>",
            showlegend=False,
        ),
        secondary_y=True,
    )

    # Add vertical line and text for T
    fig.add_vline(x=T, line=dict(color="grey", dash="dash", width=2))
    fig.add_annotation(
        x=T * 1.03,
        y=consumption[0],
        text=f"t={T}",
        showarrow=False,
        xshift=10,
        font=dict(color="grey", size=15),
    )

    # x axis limit
    if T < 10:
        x_ub = T + 1.2
    else:
        x_ub = T * 1.12

    # Add horizontal line shape
    fig.add_shape(
        type="line",
        x0=-0.5,  # Start line at the beginning of the x-axis range
        y0=1.1 * max(consumption),
        x1=x_ub,  # End line at the end of the x-axis range
        y1=1.1 * max(consumption),
        line=dict(color="black", width=2),
        xref="x",  # Refer to the x-axis
        yref="y",  # Use "y2" if the line should refer to the right y-axis
    )

    # Layout settings - in addition to my_streamlit default theme from custom thm module
    fig.update_layout(
        width=600,  # Width in pixels
        height=400,
        margin=dict(autoexpand=True, l=30, r=40, t=30, b=40, pad=0),
        font=dict(family="Sans-Serif"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=dict(
            text=f"<b>Cake Eating with {utility_fn} Utility</b>",
            font=dict(size=20),
            x=0.5,
            xanchor="center",
            xref="paper",
            y=0.98,
            yanchor="top",
            yref="container",
        ),
        hovermode="closest",
        hoverlabel=dict(
            font=dict(family="Sans-Serif", color="black"),
            bgcolor="rgb(255, 255, 255)",  # rgb(0, 0, 0, 0)
            bordercolor="black",
            namelength=-1,
        ),
        xaxis=dict(
            range=[-0.5, x_ub],
            showgrid=False,
            zeroline=False,
            autorange=False,
            title=dict(text="<b>period t</b>", font=dict(size=16)),
        ),
        yaxis=dict(
            range=[0, 1.1 * max(consumption)],
            autorange=False,
            showgrid=False,
            title=dict(
                text="<b>Consumption (cₜ)</b>",
                font=dict(size=16, color="green"),
            ),
        ),
        yaxis2=dict(
            range=[0, 1.1 * max(remaining_cake)],
            autorange=False,
            showgrid=False,
            title=dict(
                text="<b>Remaining Cake (wₜ)</b>",
                font=dict(size=16, color="blue"),
            ),
        ),
    )

    return fig


chart_fig = plot_consumption_plotly(
    consumption, remaining_cake, T, beta, utility_function, gamma
)

# Plot chart
with chart_col:
    st.plotly_chart(chart_fig, theme=None, use_container_width=True)


### PREPARE DATA FOR TABLE
# Convert data to DataFrame
data_dict = {"t": range(T + 2), "c_t": consumption, "w_t": remaining_cake}
data_df = pd.DataFrame(data_dict)


# Define a function to display the table
def display_table(df, N=5):
    col_config = {
        "c_t": st.column_config.NumberColumn(format="%.2f"),
        "w_t": st.column_config.NumberColumn(format="%.2f"),
    }

    if T < 10:
        st.dataframe(df, hide_index=True, column_config=col_config)
    else:
        show_all = st.checkbox("Show all periods")
        if show_all:
            st.dataframe(df, hide_index=True, column_config=col_config)
        else:
            col_t1, col_t2 = st.columns((1, 1))
            with col_t1:
                st.markdown("First 5 periods:")
                st.dataframe(df.head(N), hide_index=True, column_config=col_config)
            with col_t2:
                st.markdown("Last 5 periods:")
                st.dataframe(df.tail(N), hide_index=True, column_config=col_config)


pie_col, table_col = st.columns((1, 0.7))


### BUILD ANIMATED PIE CHART
def pie_chart_animated(consumption, remaining_cake, T):
    fig = go.Figure()

    def create_title(t_title):
        font_settings = dict(family="Sans-Serif", size=16)

        if t_title <= T:
            rem_cake_label = remaining_cake[t_title + 1]
        else:
            rem_cake_label = 0

        title_dict = dict(
            text=f"<b>t = {t_title}<br>"
            + f"Σc<sub>t</sub> = {sum(consumption[: t_title + 1]):.2f},"
            + f"w<sub>t+1</sub> = {rem_cake_label:.2f}</b>",
            y=0.9,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=font_settings,
        )

        return title_dict

    # Make data
    def make_pie_data(consumption, remaining_cake, T, t):
        values_eaten = list(consumption[: t + 1])

        # filter to not show values for slices below 4
        labels_eaten = [
            f"c_{i}<br>{consumption[i]:.1f}" if consumption[i] > 4 else f"c_{i}"
            for i in range(t + 1)
        ]
        # filter to not show any label for slices below 1
        # empty space doesn't work, since plotly treats identical labels as a single trace
        # create labels with invisible character * slice number to keep them unique but invisible
        invisible_char = "\u200B"

        labels_eaten = [
            labels_eaten[i] if consumption[i] > 1 else "\u200B" * i
            for i in range(t + 1)
        ]

        hovertext_eaten = [f"c_{i} = {consumption[i]:.1f}" for i in range(t + 1)]

        if t < T:
            labels = labels_eaten + [f"w_{t+1}<br>{remaining_cake[t+1]:.1f}"]
            value_remaining = [remaining_cake[t + 1]]
            hovertext = hovertext_eaten + [f"w_{t+1} = {remaining_cake[t+1]:.1f}"]
        else:
            labels = labels_eaten
            hovertext = hovertext_eaten
            value_remaining = []

        values = values_eaten + value_remaining
        pull_values = [0.05] * len(values_eaten) + [0] * len(value_remaining)
        colors = ["green"] * len(values_eaten) + ["blue"] * len(value_remaining)

        return dict(
            labels=labels,
            values=values,
            pull=pull_values,
            marker=dict(colors=colors),
            hovertext=hovertext,  # Set custom hovertext for each slice
            hoverinfo="text",  # Ensure that the custom hovertext is used
            textinfo="label",
            textposition="inside",
        )

    # Make first frame
    fig.add_trace(go.Pie(**make_pie_data(consumption, remaining_cake, T, t=0)))

    fig.update_traces(
        textinfo="label",
        # texttemplate="%{label}<br>%{value:.1f}", # this is helpful without conditional labels
        hoverinfo="label",
        direction="clockwise",
        sort=False,
        showlegend=False,
    )

    fig.update_layout(
        margin=dict(t=80, l=0, b=0, r=0),
        title=create_title(0),
        font=dict(family="Sans-Serif"),
    )

    # Define slider properties
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Time:",
            "visible": False,  # Don't show current value to save space
            "xanchor": "right",
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 50, "t": 50},
        "len": 0.9,
        "x": 0.0,
        "y": 0.0,  # Adjusted slider vertical position
        "pad": {"b": 20, "t": 0},  # Reduced top padding
        "steps": [],
    }

    # Make frames for animation and slider
    frames = []

    for t_frame in range(T + 1):
        frame = go.Frame(
            data=[go.Pie(**make_pie_data(consumption, remaining_cake, T, t_frame))],
            name=str(t_frame),
            layout=go.Layout(title=create_title(t_frame)),
        )

        frames.append(frame)

        slider_step = {
            "args": [
                [str(t_frame)],
                {
                    "frame": {"duration": 300, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 300},
                },
            ],
            "label": "" if t_frame % 3 != 0 else str(t_frame),
            "method": "animate",
        }

        sliders_dict["steps"].append(slider_step)

    fig["frames"] = frames
    fig["layout"]["sliders"] = [sliders_dict]

    # Define animation button properties
    play_button = {
        "args": [
            [str(t) for t in range(0, T + 1)],
            {
                "frame": {"duration": 500, "redraw": True},
                "fromcurrent": True,
                "transition": {
                    "duration": 300,
                    "easing": "quadratic-in-out",
                },
            },
        ],
        "label": "Play",
        "method": "animate",
    }

    pause_button = {
        "args": [
            [None],
            {
                "frame": {"duration": 0, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 0},
            },
        ],
        "label": "Pause",
        "method": "animate",
    }

    # Define reset button properties
    reset_button = {
        "args": [
            [str(0)],
            {
                "frame": {"duration": 0, "redraw": True},
                "mode": "immediate",
                "transition": {"duration": 0},
            },
        ],
        "label": "Reset to 0",
        "method": "animate",
    }

    fig["layout"]["updatemenus"] = [
        {
            "buttons": [play_button, pause_button],
            "direction": "left",
            "pad": {"r": 20, "t": 0, "b": 10},
            "showactive": False,
            "type": "buttons",
            "x": 0.0,
            "xanchor": "left",
            "y": 1.1,
            "yanchor": "bottom",
        },
        {
            "buttons": [reset_button],
            "direction": "left",
            "pad": {"r": 20, "t": 25},
            "showactive": False,
            "type": "buttons",
            "x": 0.0,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "bottom",
        },
    ]

    return fig


# PLOT ANIMATED PIE CHART
with pie_col:
    st.plotly_chart(
        pie_chart_animated(consumption, remaining_cake, T),
        use_container_width=True,
    )


with table_col:
    # display data table
    st.markdown("#### Chart data")
    display_table(data_df)

s0, c02, s1 = utl.wide_col()

# Interesting takeaways - skipping since charts are self explanatory
# with c02:
#     st.markdown("#### Interesting takeaways")
#     with st.expander("Click to expand."):
#         st.markdown(
#             r"""
#         1. TBD <br>
#             $Formula \; TBD$
#         2. TBD.
#             """,
#             unsafe_allow_html=True,
#         )


with c02:
    st.header("2. Interpretation and trivial cases")
    st.markdown("#### Parameter Interpretation")

    st.markdown(
        r"""
         $\beta$ indicates your **impatience** - you feel like eating the same amount of cake tomorrow will give you less happiness than today. $u(c)$ is your **utility** of consuming $c$ amount of cake.
        Utility function usually has a concave shape, which indicates that each bite of cake gives you less happiness than the previous one. With CRRA utility, $\gamma$ modifies the curvature of the function
        and thus indicates the preference for **consumption smoothing** (risk aversion).<br>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        r"""
        You can see that these factors go in opposite directions - because you're impatient, you'd prefer to it the cake faster, but because of diminishing marginal utility, you'd prefer to eat it slower.<br>
        """,
    )

    st.markdown("#### Trivial Cases")
    st.markdown(
        r"""
        The only interesting case is when you're impatient and your utility is concave (i.e., exhibits marginal utility is diminishing). Otherwise:"""
    )
    st.markdown(
        r"""
        1. If your utility was linear ($u(c) = c$) and you were perfectly patient ($\beta = 1$), then you wouldn't care at all which day how much to eat - any consumption plan would be optimal.<br>
        2. If your utility was linear ($u(c) = c$), but you were impatient ($\beta<1$), then you'd eat the entire cake on the first day.<br>
        3. If your utility was concave, but you were perfectly patient ($\beta = 1$), then you'd eat the same amount of cake each day to perfectly smoothen your consumption.<br>
        
                """,
        unsafe_allow_html=True,
    )

    st.markdown("#### When does cake eating take place?")
    st.markdown(
        r"""
        Looking at the law of motion (how the remaining cake relates to the remaining cake and consumption in the prior period), it is important to understand when does the cake eating take place.<br>
        $w_{t+1} = w_t - c_t$<br>
        This law of motion means that you start the period $t$ with remaining cake $w_t$. You consume $c_t$ throughout that period and then the remaining cake is calculated for the next period as $w_t - c_t$.<br>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        r"""
        This will be particularly relevant when thinking about depreciation and interest rates in the following sections. For example:<br>
        If cake depreciates at the end of period $t$, then $w_{t+1} = (1-\delta)(w_t - c_t)$.<br>
        If cake depreciates at the start of period $t$, then $w_{t+1} = (1-\delta)w_t - c_t$.<br>
        Same logic also applies to the case with interest rates, so be attentive whether the interest is calculated at the start or the end of the period.
    """,
        unsafe_allow_html=True,
    )


s0, c03, s1 = utl.wide_col()

with c03:
    # st.markdown("#### Consumption and Remaining Cake Formulas")
    st.header("3. Deriving formulas")
    st.markdown(
        r"""Derivation is demonstrated for $u(c)=\text{log}(c)$ case.<br>
                Solution with CRRA utility follows the same but $\beta$ becomes $\beta^{1/\gamma}$ through Euler equation.<br>
                A concise explanation can be found below:""",
        unsafe_allow_html=True,
    )

    st.link_button(
        "Computational Economics with Python",
        "https://juejung.github.io/jdocs/Comp/html/Slides_Optimization_2_Cake.html",
        type="secondary",
    )

    col_c, col_w = st.columns((1, 1))

    with col_c:
        st.latex(r"\text{Consumption } c_t")

        st.latex(
            r"c_0 = \frac{W}{\sum_{t=0}^{T} \beta^t} = W \frac{1-\beta}{1-\beta^{T+1}}"
        )
        st.latex(r"c_1 = \beta c_0")
        st.latex(r"c_2 = \beta c_1 = \beta^2 c_0")
        st.latex("...")
        st.latex(r"c_{T-1} = \frac{1}{\beta}c_T = \frac{1}{\beta}w_T")
        st.latex(r"c_{T} = \beta^{T} c_0 = w_{T} ")

    with col_w:
        st.latex(r"\text{Remaining cake } w_t")
        st.latex("w_0 = W")
        st.latex(
            r"w_1 = w_0 - c_0 = w_0\frac{ \sum_{t=1}^T \beta^t}{\sum_{t=0}^T \beta^t}"
        )
        st.latex(r"w_2 = w_1 - c_1 = w_1\frac{\beta }{1+\beta} ")
        st.latex("...")
        st.latex(r"w_{T} = w_{T-1} - c_{T-1} = w_{T-1}\frac{\beta}{1+\beta}")
        st.latex(r"w_{T+1} = 0")

s0, c04, s1 = utl.wide_col()

with c04:
    st.header("4. Theory references")
    st.write("Main reference:")
    st.link_button(
        "Computational Economics with Python - Constrained Optimization (Prof. Juergen Jung)",
        "https://juejung.github.io/jdocs/Comp/html/Slides_Optimization_2_Cake.html",
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

    st.write("Finite case with CRRA utility and interest rate:")
    st.link_button(
        "Dynamic economics in Practice, pp. 10-22 (Prof. Monica Costa Dias and Cormac O'Dea)",
        "https://ifs.org.uk/sites/default/files/output_url_files/Dynamic%252520Economics%252520slides%252520-%252520handout.pdf",
        type="primary",
    )

    st.write("Video explanation finite and infite horizon with log utility:")
    st.link_button(
        "Dynamic Programming Economics YouTube (by EconJohn)",
        "https://youtube.com/playlist?list=PLLAPgKPWbsiQ0Ejh-twYC3Fr8_WA9BKCc&si=jSGmuYGe_MtnvtTP",
        type="secondary",
    )
