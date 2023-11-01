import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from st_pages import show_pages_from_config

import src.scripts.plot_themes as thm
import src.scripts.utils as utl

### PAGE CONFIGS ###

st.set_page_config(
    page_title="PhD Macroeconomics - Glossary",
    page_icon="🌐",
    layout="wide",
)

utl.local_css("src/styles/styles_pages.css")

random_seed = 0
s1, c1, s2 = utl.wide_col()

### PAGE START ###
# Dashboard header
with c1:
    st.title("Definitions for Macroeconomics")
    st.header("Finding the way through macro lingo")
    st.divider()
    # st.markdown(
    #     r"""<h3>Let's find our way through the macro slang</h3>""",
    #     unsafe_allow_html=True,
    # )


s1, c2, s2 = utl.wide_col()

with c2:
    st.markdown(
        r"""
    <h5>1. Intro to dynamic programming</h5>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("Click to expand"):
        st.markdown(
            r"""

        Sequence problem:<br>
        $\max_{\{c_t\}_{t=0}^\infty} \sum_{t=0}^{\infty} \beta^t u(c_t)$<br>

        Constraints (feasible paths):<br>
        $k_{t+1} = k_t - c_t$ (law of motion, transition law, capital accumulation rule - used interchangeably)<br>
        $c_t \geq 0, k_{t+1}\geq 0 \text{ for all } t = 1, 2, ...$<br>
        $k_0 = \bar{k} \text{ given}$
        
        Euler equation:<br>
        $u'(c_t) = \beta u'(c_{t+1})$<br>

        Recursive formulation:<br>
        $V(k_t) = \max_{c_t} u(c_t) + \beta V(k_{t+1}) = \max_{k_{t+1}} u(k_{t+1} - k_t) + \beta V(k_{t+1}) $<br>

        First order condition:<br>
        $u'(c_t) = \beta V'(k_{t+1})$<br>
    
        Envelope condition (solve for t, then go forward to t+1 and plug result back in FOC):<br>
        $V'(k_t) = u'(c_t)\times c_t'(k_t)$<br>
        $V'(k_{t+1}) = u'(c_{t+1})\times c_{t+1}'(k_{t+1})$<br>

        Transversality condition:<br>
        $\lim_{t \to \infty} \beta^t V'(k_{t+1}^*)\times k_t^* = 0$<br>
        
""",
            unsafe_allow_html=True,
        )

    st.markdown(
        r"""
        <h5>2. Bellman equation in detail</h5>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Click to expand"):
        st.markdown(
            r"""
        
        From before, rename $k_t$ to $x$ (state variable) and $k_{t+1}$ to $y$ (control variable).<br>
        $u(k_{t+1} - k_t)$ becomes $F(x, y)$.<br>

        Bellman equation:<br>
        $V(x) = \max_{y} F(x, y) + \beta V(y)$<br>
        s.t. $y \in \Gamma(x)$<br>

        Optimal solution ($x$ is a state variable):<br>
        Value function at the optimum (maximizing RHS of Bellman equation):<br>
        $V(x) = F(x,y) + \beta V(y)$<br>
        Policy correspondence (function, if single valued):<br>
        $G(x) = \{y \in \Gamma(x): V(x) = F(x,y) + \beta V(y)\}$<br>
        

        Solution methods to the Bellman equation:<br>
        [Useful source for examples with different solution methods](https://web.uvic.ca/~kumara/econ552/lecture3.pdf) <br>
            1. Value function iteration (VFI) - fixed point problem<br>
            2. Guess-and-verify (Value or Policy function)<br>
            3. Policy function iteration (PFI) - additional material<br>

        Solving Bellman equation as a fixed point problem:<br>
        $T$ is called Bellman operator.<br>
        $T(f)(x) = \text{sup}_y F(x, y) + \beta f(y)$<br>
        $T(f) = f$  (fixed point)<br>
        $T(V_t(x)) = V_{t+1} (x)$<br>

        Start with arbitrary $f_0$ and iterate:<br>
        $f_{n+1} = T(f_{n})(x) = \text{sup}_y F(x, y) + \beta f_n(y)$<br>
        
        Claim: $f_n$ converges to $f$ under certain conditions.
        Need to show that $T$ is a contraction and apply contraction mapping theorem to prove that $T$ has a unique fixed point.<br>


        Contraction mapping theorem: If $(S, r)$ is a complete metric space and
        $T : S \rightarrow S$ is a contraction mapping with modulus $\beta$, then:<br>
        a. $T$ has exactly one fixed point $V$ in $S$;<br>
        b. For any $V_0 \in S, r(T^n V_0, V) \leq \beta^n \rho (V_0, V), n = 0, 1, 2, ...$<br>
        
        Solving Bellman equation by guess-and-verify (value function):<br>
        1. Guess a solution form, e.g., $V(x) = N+M \text{log}(x)$<br>
        2. Take FOC and get expression for $y(x, M)$<br>
        3. Substitute $y(x, M)$ into Bellman equation, i.e., $V(x) = V(y (x, M))$<br>
        $N + M \text{log}(x) = F(x, y) + \beta V(y(x, M))$<br>
        4. Split terms into those with $M$ and those with $N$.<br>
        5. Solve for $M$ and $N$ as a system of two equations.<br>
        6. Plug in solutions $M$ and $N$ into the optimal policy function $y(x)$.

        See Section 3 below for an example of guess-and-verify applied to the growth model.

        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        r"""
        <h5>3. Neoclassical Growth Model</h5>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Click to expand"):
        st.markdown(
            r"""
        Two sides of the market:<br>
        A representative household that chooses consumption to maximize utility.<br>
        A representative firm chooses capital and labor to maximize profits.<br>

        The household actually owns the firm and receives all its profits.<br>
        The firm doesn't own neither labor, nor capital - it hires household labor and rents its capital.<br>
        The household needs to work for the firm to get the salary and buy consumption or invest in capital that will be rented to the firm.<br>
                
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            r"""
        Firm's profit maximization:<br>
        $\pi = \max_{k_t, l_t} \sum_{t=0}^\infty p_t (y_t - w_tl_t - r_t k_t)$<br>
        Firm's production function:<br>
        $y_t = f(k_t, l_t)$<br>
                """,
            unsafe_allow_html=True,
        )

        st.markdown(
            r"""
        Household lifetime utility maximization:<br>
        $\max_{c_t}\sum_{t=0}^\infty \beta^t u(c_t)$<br>
        Household lifetime budget constraint:<br>
        $\sum_{t=0}^\infty p_t (c_t + i_t) \leq \sum_{t=0}^\infty p_t (w_t+r_t k_t) + \pi$<br>
        Household consumption constraint:<br>
        $c_t + i_t \leq f(k_t)$<br>
        Household investment in firm's capital:<br>
        $k_{t+1} = (1-\delta)k_t + i_t$<br>
        $k_0$ given
                        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            r"""Notation:<br>
         $y_t$ - output.<br>
        $c_t$ - consumption.<br>
        $i_t$ - investment.<br>
        $k_t$ - capital.<br>
        $l_t$ - labor.<br>
        $p_t$ - price of consumption. Output at time zero is numeraire, i.e., $p_0 \equiv 1$.<br>
        $w_t$ - wage in terms of time-t output.<br>
        $w_t p_t$ - price of labor (household's labor income).<br>
        $r_t$ - rental rate of capital in terms of time-t output.<br>
        $r_t p_t$ - price of capital (household's capital rental income).<br>
        $\pi$ - firm's profit (household's income from firm's production).<br>
        $R_{t+1} = \frac{1}{q_{t+1}}$ - interest rate, which is the inverse of next period output price $q_{t+1}$.<br>
        $\delta$ - capital depreciation rate.<br>
        $\beta$ - utility discount factor.<br>
        $u(c_t)$ - utility function.<br>
        $f(k_t, l_t)$ - production function.<br>
        

            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            r"""
        Equilibrium is defined by an allocation {$Y_t, C_t, K_t, L_t$} and prices {$p_t, w_t, r_t$}, s.t.:<br>
            1. Allocation solves firm's problem and household's problem, taking prices as given.<br>
            2. Markets clear: $Y_t = C_t + I_t$, $L_t = 1$
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown(
            r"""
            Solving the firm's problem:<br>
            $r=f_K(k, l)$<br>
            $w=f_L(k, l)$<br>
            $f(K, L) =f_K(K, L)k - f_L(K, L)l$ (Euler's Theorem if $f$ is diff'able and h.d.1)<br>
            $\pi = \sum_{t=0}^\infty p_t (f(k_t, l_t) - rk_t - wl_t)$ = 0<br>
            
            Solving the household's problem:<br>
            $u'(c_t) = \beta \frac{p_t}{p_{t+1}} u'(c_{t+1}) = \beta R_{t+1} u'(c_{t+1})$ (Euler equation) <br>
            $R_{t+1} \equiv \frac{p_t}{p_{t+1}} \frac{1}{q_{t+1}}$<br>
            $R_{t+1} = r_{t+1} + 1 - \delta$ (no arbitrage condition)<br>

            At the equilibrium:<br>
            $f_K(k, 1) = r$<br>
            $k_{t+1} = f(k_t, 1) - c_t + (1 - \delta k_t)$<br>

            To finish the solution:
            Need a given $k_0$ and <br>
            Define a steady state, s.t. at some $t$, $k_{t+1} = k_t = k_{ss}$ and $c_{t+1} = c_t = c_{ss}$, then:<br>
            $k_{ss} = f(k_{ss}, 1) - c_{ss} + (1 - \delta k_{ss})$<br>
            $c_{ss} = f(k_{ss}, 1) - \delta k_{ss}$<br>
            
            Phase diagram:
            """,
            unsafe_allow_html=True,
        )
        st.image("src/images/growth_phase_diagram.png", width=600)

        st.divider()

        st.markdown(
            r"""

        Solve the growth model with guess-and-verify:<br>
        $u(c_t) = \text{log}(c_t)$, $f(k_t) = k_t^\alpha$, $\delta=1$, $c_t = k_t^\alpha - k_{t+1}$<br>

        Bellman equation:<br>
        $V(k_t) = u(c_t) + \beta V(k_{t+1})$<br>        

        1. Guess $V(k_t) = N+M \text{log}(k_t)$
        2. From FOC and budget constraint:<br>
            $\lambda_t = \frac{1}{c_t} = \frac{\beta M}{k_{t+1}} = \frac{\beta M}{k_t^\alpha - c_t}$<br>
            $c_t = k_t^\alpha - k_{t+1} = k_t^\alpha - \frac{\beta M}{1+\beta M} k_t^\alpha =
            \frac{1}{1 + \beta M} k_t^\alpha$<br> 
            $k_{t+1} = \frac{\beta M}{1+\beta M} k_t^\alpha $<br>
        
        3. $N + M \text{log}(k_t) = u(c_t) + \beta V(k_{t+1})$<br>
            $N + M \text{log}(k_t) = \text{log}(\frac{k_t^\alpha}{1+\beta M}) + \beta V(k_{t+1})$<br>
            $N + M \text{log}(k_t) = \text{log}(\frac{k_t^\alpha}{1+\beta M}) + \beta (N + M \text{log} (\frac{\beta M k_t^\alpha}{1+ \beta M}))$
            

        4. Separate $M$ and $N$:<br>
            $M = \alpha + \beta M \alpha$<br>
            $N = \text{log}(\frac{1}{1+\beta M}) + \beta (N + M \text{log} (\frac{\beta M}{1+ \beta M}))$<br>

        5. Find solution for $M$ and $N$:<br>
            $M = \frac{\alpha}{1-\beta \alpha}$<br>
            $N = \frac{1}{1 - \beta} (\text{log}(1 - \beta \alpha) + \frac{\beta \alpha}{1- \beta \alpha} \text{log} (\beta \alpha) )$<br>

        6. Plug in solutions $M$ and $N$ into the optimal policy function:<br>
        $k_{t+1} = \beta \alpha k_t^\alpha$<br>
        $c_t = (1-\beta \alpha) k_t^\alpha$<br>

""",
            unsafe_allow_html=True,
        )

#     st.markdown(
#         r"""
#         <h5>4. Markets and Exchange Economy</h5>
#         """,
#         unsafe_allow_html=True,
#     )

#     with st.expander("Click to expand"):
#         st.markdown(
#             r"""
# """,
#             unsafe_allow_html=True,
#         )
