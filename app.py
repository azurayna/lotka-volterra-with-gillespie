import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def gillespie_lv(prey_init, pred_init, a, b, c, d, max_time=50):
    """
    Gillespie algorithm for Lotka-Volterra model.
    
    prey_init: initial prey population
    pred_init: initial predator population
    a: prey birth rate
    b: predation rate (predator eats prey)
    c: predator death rate
    d: predator reproduction rate per eaten prey
    max_time: simulation time limit
    
    Returns time points and population arrays.
    """
    prey = prey_init
    pred = pred_init
    
    t = 0
    times = [0]
    prey_pop = [prey]
    pred_pop = [pred]
    
    while t < max_time and prey > 0 and pred > 0:
        # rates for each event
        rates = [
            a * prey,         # prey birth
            b * prey * pred,  # predation event
            c * pred,         # predator death
            d * prey * pred   # predator reproduction
        ]
        total_rate = sum(rates)
        if total_rate == 0:
            break
        
        # time until next event
        dt = np.random.exponential(1 / total_rate)
        t += dt
        
        # choose which event occurs
        r = np.random.uniform(0, total_rate)
        if r < rates[0]:           # prey birth
            prey += 1
        elif r < rates[0] + rates[1]:  # predation (prey dies)
            prey -= 1
        elif r < rates[0] + rates[1] + rates[2]:  # predator death
            pred -= 1
        else:                      # predator reproduction
            pred += 1
        
        # record populations
        times.append(t)
        prey_pop.append(prey)
        pred_pop.append(pred)
        
    return np.array(times), np.array(prey_pop), np.array(pred_pop)

# --- Streamlit app ---
st.title("Lotka-Volterra with Gillespie Algorithm")

# Sidebar sliders for parameters
st.sidebar.header("Parameters")
prey_init = st.sidebar.number_input("Initial Prey Population", min_value=1, value=50, step=1)
pred_init = st.sidebar.number_input("Initial Predator Population", min_value=1, value=20, step=1)

a = st.sidebar.slider("Prey Birth Rate (a)", 0.0, 2.0, 1.0, 0.01)
b = st.sidebar.slider("Predation Rate (b)", 0.0, 0.1, 0.02, 0.001)
c = st.sidebar.slider("Predator Death Rate (c)", 0.0, 2.0, 1.0, 0.01)
d = st.sidebar.slider("Predator Reproduction Rate (d)", 0.0, 0.1, 0.01, 0.001)

max_time = st.sidebar.number_input("Max Simulation Time", min_value=1, value=50)

if st.button("Run Simulation"):
    times, prey_pop, pred_pop = gillespie_lv(prey_init, pred_init, a, b, c, d, max_time)
    
    # Plot results
    fig, ax = plt.subplots()
    ax.step(times, prey_pop, label="Prey", where="post")
    ax.step(times, pred_pop, label="Predator", where="post")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.legend()
    st.pyplot(fig)
    
    # Stats
    st.write(f"Final Prey Population: {prey_pop[-1]}")
    st.write(f"Final Predator Population: {pred_pop[-1]}")
    st.write(f"Total Time Simulated: {times[-1]:.2f}")
