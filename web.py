import streamlit as st
import numpy as np
from optimizers import sgd_optimization, adam_optimization, cmaes_optimization, lra_cma_optimization, bfgs_optimization, lbfgsb_optimization
from math_funcs import Ackley, Rastrigin, Rosenbrock, Fletcher, Michalewicz
from plot import plot_3d
from io import BytesIO
from utils import icon

st.set_page_config(page_title="FastOpt",
                   page_icon="🧬",
                   layout="wide")
icon.show_icon("ִֶָ𓂃 ࣪˖ ִֶָ🐇་༘࿐")

st.header(":rainbow[Functions...]")

# Initialize session state for optimizer selection
if 'optimizer_selected' not in st.session_state:
    st.session_state.optimizer_selected = []

# Sidebar for optimizer selection and input
with st.sidebar:
    st.title("🧬 FastOpt")
    st.write("This app allows you to see optimization paths using different optimizers.")
    st.divider()
    st.info("**Start here ↓**", icon="👋🏾")

    Func = st.selectbox(
        '1️⃣ Select a function to optimize',
        ['Ackley', 'Rastrigin', 'Rosenbrock', 'Fletcher', 'Michalewicz']
    )
    Optimizer = st.multiselect(
        '2️⃣ Select optimizers',
        ['SGD', 'Adam', 'CMA-ES', 'LRA-CMA', 'BFGS', 'L-BFGS-B'],
        default=st.session_state.optimizer_selected
    )
    
    # Update session state and rerun if optimizer selection changes
    if Optimizer != st.session_state.optimizer_selected:
        st.session_state.optimizer_selected = Optimizer
        st.rerun()
    
    # Initialize iteration variables with default values
    lr_rate_sgd = lr_rate_adam = sigma_cmaes = sigma_lra_cma = iterations_sgd = iterations_adam = iterations_cmaes = iterations_lra_cma = iterations_bfgs = iterations_lbfgsb = 50
    if 'SGD' in Optimizer:
        with st.expander("**SGD params**", icon="📊"):
            lr_rate_sgd = st.number_input('Enter learning rate for SGD', value=0.1)
            iterations_sgd = st.number_input('Enter number of iterations for SGD', value=50)
    if 'Adam' in Optimizer:
        with st.expander("**Adam params**", icon="📊"):
            lr_rate_adam = st.number_input('Enter learning rate for Adam', value=0.01)
            iterations_adam = st.number_input('Enter number of iterations for Adam', value=50)
    if 'CMA-ES' in Optimizer:
        with st.expander("**CMA-ES params**", icon="📊"):
            sigma_cmaes = st.number_input('Enter sigma for CMA-ES', value=1.3)
            iterations_cmaes = st.number_input('Enter number of iterations for CMA-ES', value=50)
    if 'LRA-CMA' in Optimizer:
        with st.expander("**LRA-CMA params**", icon="📊"):
            sigma_lra_cma = st.number_input('Enter sigma for LRA-CMA', value=1.3)
            iterations_lra_cma = st.number_input('Enter number of iterations for LRA-CMA', value=50)
    if 'BFGS' in Optimizer:
        with st.expander("**BFGS params**", icon="📊"):
            iterations_bfgs = st.number_input('Enter number of iterations for BFGS', value=50)
    if 'L-BFGS-B' in Optimizer:
        with st.expander("**L-BFGS-B params**", icon="📊"):
            iterations_lbfgsb = st.number_input('Enter number of iterations for L-BFGS-B', value=50)

    start_point = st.text_input('3️⃣ Enter start point (comma-separated)', '3,2')
    start_point = np.array([float(x) for x in start_point.split(',')])
    
    button_disabled = not bool(Optimizer)

    submitted = st.button("Optimize!", use_container_width=True, disabled=button_disabled)

# Define functions
dim = 2
f1 = Ackley(dim)
f2 = Rastrigin(dim)
f3 = Rosenbrock(dim)
f4 = Fletcher(dim)
f5 = Michalewicz(dim)

functions = {
    'Ackley': f1,
    'Rastrigin': f2,
    'Rosenbrock': f3,
    'Fletcher': f4,
    'Michalewicz': f5
}

cols = st.columns(len(functions))
# for col, (name, func) in zip(cols, functions.items()):
#     fig = plot_3d(func, points_by_dim=70, title='', bounds=None, 
#                   show_best_if_exists=False, save_as=None, cmap='viridis', 
#                   plot_surface=True, plot_heatmap=False)
    
#     buf = BytesIO()
#     fig.savefig(buf, format="png")
#     col.text(name)
#     col.caption('Global min: (0,0)')
#     col.image(buf)

for col, (name, func) in zip(cols, functions.items()):
    col.text(name)
    if name == 'Ackley':
        col.caption('Global min: (0,0)')
    elif name == 'Rastrigin':
        col.caption('Global min: (0,0)')
    elif name == 'Rosenbrock':
        col.caption('Global min: (1,1)')
    elif name == 'Fletcher':
        col.caption('Global min: (0,0)')
    else:
        col.caption('Global min: (2.20, 1.57)')
    col.image(f'assets/{name}.png')

st.divider()
st.header(":rainbow[Optimization Paths...]")

# Activate if submit button is push
if submitted:
    f = functions[Func]
    terminate_points = {}
    optimization_paths = []

    if 'SGD' in Optimizer:
        sgd_path, sgd_reach_min, sgd_opt_steps, sgd_end_point = sgd_optimization(f, lr=lr_rate_sgd, start_point=start_point, iterations=iterations_sgd)
        terminate_points['SGD'] = [sgd_reach_min, sgd_opt_steps, sgd_end_point]
        optimization_paths.append((sgd_path, 'SGD', 'blue'))
        
    if 'Adam' in Optimizer:
        adam_path, adam_reach_min, adam_opt_steps, adam_end_point = adam_optimization(f, lr=lr_rate_adam, start_point=start_point, iterations=iterations_adam)
        terminate_points['Adam'] = [adam_reach_min, adam_opt_steps, adam_end_point]
        optimization_paths.append((adam_path, 'Adam', 'green'))
        
    if 'CMA-ES' in Optimizer:
        cmaes_path, cmaes_reach_min, cmaes_opt_steps, cmaes_end_point = cmaes_optimization(f, start_point, sigma=sigma_cmaes, iterations=iterations_cmaes)
        terminate_points['CMA-ES'] = [cmaes_reach_min, cmaes_opt_steps, cmaes_end_point]
        optimization_paths.append((cmaes_path, 'CMA-ES', 'purple'))
    if 'LRA-CMA' in Optimizer:
        lra_cma_path, lra_cma_reach_min, lra_cma_opt_steps, lra_cma_end_point = lra_cma_optimization(f, start_point, sigma=sigma_lra_cma, iterations=iterations_lra_cma)
        terminate_points['LRA-CMA'] = [lra_cma_reach_min, lra_cma_opt_steps, lra_cma_end_point]
        optimization_paths.append((lra_cma_path, 'LRA-CMA', 'yellow'))
    if 'BFGS' in Optimizer:
        bfgs_path, bfgs_reach_min, bfgs_opt_steps, bfgs_end_point = bfgs_optimization(f, start_point, iterations=iterations_bfgs)
        terminate_points['BFGS'] = [bfgs_reach_min, bfgs_opt_steps, bfgs_end_point]
        optimization_paths.append((bfgs_path, 'BFGS', 'orange'))
    if 'L-BFGS-B' in Optimizer:
        lbfgsb_path, lbfgsb_reach_min, lbfgsb_opt_steps, lbfgsb_end_point = lbfgsb_optimization(f, start_point, iterations=iterations_lbfgsb)
        terminate_points['L-BFGS-B'] = [lbfgsb_reach_min, lbfgsb_opt_steps, lbfgsb_end_point]
        optimization_paths.append((lbfgsb_path, 'L-BFGS-B', 'red'))

    if Func == 'Ackley':
        f = f1
        fig = plot_3d(f, points_by_dim=70, title=fr"{type(f1).__name__}", bounds=None, 
                show_best_if_exists=False, save_as=None, cmap='viridis', 
                plot_surface=False, plot_heatmap=True, optimization_paths=optimization_paths)
    elif Func == 'Rastrigin':
        f = f2
        fig = plot_3d(f, points_by_dim=70, title=fr"{type(f2).__name__}", bounds=None, 
                show_best_if_exists=False, save_as=None, cmap='viridis', 
                plot_surface=False, plot_heatmap=True, optimization_paths=optimization_paths)
    elif Func == 'Rosenbrock':
        f = f3
        fig = plot_3d(f, points_by_dim=70, title=fr"{type(f3).__name__}", bounds=None, 
                show_best_if_exists=False, save_as=None, cmap='viridis', 
                plot_surface=False, plot_heatmap=True, optimization_paths=optimization_paths)
    elif Func == 'Fletcher':
        f = f4
        fig = plot_3d(f, points_by_dim=70, title=fr"{type(f4).__name__}", bounds=None, 
                show_best_if_exists=False, save_as=None, cmap='viridis', 
                plot_surface=False, plot_heatmap=True, optimization_paths=optimization_paths)
    else:
        f = f5
        fig = plot_3d(f, points_by_dim=70, title=fr"{type(f5).__name__}", bounds=None, 
                show_best_if_exists=False, save_as=None, cmap='viridis', 
                plot_surface=False, plot_heatmap=True, optimization_paths=optimization_paths)
    
    cols = st.columns(2)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf, use_column_width=True)

    optimizer_iterations = {
        'SGD': iterations_sgd,
        'Adam': iterations_adam,
        'CMA-ES': iterations_cmaes,
        'LRA-CMA': iterations_lra_cma,
        'BFGS': iterations_bfgs,
        'L-BFGS-B': iterations_lbfgsb
    }

    # Create row1 and row2 columns
    row1 = st.columns(3)
    row2 = st.columns(3)
    all_cols = row1 + row2

    for col, opt in zip(all_cols, Optimizer):
        max_iter = optimizer_iterations[opt]
        
        if terminate_points[opt][0]:
            tile = col.container(height=135)
            tile.markdown(
                f"""
                🎈 **{opt}** <br>
                :green[Likely reached the global minimum] 
                after {terminate_points[opt][1]} iterations <br>
                Terminate point: **({terminate_points[opt][2][0]:.2f}, {terminate_points[opt][2][1]:.2f})**
                """,
                unsafe_allow_html=True
            )
        elif terminate_points[opt][1] < max_iter:
            tile = col.container(height=135)
            tile.markdown(
                f"""
                🎈 **{opt}** <br>
                :red[Did not reach global minimum and got stuck] 
                after {terminate_points[opt][1]} iterations <br>
                Terminate point: **({terminate_points[opt][2][0]:.2f}, {terminate_points[opt][2][1]:.2f})**
                """,
                unsafe_allow_html=True
            )
        else:
            tile = col.container(height=135)
            tile.markdown(
                f"""
                🎈 **{opt}** <br>
                :red[Did not reach global minimum] 
                after {terminate_points[opt][1]} iterations <br>
                Terminate point: **({terminate_points[opt][2][0]:.2f}, {terminate_points[opt][2][1]:.2f})**
                """,
                unsafe_allow_html=True
            )

    # for opt, terminate_point in terminate_points.items():
    #     cols[1].markdown(f"**{opt}** terminated at **({terminate_point[0]:.4f}, {terminate_point[1]:.4f})**")

else:
    st.write("Select optimizers and click :green['Optimize!'] to see optimization paths.")
    



