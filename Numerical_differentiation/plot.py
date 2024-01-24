import numpy as np
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append('')

def plot_comparison_plotly(time_value, x1_values_list, x2_values_list ,main_title ,titles):
    num_methods = len(x1_values_list)

    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=["x1_values Comparison", "x2_values Comparison"])

    # Add traces for x1_values
    for i in range(num_methods):
        fig.add_trace(go.Scatter(x=time_value, y=x1_values_list[i], mode='lines+markers', name=titles[i], line=dict(width=0.01)),
                      row=1, col=1)

    # Add traces for x2_values
    for i in range(num_methods):
        fig.add_trace(go.Scatter(x=time_value, y=x2_values_list[i], mode='lines+markers', name=titles[i], line=dict(width=0.01)),
                      row=1, col=2)

    # Update layout
    fig.update_layout(title_text=main_title)

    # Show the plot
    fig.show()

def plot_1_zad():
    # Loading result from file
    data_euler = np.loadtxt('DZ5/plotting/1_zad/euler_method_plot.txt')

    # Separate the data
    x1_values_euler = data_euler[0, :]
    x2_values_euler  = data_euler[1, :]
    time_values_euler  = data_euler[2, :]
    title_euler = 'Eulerov postupak'

    data_backeuler = np.loadtxt('DZ5/plotting/1_zad/backward_euler_method_plot.txt')
    x1_values_backeuler= data_backeuler[0, :]
    x2_values_backeuler = data_backeuler[1, :]
    time_values_backeuler = data_backeuler[2, :]
    title_backeuler = 'Unazadni Eulerov postupak'

    data_trapezoid = np.loadtxt('DZ5/plotting/1_zad/trapezoidal_method_plot.txt')
    x1_values_trapezoid= data_trapezoid[0, :]
    x2_values_trapezoid = data_trapezoid[1, :]
    time_values_trapezoid = data_trapezoid[2, :]
    title_trapezoid = 'Trapezoidni postupak'

    data_runge_kutta = np.loadtxt('DZ5/plotting/1_zad/trapezoidal_method_plot.txt')
    x1_values_runge_kutta= data_runge_kutta[0, :]
    x2_values_runge_kutta = data_runge_kutta[1, :]
    time_values_runge_kutta = data_runge_kutta[2, :]
    title_runge_kutta = 'Runge-Kutta postupak'

    data_PECE = np.loadtxt('DZ5/plotting/1_zad/PE_CE_method_plot.txt')
    x1_values_PE_CE= data_PECE[0, :]
    x2_values_PE_CE = data_PECE[1, :]
    time_values_PE_CE = data_PECE[2, :]
    title_PE_CE = 'PE_CE, P = Euler, C=Trapezni postupak'

    data_PECE_2 = np.loadtxt('DZ5/plotting/1_zad/PE_CE_2_method_plot.txt')
    x1_values_PECE_2= data_PECE_2[0, :]
    x2_values_PECE_2E = data_PECE_2[1, :]
    time_values_PECE_2= data_PECE_2[2, :]
    title_PECE_2 = 'PE_CE_2, P = Euler, C = Obrnuti Euler'

    x1_values_list = [x1_values_euler, x1_values_backeuler, x1_values_trapezoid, x1_values_runge_kutta, x1_values_PE_CE, x1_values_PECE_2]
    x2_values_list = [x2_values_euler, x2_values_backeuler, x2_values_trapezoid, x2_values_runge_kutta, x2_values_PE_CE, x2_values_PECE_2E]
    titles_list = [title_euler, title_backeuler, title_trapezoid, title_runge_kutta, title_PE_CE, title_PECE_2]

    main_title = "Solution 1st task"
    plot_comparison_plotly(time_values_euler, x1_values_list, x2_values_list,main_title, titles_list)

def plot_2_zad():
    # Loading result from file
    data_euler = np.loadtxt('DZ5/plotting/2_zad/euler_method_plot.txt')

    # Separate the data
    x1_values_euler = data_euler[0, :]
    x2_values_euler  = data_euler[1, :]
    time_values_euler  = data_euler[2, :]
    title_euler = 'Eulerov postupak'

    data_backeuler = np.loadtxt('DZ5/plotting/2_zad/backward_euler_method_plot.txt')
    x1_values_backeuler= data_backeuler[0, :]
    x2_values_backeuler = data_backeuler[1, :]
    time_values_backeuler = data_backeuler[2, :]
    title_backeuler = 'Unazadni Eulerov postupak'

    data_trapezoid = np.loadtxt('DZ5/plotting/2_zad/trapezoidal_method_plot.txt')
    x1_values_trapezoid= data_trapezoid[0, :]
    x2_values_trapezoid = data_trapezoid[1, :]
    time_values_trapezoid = data_trapezoid[2, :]
    title_trapezoid = 'Trapezoidni postupak'

    data_runge_kutta = np.loadtxt('DZ5/plotting/2_zad/trapezoidal_method_plot.txt')
    x1_values_runge_kutta= data_runge_kutta[0, :]
    x2_values_runge_kutta = data_runge_kutta[1, :]
    time_values_runge_kutta = data_runge_kutta[2, :]
    title_runge_kutta = 'Runge-Kutta postupak'

    data_PECE = np.loadtxt('DZ5/plotting/2_zad/PE_CE_method_plot.txt')
    x1_values_PE_CE= data_PECE[0, :]
    x2_values_PE_CE = data_PECE[1, :]
    time_values_PE_CE = data_PECE[2, :]
    title_PE_CE = 'PE_CE, P = Euler, C=Trapezni postupak'


    data_PECE_2 = np.loadtxt('DZ5/plotting/2_zad/PE_CE_2_method_plot.txt')
    x1_values_PECE_2= data_PECE_2[0, :]
    x2_values_PECE_2E = data_PECE_2[1, :]
    time_values_PECE_2= data_PECE_2[2, :]
    title_PECE_2 = 'PE_CE_2, P = Euler, C = Obrnuti Euler'

    x1_values_list = [x1_values_euler, x1_values_backeuler, x1_values_trapezoid, x1_values_runge_kutta, x1_values_PE_CE, x1_values_PECE_2]
    x2_values_list = [x2_values_euler, x2_values_backeuler, x2_values_trapezoid, x2_values_runge_kutta, x2_values_PE_CE, x2_values_PECE_2E]
    titles_list = [title_euler, title_backeuler, title_trapezoid, title_runge_kutta, title_PE_CE, title_PECE_2]


    main_title = "Solution 2nd task"
    plot_comparison_plotly(time_values_euler, x1_values_list, x2_values_list,main_title, titles_list)

def plot_3_zad():
    # Loading result from file
    data_euler = np.loadtxt('DZ5/plotting/3_zad/euler_method_plot.txt')

    # Separate the data
    x1_values_euler = data_euler[0, :]
    x2_values_euler  = data_euler[1, :]
    time_values_euler  = data_euler[2, :]
    title_euler = 'Eulerov postupak'

    data_backeuler = np.loadtxt('DZ5/plotting/3_zad/backward_euler_method_plot.txt')
    x1_values_backeuler= data_backeuler[0, :]
    x2_values_backeuler = data_backeuler[1, :]
    time_values_backeuler = data_backeuler[2, :]
    title_backeuler = 'Unazadni Eulerov postupak'

    data_trapezoid = np.loadtxt('DZ5/plotting/3_zad/trapezoidal_method_plot.txt')
    x1_values_trapezoid= data_trapezoid[0, :]
    x2_values_trapezoid = data_trapezoid[1, :]
    time_values_trapezoid = data_trapezoid[2, :]
    title_trapezoid = 'Trapezoidni postupak'

    data_runge_kutta = np.loadtxt('DZ5/plotting/3_zad/trapezoidal_method_plot.txt')
    x1_values_runge_kutta= data_runge_kutta[0, :]
    x2_values_runge_kutta = data_runge_kutta[1, :]
    time_values_runge_kutta = data_runge_kutta[2, :]
    title_runge_kutta = 'Runge-Kutta postupak'
    
    data_PECE = np.loadtxt('DZ5/plotting/3_zad/PE_CE_method_plot.txt')
    x1_values_PE_CE= data_PECE[0, :]
    x2_values_PE_CE = data_PECE[1, :]
    time_values_PE_CE = data_PECE[2, :]
    title_PE_CE = 'PE_CE, P = Euler, C=Trapezni postupak'

    data_PECE_2 = np.loadtxt('DZ5/plotting/3_zad/PE_CE_2_method_plot.txt')
    x1_values_PECE_2= data_PECE_2[0, :]
    x2_values_PECE_2E = data_PECE_2[1, :]
    time_values_PECE_2= data_PECE_2[2, :]
    title_PECE_2 = 'PE_CE_2, P = Euler, C = Obrnuti Euler'

    x1_values_list = [x1_values_euler, x1_values_backeuler, x1_values_trapezoid, x1_values_runge_kutta, x1_values_PE_CE, x1_values_PECE_2]
    x2_values_list = [x2_values_euler, x2_values_backeuler, x2_values_trapezoid, x2_values_runge_kutta, x2_values_PE_CE, x2_values_PECE_2E]
    titles_list = [title_euler, title_backeuler, title_trapezoid, title_runge_kutta, title_PE_CE, title_PECE_2]
    
    main_title = "Solution 3rd task"
    plot_comparison_plotly(time_values_euler, x1_values_list, x2_values_list,main_title, titles_list)

def plot_4_zad():
    # Loading result from file
    data_euler = np.loadtxt('DZ5/plotting/4_zad/euler_method_plot.txt')

    # Separate the data
    x1_values_euler = data_euler[0, :]
    x2_values_euler  = data_euler[1, :]
    time_values_euler  = data_euler[2, :]
    title_euler = 'Eulerov postupak'

    data_backeuler = np.loadtxt('DZ5/plotting/4_zad/backward_euler_method_plot.txt')
    x1_values_backeuler= data_backeuler[0, :]
    x2_values_backeuler = data_backeuler[1, :]
    time_values_backeuler = data_backeuler[2, :]
    title_backeuler = 'Unazadni Eulerov postupak'

    data_trapezoid = np.loadtxt('DZ5/plotting/4_zad/trapezoidal_method_plot.txt')
    x1_values_trapezoid= data_trapezoid[0, :]
    x2_values_trapezoid = data_trapezoid[1, :]
    time_values_trapezoid = data_trapezoid[2, :]
    title_trapezoid = 'Trapezoidni postupak'

    data_runge_kutta = np.loadtxt('DZ5/plotting/4_zad/trapezoidal_method_plot.txt')
    x1_values_runge_kutta= data_runge_kutta[0, :]
    x2_values_runge_kutta = data_runge_kutta[1, :]
    time_values_runge_kutta = data_runge_kutta[2, :]
    title_runge_kutta = 'Runge-Kutta postupak'

    data_PECE = np.loadtxt('DZ5/plotting/4_zad/PE_CE_method_plot.txt')
    x1_values_PE_CE= data_PECE[0, :]
    x2_values_PE_CE = data_PECE[1, :]
    time_values_PE_CE = data_PECE[2, :]
    title_PE_CE = 'PE_CE, P = Euler, C=Trapezni postupak'

    data_PECE_2 = np.loadtxt('DZ5/plotting/4_zad/PE_CE_2_method_plot.txt')
    x1_values_PECE_2= data_PECE_2[0, :]
    x2_values_PECE_2E = data_PECE_2[1, :]
    time_values_PECE_2= data_PECE_2[2, :]
    title_PECE_2 = 'PE_CE_2, P = Euler, C = Obrnuti Euler'

    x1_values_list = [x1_values_euler, x1_values_backeuler, x1_values_trapezoid, x1_values_runge_kutta, x1_values_PE_CE, x1_values_PECE_2]
    x2_values_list = [x2_values_euler, x2_values_backeuler, x2_values_trapezoid, x2_values_runge_kutta, x2_values_PE_CE, x2_values_PECE_2E]
    titles_list = [title_euler, title_backeuler, title_trapezoid, title_runge_kutta, title_PE_CE, title_PECE_2]

    main_title = "Solution 4th task"
    plot_comparison_plotly(time_values_euler, x1_values_list, x2_values_list,main_title, titles_list)