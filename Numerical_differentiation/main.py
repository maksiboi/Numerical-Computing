import numerical_differentiation
from Solving_systems_of_linear_algebraic_equations.matrica import Matrica
import math
import plot
from typing import List

def write_plotting_info(matrice: List[Matrica], y_time: List[float], filename: str) -> None:
    '''
    Write the plotting information to a file.

    Parameters:
    - matrice: List[Matrica]
        List of matrices to be written to the file.
    - y_time: List[float]
        List of time values to be written to the file.
    - ime_datoteke: str
        File name for writing the plotting information.
    '''
    # Otvori datoteku za pisanje
    with open(filename, 'w') as file:
        # Transpose matrices to write each matrix in its own column
        T_matrice = list(map(list, zip(*matrice)))

        # Write each transposed matrix to the file
        for matrica in T_matrice:
            # Format each element of the matrix as a string and write them to the file
            file.write(" ".join([f"{str(element[0])}" for element in matrica]) + "\n")
        
        # Write the time values to the file
        file.write(" ".join(map(str, y_time)))

def write_logs_to_txt(log_output: List[float], file_path: str) -> None:
    with open(file_path, 'w') as file:
        for row in log_output:
            file.write(row + '\n')

def prvi():
    A = Matrica()
    B = None
    x0 = Matrica()
    T = 0.01
    t_max = [0,10]
    const_t = False # t is not constant
    logging = 100
    A.citaj_iz_datoteke("DZ5/matrice/1_zad/A.txt")
    x0.citaj_iz_datoteke("DZ5/matrice/1_zad/x0.txt")

    #Euler method
    solution_euler,time_values_euler, log_output_euler = numerical_differentiation.euler_method(A, B, x0, T, t_max, const_t, logging)

    x1_real = []
    x2_real = []
    for t in time_values_euler:
        x1_real.append(x0[0][0]*math.cos(t) + x0[1][0]*math.sin(t))
        x2_real.append(x0[1][0]*math.cos(t) - x0[0][0]*math.sin(t))

    approximation_error_x1, approximation_error_x2 = 0,0
    for i,x_approximated in enumerate(solution_euler):
        approximation_error_x1 += abs((x_approximated[0][0] - x1_real[i]))
        approximation_error_x2 += abs((x_approximated[1][0] - x2_real[i]))
    
    log_output_euler.append(f"Approximation_error for Euler method: {[approximation_error_x1, approximation_error_x2]}")
    
    write_logs_to_txt(log_output_euler,'DZ5/solution/1_zad/euler_method_logs.txt')
    write_plotting_info(solution_euler,time_values_euler, 'DZ5/plotting/1_zad/euler_method_plot.txt')
    #############################################################################
    #Backward Euler method
    solution_backeuler,time_values_backeuler, log_output_backeuler = numerical_differentiation.backward_euler_method(A, B, x0, T, t_max, const_t, logging, prediktor=None)

    approximation_error_x1, approximation_error_x2 = 0,0
    for i,x_approximated in enumerate(solution_backeuler):
        approximation_error_x1 += abs((x_approximated[0][0] - x1_real[i]))
        approximation_error_x2 += abs((x_approximated[1][0] - x2_real[i]))
    
    log_output_backeuler.append(f"Approximation_error for Backward-Euler method: {[approximation_error_x1, approximation_error_x2]}")
    
    write_logs_to_txt(log_output_backeuler,'DZ5/solution/1_zad/backward_euler_method_logs.txt')
    write_plotting_info(solution_backeuler,time_values_backeuler, 'DZ5/plotting/1_zad/backward_euler_method_plot.txt')
    #############################################################################################
    #Trapezoidal rule method
    solution_trapezoidal,time_values_trapezoidal, log_output_trapezoidal = numerical_differentiation.trapezoidal_rule(A, B, x0, T, t_max, const_t, logging, prediktor=None)

    approximation_error_x1, approximation_error_x2 = 0,0

    for i,x_approximated in enumerate(solution_trapezoidal):
        approximation_error_x1 += abs((x_approximated[0][0] - x1_real[i]))
        approximation_error_x2 += abs((x_approximated[1][0] - x2_real[i]))
    
    log_output_trapezoidal.append(f"Approximation_error for Trapezoidal rule method: {[approximation_error_x1, approximation_error_x2]}")
    
    write_logs_to_txt(log_output_trapezoidal,'DZ5/solution/1_zad/trapezoidal_method_logs.txt')
    write_plotting_info(solution_trapezoidal,time_values_trapezoidal, 'DZ5/plotting/1_zad/trapezoidal_method_plot.txt')
    ##########################################################################################
    #Runge-Kutta 4 method
    solution_runge_kutta,time_values_runge_kutta, log_output_runge_kutta = numerical_differentiation.runge_kutta_4(A, B, x0, T, t_max, const_t, logging)

    approximation_error_x1, approximation_error_x2 = 0,0

    for i,x_approximated in enumerate(solution_runge_kutta):
        approximation_error_x1 += abs((x_approximated[0][0] - x1_real[i]))
        approximation_error_x2 += abs((x_approximated[1][0] - x2_real[i]))
    
    log_output_runge_kutta.append(f"Approximation_error for Runge-Kutta method: {[approximation_error_x1, approximation_error_x2]}")
    
    write_logs_to_txt(log_output_runge_kutta,'DZ5/solution/1_zad/runge_kutta_method_logs.txt')
    write_plotting_info(solution_runge_kutta,time_values_runge_kutta, 'DZ5/plotting/1_zad/runge_kutta_method_plot.txt')
    #########################################################################################
    #PECE where P is Euler method and C is Trapezoidal rule 
    solution_PE_CE,time_values_PE_CE, log_outputPE_CE = numerical_differentiation.linear_multistep_PE_CE(A, B, x0, T, t_max, const_t, logging, numerical_differentiation.euler_method, numerical_differentiation.trapezoidal_rule)
    
    approximation_error_x1, approximation_error_x2 = 0,0

    for i,x_approximated in enumerate(solution_PE_CE):
        approximation_error_x1 += abs((x_approximated[0][0] - x1_real[i]))
        approximation_error_x2 += abs((x_approximated[1][0] - x2_real[i]))
    
    log_outputPE_CE.append(f"Approximation_error for PE_CE method: {[approximation_error_x1, approximation_error_x2]}")
    
    write_logs_to_txt(log_outputPE_CE,'DZ5/solution/1_zad/PE_CE_method_logs.txt')
    
    write_plotting_info(solution_PE_CE,time_values_PE_CE, 'DZ5/plotting/1_zad/PE_CE_method_plot.txt')
    #########################################################################################
    #PE(CE)**2 where P is Euler method and C is Backward Euler method
    solution_PE_CE_2,time_values_PE_CE_2, log_output_PE_CE_2 = numerical_differentiation.linear_multistep_PE_CE_2(A, B, x0, T, t_max, const_t, logging, numerical_differentiation.euler_method, numerical_differentiation.backward_euler_method)
    
    approximation_error_x1, approximation_error_x2 = 0,0

    for i,x_approximated in enumerate(solution_PE_CE_2):
        approximation_error_x1 += abs((x_approximated[0][0] - x1_real[i]))
        approximation_error_x2 += abs((x_approximated[1][0] - x2_real[i]))
    
    log_output_PE_CE_2.append(f"Approximation_error for PE_CE_2 method: {[approximation_error_x1, approximation_error_x2]}")
    
    write_logs_to_txt(log_output_PE_CE_2,'DZ5/solution/1_zad/PE_CE_2_method_logs.txt')
    
    write_plotting_info(solution_PE_CE_2,time_values_PE_CE_2, 'DZ5/plotting/1_zad/PE_CE_2_method_plot.txt')

    #plot result
    plot.plot_1_zad()

def drugi():
    A = Matrica()
    B = None
    x0 = Matrica()
    T = 0.01
    t_max = [0,1]
    const_t = False # t is not constant
    A.citaj_iz_datoteke("DZ5/matrice/2_zad/A.txt")
    x0.citaj_iz_datoteke("DZ5/matrice/2_zad/x0.txt")
    logging = 100

    #Euler method
    solution_euler,time_values_euler, log_output_euler = numerical_differentiation.euler_method(A, B, x0, T, t_max, const_t, logging)
    
    log_output_euler.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_euler[-1].u_listu()}")
    
    write_logs_to_txt(log_output_euler,'DZ5/solution/2_zad/euler_method_logs.txt')
    write_plotting_info(solution_euler,time_values_euler, 'DZ5/plotting/2_zad/euler_method_plot.txt')
    #############################################################################
    #Backward Euler method
    solution_backeuler,time_values_backeuler, log_output_backeuler = numerical_differentiation.backward_euler_method(A, B, x0, T, t_max, const_t, logging, prediktor=None)
    
    log_output_backeuler.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_backeuler[-1].u_listu()}")
   
    
    write_logs_to_txt(log_output_backeuler,'DZ5/solution/2_zad/backward_euler_method_logs.txt')
    write_plotting_info(solution_backeuler,time_values_backeuler, 'DZ5/plotting/2_zad/backward_euler_method_plot.txt')
    #############################################################################################
    #Trapezoidal rule method
    solution_trapezoidal,time_values_trapezoidal, log_output_trapezoidal = numerical_differentiation.trapezoidal_rule(A, B, x0, T, t_max, const_t, logging, prediktor=None)

    log_output_trapezoidal.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_trapezoidal[-1].u_listu()}")
    
    write_logs_to_txt(log_output_trapezoidal,'DZ5/solution/2_zad/trapezoidal_method_logs.txt')
    write_plotting_info(solution_trapezoidal,time_values_trapezoidal, 'DZ5/plotting/2_zad/trapezoidal_method_plot.txt')
    ##########################################################################################
    #Runge-Kutta 4 method
    solution_runge_kutta,time_values_runge_kutta, log_output_runge_kutta = numerical_differentiation.runge_kutta_4(A, B, x0, T, t_max, const_t, logging)

    log_output_runge_kutta.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_runge_kutta[-1].u_listu()}")
    
    write_logs_to_txt(log_output_runge_kutta,'DZ5/solution/2_zad/runge_kutta_method_logs.txt')
    write_plotting_info(solution_runge_kutta,time_values_runge_kutta, 'DZ5/plotting/2_zad/runge_kutta_method_plot.txt')
    #########################################################################################
    #PECE where P is Euler method and C is Trapezoidal rule 

    solution_PE_CE,time_values_PE_CE, log_outputPE_CE = numerical_differentiation.linear_multistep_PE_CE(A, B, x0, T, t_max, const_t, logging, numerical_differentiation.euler_method, numerical_differentiation.trapezoidal_rule)
    log_outputPE_CE.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_PE_CE[-1].u_listu()}")
    
    write_logs_to_txt(log_outputPE_CE,'DZ5/solution/2_zad/PE_CE_method_logs.txt')
    
    write_plotting_info(solution_PE_CE,time_values_PE_CE, 'DZ5/plotting/2_zad/PE_CE_method_plot.txt')
    #########################################################################################
    #PE(CE)**2 where P is Euler method and C is Backward Euler method   
    solution_PE_CE_2,time_values_PE_CE_2, log_output_PE_CE_2 = numerical_differentiation.linear_multistep_PE_CE_2(A, B, x0, T, t_max, const_t, logging, numerical_differentiation.euler_method, numerical_differentiation.backward_euler_method)
    
    log_output_PE_CE_2.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_backeuler[-1].u_listu()}")
    write_logs_to_txt(log_output_PE_CE_2,'DZ5/solution/2_zad/PE_CE_2_method_logs.txt')
    
    write_plotting_info(solution_PE_CE_2,time_values_PE_CE_2, 'DZ5/plotting/2_zad/PE_CE_2_method_plot.txt')

    #plot result
    plot.plot_2_zad()

def treci():
    A = Matrica()
    B = Matrica()
    x0 = Matrica()
    T = 0.01
    t_max = [0,10]
    logging = 100
    r_t_const = True # t is not constant
    A.citaj_iz_datoteke("DZ5/matrice/3_zad/A.txt")
    B.citaj_iz_datoteke("DZ5/matrice/3_zad/B.txt")
    x0.citaj_iz_datoteke("DZ5/matrice/3_zad/x0.txt")
    
    #Euler method
    solution_euler,time_values_euler, log_output_euler = numerical_differentiation.euler_method(A, B, x0, T, t_max, r_t_const, logging)

    log_output_euler.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_euler[-1].u_listu()}")
    
    write_logs_to_txt(log_output_euler,'DZ5/solution/3_zad/euler_method_logs.txt')
    write_plotting_info(solution_euler,time_values_euler, 'DZ5/plotting/3_zad/euler_method_plot.txt')
    #############################################################################
    #Backward Euler method
    solution_backeuler,time_values_backeuler, log_output_backeuler = numerical_differentiation.backward_euler_method(A, B, x0, T, t_max, r_t_const, logging, prediktor=None)

    log_output_backeuler.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_backeuler[-1].u_listu()}")
    
    write_logs_to_txt(log_output_backeuler,'DZ5/solution/3_zad/backward_euler_method_logs.txt')
    write_plotting_info(solution_backeuler,time_values_backeuler, 'DZ5/plotting/3_zad/backward_euler_method_plot.txt')
    #############################################################################################
    #Trapezoidal rule method
    solution_trapezoidal,time_values_trapezoidal, log_output_trapezoidal = numerical_differentiation.trapezoidal_rule(A, B, x0, T, t_max, r_t_const, logging, prediktor = None)

    log_output_trapezoidal.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_trapezoidal[-1].u_listu()}")
    
    write_logs_to_txt(log_output_trapezoidal,'DZ5/solution/3_zad/trapezoidal_method_logs.txt')
    write_plotting_info(solution_trapezoidal,time_values_trapezoidal, 'DZ5/plotting/3_zad/trapezoidal_method_plot.txt')
    ##########################################################################################
    #Runge-Kutta 4 method
    solution_runge_kutta,time_values_runge_kutta, log_output_runge_kutta = numerical_differentiation.runge_kutta_4(A, B, x0, T, t_max, r_t_const, logging)

    log_output_runge_kutta.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_runge_kutta[-1].u_listu()}")
    
    write_logs_to_txt(log_output_runge_kutta,'DZ5/solution/3_zad/runge_kutta_method_logs.txt')
    write_plotting_info(solution_runge_kutta,time_values_runge_kutta, 'DZ5/plotting/3_zad/runge_kutta_method_plot.txt')
    #########################################################################################
    solution_PE_CE,time_values_PE_CE, log_outputPE_CE = numerical_differentiation.linear_multistep_PE_CE(A, B, x0, T, t_max, r_t_const, logging, numerical_differentiation.euler_method, numerical_differentiation.trapezoidal_rule)
    
    log_outputPE_CE.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_PE_CE[-1].u_listu()}")
    
    write_logs_to_txt(log_outputPE_CE,'DZ5/solution/3_zad/PE_CE_method_logs.txt')
    
    write_plotting_info(solution_PE_CE,time_values_PE_CE, 'DZ5/plotting/3_zad/PE_CE_method_plot.txt')
    #########################################################################################
    #PE(CE)**2 where P is Euler method and C is Backward Euler method   
    solution_PE_CE_2,time_values_PE_CE_2, log_output_PE_CE_2 = numerical_differentiation.linear_multistep_PE_CE_2(A, B, x0, T, t_max, r_t_const, logging, numerical_differentiation.euler_method, numerical_differentiation.backward_euler_method)
    log_output_PE_CE_2.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_backeuler[-1].u_listu()}") 
    
    write_logs_to_txt(log_output_PE_CE_2,'DZ5/solution/3_zad/PE_CE_2_method_logs.txt')
    
    write_plotting_info(solution_PE_CE_2,time_values_PE_CE_2, 'DZ5/plotting/3_zad/PE_CE_2_method_plot.txt')
    
    #plot result
    plot.plot_3_zad()

def cetvrti():
    A = Matrica()
    B = Matrica()
    x0 = Matrica()
    T = 0.01
    t_max = [0,1]
    logging = 10
    r_t_const = False # t is not constant
    A.citaj_iz_datoteke("DZ5/matrice/4_zad/A.txt")
    B.citaj_iz_datoteke("DZ5/matrice/4_zad/B.txt")
    x0.citaj_iz_datoteke("DZ5/matrice/4_zad/x0.txt")

    #Euler method
    solution_euler,time_values_euler, log_output_euler = numerical_differentiation.euler_method(A, B, x0, T, t_max, r_t_const, logging)

    log_output_euler.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_euler[-1].u_listu()}")
    
    write_logs_to_txt(log_output_euler,'DZ5/solution/4_zad/euler_method_logs.txt')
    write_plotting_info(solution_euler,time_values_euler, 'DZ5/plotting/4_zad/euler_method_plot.txt')
    #############################################################################
    #Backward Euler method
    solution_backeuler,time_values_backeuler, log_output_backeuler = numerical_differentiation.backward_euler_method(A, B, x0, T, t_max, r_t_const, logging, prediktor=None)

    log_output_backeuler.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_backeuler[-1].u_listu()}")
    
    write_logs_to_txt(log_output_backeuler,'DZ5/solution/4_zad/backward_euler_method_logs.txt')
    write_plotting_info(solution_backeuler,time_values_backeuler, 'DZ5/plotting/4_zad/backward_euler_method_plot.txt')
    #############################################################################################
    #Trapezoidal rule method
    solution_trapezoidal,time_values_trapezoidal, log_output_trapezoidal = numerical_differentiation.trapezoidal_rule(A, B, x0, T, t_max, r_t_const, logging, prediktor=None)

    log_output_trapezoidal.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_trapezoidal[-1].u_listu()}")
    
    write_logs_to_txt(log_output_trapezoidal,'DZ5/solution/4_zad/trapezoidal_method_logs.txt')
    write_plotting_info(solution_trapezoidal,time_values_trapezoidal, 'DZ5/plotting/4_zad/trapezoidal_method_plot.txt')
    ##########################################################################################
    #Runge-Kutta 4 method
    solution_runge_kutta,time_values_runge_kutta, log_output_runge_kutta = numerical_differentiation.runge_kutta_4(A, B, x0, T, t_max, r_t_const, logging)

    log_output_runge_kutta.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_runge_kutta[-1].u_listu()}")
    
    write_logs_to_txt(log_output_runge_kutta,'DZ5/solution/4_zad/runge_kutta_method_logs.txt')
    write_plotting_info(solution_runge_kutta,time_values_runge_kutta, 'DZ5/plotting/4_zad/runge_kutta_method_plot.txt')
    #########################################################################################
    
    solution_PE_CE,time_values_PE_CE, log_outputPE_CE = numerical_differentiation.linear_multistep_PE_CE(A, B, x0, T, t_max, r_t_const, logging, numerical_differentiation.euler_method, numerical_differentiation.trapezoidal_rule)
    
    log_outputPE_CE.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_PE_CE[-1].u_listu()}")
    
    write_logs_to_txt(log_outputPE_CE,'DZ5/solution/4_zad/PE_CE_method_logs.txt')
    
    write_plotting_info(solution_PE_CE,time_values_PE_CE, 'DZ5/plotting/4_zad/PE_CE_method_plot.txt')
    #########################################################################################
    #PE(CE)**2 where P is Euler method and C is Backward Euler method   
    solution_PE_CE_2,time_values_PE_CE_2, log_output_PE_CE_2 = numerical_differentiation.linear_multistep_PE_CE_2(A, B, x0, T, t_max, r_t_const, logging, numerical_differentiation.euler_method, numerical_differentiation.backward_euler_method)
    
    log_output_PE_CE_2.append(f"Method stopped in point: x(t={t_max[1]}) = {solution_backeuler[-1].u_listu()}")
    write_logs_to_txt(log_output_PE_CE_2,'DZ5/solution/4_zad/PE_CE_2_method_logs.txt')
    
    write_plotting_info(solution_PE_CE_2,time_values_PE_CE_2, 'DZ5/plotting/4_zad/PE_CE_2_method_plot.txt')
    
    #plot result
    plot.plot_4_zad()

def main():
    prvi()
    drugi()
    treci()
    cetvrti()

if __name__ == "__main__":
    main()