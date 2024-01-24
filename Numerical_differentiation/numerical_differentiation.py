import math
import sys
from typing import List
sys.path.append('')

from Solving_systems_of_linear_algebraic_equations.matrica import Matrica

def euler_method(A: Matrica,B: Matrica or None, x0: Matrica, T: float, t_max:list or float, r_t_const: bool, log:int) -> (List[Matrica], List[float], List[str]):
    '''
    Performs Euler's method for numerical integration of a system of ordinary differential equations.

    Parameters:
    - A: Matrica
        Coefficient matrix for the state variables.
    - B: Matrica or None
        Control matrix. If None, the system is considered without control.
    - x0: Matrica
        Initial state vector.
    - T: float
        Time step for numerical integration.
    - t_max: list or float
        If list, represents the time interval [0, t_max[1]] with step T. If float, represents [0, t_max] with step T=1.
    - r_t_const: bool
        If True, assumes the control input is constant over time.
    - log: int
        Log interval for printing state variables.

    Returns:
    - x: List[Matrica]
        List of state matrices at each time step.
    - time_values: List[float]
        List of time values corresponding to each state matrix.
    - log_output: List[str]
        List of log messages at each specified log interval.
    '''
    # Generate time values based on the given time interval
    if isinstance(t_max, list):
        num_steps = math.ceil(t_max[1] / T) + 1
        time_values = [round(i * T, 5) for i in range(num_steps)]
    
    #calling function from PECE or PE(CE)**2
    elif isinstance(t_max, float):
        num_steps = 2
        time_values = [None,t_max]
    
    # Initialize the state matrices
    x = [Matrica(x0.redaka, x0.stupaca) for _ in range(num_steps)]
    x[0] = x0
    
    # Initialize log output
    log_output = []
    log_output.append(f"T = {T}, diferential_interval = {t_max}")
    log_output.append(f"I{0}: starting state variable {x0.u_listu()}, t_i = {time_values[0]}")
    
    # Initialize control input matrix
    r = Matrica(x0.redaka, x0.stupaca) 

    # Perform Euler's method for numerical integration
    for i in range(1, num_steps):
        #x_k+1 = x_k + T * x_k_'
        if(isinstance(B,Matrica) == True and r_t_const==False):
            r.set_matricu([[time_values[i]], [time_values[i]]])
            x[i] = x[i - 1] + T * ((A * x[i - 1]) + (B * r))
        elif(isinstance(B,Matrica) == True and r_t_const == True):
            r.set_matricu([[1.0], [1.0]])
            x[i] = x[i - 1] + T * ((A * x[i - 1]) + (B*r))
        else:
            x[i] = x[i - 1] + T * (A * x[i - 1])
        
        if(i % log == 0):
            log_output.append(f"I{i}: state variable {x[i].u_listu()}, t_i = {time_values[i]}")

    return x,time_values,log_output
        
def backward_euler_method(A: Matrica,B: Matrica or None, x0: Matrica, T: float, t_max:list or float, r_t_const: bool, log:int, prediktor:Matrica or None) -> (List[Matrica], List[float], List[str]):
    '''
    Performs backward Euler's method for numerical integration of a system of ordinary differential equations.

    Parameters:
    - A: Matrica
        Coefficient matrix for the state variables.
    - B: Matrica or None
        Control matrix. If None, the system is considered without control.
    - x0: Matrica
        Initial state vector.
    - T: float
        Time step for numerical integration.
    - t_max: list or float
        If list, represents the time interval [0, t_max[1]] with step T. If float, represents [0, t_max] with step T=1.
    - r_t_const: bool
        If True, assumes the control input is constant over time.
    - log: int
        Log interval for printing state variables.
    - prediktor: Matrica or None
        Predictor matrix used for PECE or PE(CE)**2 method. If None, standard backward Euler is performed.

    Returns:
    - x: List[Matrica]
        List of state matrices at each time step.
    - time_values: List[float]
        List of time values corresponding to each state matrix.
    - log_output: List[str]
        List of log messages at each specified log interval.
    '''
    # Generate time values based on the given time interval
    if isinstance(t_max, list):
        num_steps = math.ceil(t_max[1] / T) + 1
        time_values = [round(i * T, 5) for i in range(num_steps)]
    
    #calling function from PECE or PE(CE)**2
    elif isinstance(t_max, float):
        num_steps = 2
        time_values = [t_max,t_max+T]

    # Initialize the state matrices
    x = [Matrica(x0.redaka, x0.stupaca) for _ in range(num_steps)]
    x[0] = x0
    
    
    log_output = []
    log_output.append(f"T = {T}, diferential_interval = {t_max}")
    log_output.append(f"I{0}: starting state variable {x0.u_listu()}, t_i = {time_values[0]}")

    # Identity matrix 
    I = Matrica(A.redaka, A.stupaca)
    I.set_matricu(elements=[[1 if i == j else 0 for j in range(A.stupaca)] for i in range(A.redaka)])
    
    # Calculate matrices P and Q for the backward Euler method
    P = Matrica()
    P =(I - (A * T)).inverz()

    Q = Matrica()
    if(isinstance(B,Matrica) == True):
        Q = P * T * B
    else:
        Q = 0

    # Initialize control input matrix
    r = Matrica(x0.redaka, x0.stupaca)

    # Perform backward Euler's method for numerical integration
    for i in range(1, num_steps):
        # x_k+1 = P * x_k + T * Q * r(t_k+1)

        # Calculate the control input matrix based on r_t_const
        if(r_t_const == True):
            r.set_matricu([[1.0], [1.0]])
            r_i = r
        else:
            r.set_matricu(elements=[[time_values[i] for _ in range(x0.stupaca)] for _ in range(x0.redaka)])
            r_i = round(r,4)

        #using explicit equation
        if(prediktor == None):
            x[i] = P * x[i - 1] + Q * r_i
        
        # Perform PECE or PE(CE)**2 method with implicit equation
        else:            
            if(B == None):
                x[i] = x[i-1] + T * (A*prediktor)
            else:
                x[i] = x[i-1] + T * (A*prediktor + B*r_i)

        if(i % log == 0):
            log_output.append(f"I{i}: state variable {x[i].u_listu()}, t_i = {time_values[i]}")

    return x, time_values, log_output

def trapezoidal_rule(A: Matrica,B: Matrica or None, x0: Matrica, T: float, t_max:list or float, r_t_const: bool, log:int ,prediktor:Matrica or None) -> (List[Matrica], List[float], List[str]):
    '''
    Performs numerical integration using the trapezoidal rule for a system of ordinary differential equations.

    Parameters:
    - A: Matrica
        Coefficient matrix for the state variables.
    - B: Matrica or None
        Control matrix. If None, the system is considered without control.
    - x0: Matrica
        Initial state vector.
    - T: float
        Time step for numerical integration.
    - t_max: list or float
        If list, represents the time interval [0, t_max[1]] with step T. If float, represents [0, t_max] with step T=1.
    - r_t_const: bool
        If True, assumes the control input is constant over time.
    - log: int
        Log interval for printing state variables.
    - prediktor: Matrica or None
        Predictor matrix used for PECE or PE(CE)**2 method. If None, standard trapezoidal rule is performed.

    Returns:
    - x: List[Matrica]
        List of state matrices at each time step.
    - time_values: List[float]
        List of time values corresponding to each state matrix.
    - log_output: List[str]
        List of log messages at each specified log interval.
    '''
    # Generate time values based on the given time interval
    if isinstance(t_max, list):
        num_steps = math.ceil(t_max[1] / T) + 1
        time_values = [round(i * T, 5) for i in range(num_steps)]

    # Handle the case where t_max is a single float
    elif isinstance(t_max, float):
        num_steps = 2
        time_values = [t_max, t_max + T]

    # Initialize the state matrices
    x = [Matrica(x0.redaka, x0.stupaca) for _ in range(num_steps)]
    x[0] = x0

    # Initialize log output
    log_output = []
    log_output.append(f"T = {T}, differential_interval = {t_max}")
    log_output.append(f"I{0}: starting state variable {x0.u_listu()}, t_i = {time_values[0]}")

    # Identity matrix
    I = Matrica(A.redaka, A.stupaca)
    I.set_matricu(elements=[[1 if i == j else 0 for j in range(A.stupaca)] for i in range(A.redaka)])

    # Calculate matrices R and S for the trapezoidal rule
    R = ((I - (A * (T / 2))).inverz()) * (I + (A * (T / 2)))

    if isinstance(B, Matrica) == True:
        S = (I - (A * (T / 2))).inverz() * (T / 2) * B
    else:
        S = 0

    # Initialize control input matrices
    r_current = Matrica(x0.redaka, x0.stupaca)
    r_next = Matrica(x0.redaka, x0.stupaca)

    # Perform trapezoidal rule for numerical integration
    for i in range(1, num_steps):
        # Calculate the control input matrix based on r_t_const
        if(r_t_const == True):
            r = Matrica(x0.redaka, x0.stupaca)
            r.set_matricu([[1.0], [1.0]])
        else:
            r_current.set_matricu(elements=[[round(time_values[i-1],3) for _ in range(x0.stupaca)] for _ in range(x0.redaka)])
            r_next.set_matricu(elements=[[round(time_values[i], 3) for _ in range(x0.stupaca)] for _ in range(x0.redaka)])
            r = round((r_current + r_next),4)
        
        # x_k+1 = R * x_k + T * Q * (r(t_k) + r(t_k+1)  
        # Perform trapezoidal rule with explicit equation    
        if(prediktor == None):
            x[i] = ( R * x[i - 1] ) + ( S * r )
        
        else:
            # Perform trapezoidal rule with implicit equation 
            if(B ==None):
                x[i] = x[i-1] + (T/2) * (A*x[i-1] + A*prediktor)
            else:
                x[i] = x[i-1] + (T/2) * (A*x[i-1] + B*r_current + A*prediktor + B*r_next)

        if(i % log == 0):
            log_output.append(f"I{i}: state variable {x[i].u_listu()}, t_i = {time_values[i]}")

    return x, time_values, log_output    

def runge_kutta_4(A: Matrica, B: Matrica or None, x0: Matrica, T: float, t_max: list, r_t_const: bool, log:int) -> (List[Matrica], List[float], List[str]):
    '''
    Performs numerical integration using the fourth-order Runge-Kutta method for a system of ordinary differential equations.

    Parameters:
    - A: Matrica
        Coefficient matrix for the state variables.
    - B: Matrica or None
        Control matrix. If None, the system is considered without control.
    - x0: Matrica
        Initial state vector.
    - T: float
        Time step for numerical integration.
    - t_max: list
        Time interval [0, t_max[1]].
    - r_t_const: bool
        If True, assumes the control input is constant over time.
    - log: int
        Log interval for printing state variables.

    Returns:
    - x: List[Matrica]
        List of state matrices at each time step.
    - time_values: List[float]
        List of time values corresponding to each state matrix.
    - log_output: List[str]
        List of log messages at each specified log interval.
    '''

    num_steps = math.ceil(t_max[1] / T) + 1
    time_values = [round(i * T, 5) for i in range(num_steps)]

    # Initialize the state matrices
    x = [Matrica(x0.redaka, x0.stupaca) for _ in range(num_steps)]
    x[0] = x0
    
    log_output = []
    log_output.append(f"T = {T}, diferential_interval = {t_max}")
    log_output.append(f"I{0}: starting state variable {x0.u_listu()}, t_i = {time_values[0]}")

    # Initialize control input matrices
    r = Matrica(x0.redaka, x0.stupaca)
    T_pol, T_full = Matrica(x0.redaka, x0.stupaca),Matrica(x0.redaka, x0.stupaca)
    T_pol.set_matricu([[T/2], [T/2]])
    T_full.set_matricu([[T], [T]])

    for i in range(1, num_steps):
        #x = Ax + B * r(t).
        if(isinstance(B,Matrica) == True):
            if(r_t_const == True):
                r.set_matricu([[1], [1]])
            else:
                r.set_matricu([[time_values[i-1]], [time_values[i-1]]])
            
            m1 =  (A * x[i - 1]) + (B * r)
            m2 = (A * (x[i - 1] + (T/2) * m1)) + (B * (r + T_pol))   
            m3 = (A * (x[i - 1] + (T/2) * m2)) + (B * (r + T_pol))   
            m4 = (A * (x[i - 1] + T * m3)) + (B * (r + T_full))   
        else:
            m1 =  (A * x[i - 1])
            m2 = A * (x[i - 1] + (T/2) * m1)   
            m3 = A * (x[i - 1] + (T/2) * m2)
            m4 = A * (x[i - 1] + T * m3)


        x[i] = x[i - 1] + (T/6) * (m1 + 2*m2 + 2*m3 + m4) 
        
        if(i % log == 0):
            log_output.append(f"I{i}: state variable {x[i].u_listu()}, t_i = {time_values[i]}")

    return x, time_values, log_output

def linear_multistep_PE_CE(A: Matrica, B: Matrica or None, x0: Matrica, T: float, t_max: list, const_t: bool,log: int, prediktor: callable, korektor: callable) -> (List[Matrica], List[float], List[str]):
    '''
    Performs linear multistep integration using predictor-corrector method for a system of ordinary differential equations.

    Parameters:
    - A: Matrica
        Coefficient matrix for the state variables.
    - B: Matrica or None
        Control matrix. If None, the system is considered without control.
    - x0: Matrica
        Initial state vector.
    - T: float
        Time step for numerical integration.
    - t_max: list
        Time interval [0, t_max[1]].
    - const_t: bool
        If True, assumes the control input is constant over time.
    - log: int
        Log interval for printing state variables.
    - prediktor: callable
        Predictor function used for the integration method.
    - korektor: callable
        Corrector function used for the integration method.

    Returns:
    - x: List[Matrica]
        List of state matrices at each time step.
    - time_values: List[float]
        List of time values corresponding to each state matrix.
    - log_output: List[str]
        List of log messages at each specified log interval.
    '''
    # Calculate the number of steps based on the given time interval
    num_steps = math.ceil(t_max[1] / T) + 1
    time_values = [round(i * T, 5) for i in range(num_steps)]

    # Initialize the list of state matrices
    x = [Matrica(x0.redaka, x0.stupaca) for _ in range(num_steps)]
    x[0] = x0

    log_output = []
    log_output.append(f"T = {T}, diferential_interval = {t_max}")
    log_output.append(f"I{0}: starting state variable {x0.u_listu()}, t_i = {time_values[0]}")
    
    # Perform the integration using predictor-corrector method
    for i in range(1,num_steps):
        # x_{k+1} = EXPLICIT(x_k, t_k)
        
        x_prediktor, _, _ = prediktor(A, B, x[i-1], T, time_values[i-1], const_t, log)

        # x_{k+1}=IMPLICIT(x_k, t_k, EXPLICITNI(x_k, t_k))
        x_korektor, _, _ = korektor(A, B, x[i-1], T, time_values[i-1], const_t, log, prediktor = x_prediktor[1])            
        
        x[i] = x_korektor[1]
        if(i % log == 0):
            log_output.append(f"I{i}: state variable {x[i].u_listu()}, t_i = {time_values[i]}")

    return x, time_values, log_output

def linear_multistep_PE_CE_2(A: Matrica, B: Matrica or None, x0: Matrica, T: float, t_max: list, const_t: bool, log: int, prediktor: callable, korektor: callable) -> (List[Matrica], List[float], List[str]):
    '''
    Performs a linear multistep integration using a predictor-corrector method for a system of ordinary differential equations.

    Parameters:
    - A: Matrica
        Coefficient matrix for the state variables.
    - B: Matrica or None
        Control matrix. If None, the system is considered without control.
    - x0: Matrica
        Initial state vector.
    - T: float
        Time step for numerical integration.
    - t_max: list
        Time interval [0, t_max[1]].
    - const_t: bool
        If True, assumes the control input is constant over time.
    - log: int
        Log interval for printing state variables.
    - prediktor: callable
        Predictor function used for the integration method.
    - korektor: callable
        Corrector function used for the integration method.

    Returns:
    - x: List[Matrica]
        List of state matrices at each time step.
    - time_values: List[float]
        List of time values corresponding to each state matrix.
    - log_output: List[str]
        List of log messages at each specified log interval.
    '''
    # Calculate the number of steps based on the given time interval
    num_steps = math.ceil(t_max[1] / T) + 1
    time_values = [round(i * T, 5) for i in range(num_steps)]

    # Initialize the list of state matrices
    x = [Matrica(x0.redaka, x0.stupaca) for _ in range(num_steps)]
    x[0] = x0

    log_output = []
    log_output.append(f"T = {T}, diferential_interval = {t_max}")
    log_output.append(f"I{0}: starting state variable {x0.u_listu()}, t_i = {time_values[0]}")
    
    # Perform the integration using predictor-corrector method
    for i in range(1,num_steps):
        #x_{k+1}=IMPLICIT(x_k, t_k, IMPLICIT(x_k, t_k, EXPLICIT(x_k, t_k)))    
        x_prediktor, _, _ = prediktor(A, B, x[i-1], T, time_values[i-1], const_t, log)

        x_korektor_1, _, _ = korektor(A, B, x[i-1], T, time_values[i-1], const_t, log, prediktor = x_prediktor[1]) 
        
        x_korektor_2, _, _ = korektor(A, B, x[i-1], T, time_values[i-1], const_t, log, prediktor = x_korektor_1[1])           
        
        x[i] = x_korektor_2[1]
        if(i % log == 0):
            log_output.append(f"I{i}: state variable {x[i].u_listu()}, t_i = {time_values[i]}")

    return x, time_values, log_output

