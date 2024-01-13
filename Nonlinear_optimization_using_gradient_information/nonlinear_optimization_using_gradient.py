import math
import sys
sys.path.append('')

from Solving_systems_of_linear_algebraic_equations.matrica import Matrica



def zapis_gradijent_u_datoteku(prvi: int, drugi: int, ime_datoteke: str) -> None:
    '''
    Helper function to write the first and third parameters to a file.
    This is intended for storing gradient information in a file for later use in creating an object of the 'Matrix' class.

    Parameters:
    - first (int): The first parameter to be written.
    - third (int): The third parameter to be written.
    - file_name (str): The name of the file to which the parameters will be written.

    Returns:
    - None
    '''
    with open(ime_datoteke, 'w') as datoteka:
        datoteka.write(f"{prvi} \n{drugi} ")


def poboljsanje(lst: list)-> bool:
    '''
    Checks for improvement based on the last 10 elements of the given list.

    Parameters:
    - lst (list): The list to check for improvement.

    Returns:
    - bool: True if the last 10 elements are the same and occur at least 5 times, otherwise False.
    '''

    if len(lst) >= 10:
        # Extract the last 10 elements
        last_10 = lst[-10:]
        # Check if at least 5 of the last 10 elements are the same
        if last_10.count(last_10[0]) >= 5:
            return True
    return False

def norma(vektor: list) -> float:
    '''
    Calculates the Euclidean norm (L2 norm) of a vector.

    Parameters:
    - vector (list or numeric): Input vector or a single numeric value.

    Returns:
    - float: Euclidean norm of the input vector.
    '''

    norm = 0
    for i in vektor:
        norm += i**2

    return math.sqrt(norm)

def unimodalni_interval(start_point: int, step_size: float, target_function: 'function') -> (float, float):
    '''
    Find a unimodal interval around the given start_point for the target function.

    Args:
    - start_point (int): The initial point for the search.
    - step_size (float): The step size used in the search.
    - target_function (function): The target function to find the unimodal interval for.

    Returns:
    - Tuple(float, float): A tuple containing the left and right boundaries of the unimodal interval.
    '''
    l = start_point - step_size
    r= start_point + step_size
    m = start_point
    fl, fm, fr = target_function(l), target_function(m), target_function(r)
    step = 1

    if fm < fr and fm < fl:
        return l, r

    elif fm > fr:
        while fm > fr:
            l = m
            m = r
            fm = fr
            r = start_point + step_size * (2 ** step)
            fr = target_function(r)
            step += 1
    else:
        while fm > fl:
            r = m
            m = l
            fm = fl
            l = start_point - step_size * (2 ** step)
            fl = target_function(l)
            step += 1

    return l, r

def zlatni_rez(a: float, b: float, precision: float, target_function: 'function',start_point = None) -> (float, float, float):
    '''
    Find the minimum of a unimodal function using the golden section search algorithm.

    Args:
    - a: The left boundary of the initial interval.
    - b: The right boundary of the initial interval.
    - precision: The desired precision of the minimum.
    - target_function: The target function to minimize.
    - start_point: If provided, the function will use this point to determine the initial interval.

    Returns:
    - Tuple(float, float, float): A tuple containing the left boundary, right boundary, and the minimum point.
    '''

    k = 0.5 * (math.sqrt(5) - 1)

    if  start_point != None:
        a, b = unimodalni_interval(start_point, 1.0, target_function)
        c = b - k * (b - a)
        d = a + k * (b - a)
    else:
        c = b - k * (b - a)
        d = a + k * (b - a)

    fc = target_function(c)
    fd = target_function(d)

    while abs(b - a) > precision:
        if fc < fd:
            b = d
            d = c
            c = b - k * (b - a)
            fd = fc
            fc = target_function(c)
        else:
            a = c
            c = d
            d = a + k * (b - a)
            fc = fd
            fd = target_function(d)

    return a, b , (a + b) / 2

def gradijetni_spust(f: 'function', gradient_f: 'function', x0: list, e: float, use_golden_ratio:bool) -> (list,float):
    '''
    Gradient descent optimization algorithm.

    Args:
    - f (function): Objective function to minimize.
    - gradient_f (function): Gradient of the objective function.
    - x0 (list): Initial guess for the minimum point.
    - e (float): Tolerance for stopping criteria.
    - use_golden_ratio (bool): Flag to determine whether to use golden ratio for step size optimization.

    Returns:
    - Tuple(list, float, str): A tuple containing the optimal point, the function value at the optimal point, 
      and a warning message if the algorithm stagnates.
    '''

    x = [i for i in x0]
    pob = []  # List to store function values for checking improvement
    brojac = 0  # Counter for iterations

    while True :
        gradijent = gradient_f(x) #determining the direction
        
        if use_golden_ratio:
            norma_gr = norma(gradijent)
            v = [-x/norma_gr for x in gradijent] # Normalized gradient direction
        else:
            v=[-x for x in gradijent] # Non-normalized gradient direction

        for i,x_i in enumerate(x):
            if use_golden_ratio:
                 # Using golden ratio to find the optimal step size
                a, b = unimodalni_interval(0, 1.0, lambda delta: f([x[i] + delta * v[i] for i in range(len(x0))]))
                
                def modified_function(lambda_val):
                    return f([x[j] + lambda_val * v[j] for j in range(len(x))])      # Using golden ratio to find the optimal step size
                _, _, hp = zlatni_rez(a, b, e, modified_function,x[i])
           
            else:
                # Full step size
                hp = 1
            
            x[i] = x[i] + hp * v[i]
        
        # Check for convergence based on the gradient
        if(norma(gradijent) < e):
            break
    
        ## Check for improvement in the objective function
        pob.append(f(x))
        if(poboljsanje(pob) or brojac == 1000000):
            return x,pob[-1],"Upozorenje: Algoritam stagnira!"
        brojac += 1

    return x,pob[-1]

def newton_raphson(f: 'function', gradient_f: 'function', hess_f: 'function', x0: list, e: float ,ime_datoteke: str,use_golden_ratio: bool) -> (list,float,str):
    '''
    Newton-Raphson optimization algorithm for unconstrained optimization.

    Parameters:
    - f (function): Objective function to minimize.
    - gradient_f (function): Gradient (first derivative) of the objective function.
    - hess_f (function): Hessian matrix (second derivative) of the objective function.
    - x0 (list): Initial guess for the minimum point.
    - e (float): Tolerance for stopping criteria based on the gradient norm.
    - file_name (str): Name of the file to store the gradient and Hessian matrix.
    - use_golden_ratio (bool): If True, use the golden ratio search for line search; otherwise, use a fixed step size.

    Returns:
    - Tuple[list, float, None/str]: A tuple containing the optimal point, the final value of the objective function,
      and a warning message if the algorithm stagnates, or None if no warning.
    '''

    
    # Initialize the current point
    x = [i for i in x0]
    # List to store function values during optimization
    pob = []

    while True :
        # Compute the gradient for determining the direction
        gradijent = gradient_f(x) 

        zapis_gradijent_u_datoteku(gradijent[0],gradijent[1],'DZ3/gradijent/gradijent_ulaz.txt')
        F = Matrica()
        F.citaj_iz_datoteke('DZ3/gradijent/gradijent_ulaz.txt')
        
        # Compute the Hessian matrix and read it from the file
        hess_f(x,ime_datoteke)
        H = Matrica()
        H.citaj_iz_datoteke(ime_datoteke)
        # Compute the inverse of the Hessian matrix
        H_inv = -1 * H.inverz()

        # Compute the change in position using the Newton-Raphson formula       
        delta_x = H_inv * F
        delta_x = delta_x.u_listu()

        # Update the position using a line search
        for i,x_i in enumerate(x):
            if use_golden_ratio:
                # Use the golden ratio search to find the optimal step size
                a, b = unimodalni_interval(0, 1.0, lambda delta: f([x[i] + delta * delta_x[i] for i in range(len(x0))]))
                
                def modified_function(lambda_val):
                    return f([x[j] + lambda_val * delta_x[j] for j in range(len(x))])     #pretrazujem u smjeru vektora v
                _, _, hp = zlatni_rez(a, b, e, modified_function,x[i])
            else:
                # Use a fixed step size
                hp = 1
            
            # Update the position
            x[i] = x[i] + hp * delta_x[i]

        # Check for convergence based on the gradient
        if(norma(gradijent) < e):
            break
    
        # Check for lack of improvement in the objective function
        pob.append(f(x))
        if(poboljsanje(pob)):
            print("Upozorenje: Algoritam stagnira!")
            return x,pob[-1],"Upozorenje: Algoritam stagnira!"
    
    return x,pob[-1],None

def gauss_newton(f: 'function', G_input: 'function', Jacobian: 'function', x0: list, e: float, use_golden_ratio: bool) -> (list, list, str):
    '''
    Gauss-Newton optimization algorithm for nonlinear least squares problems.

    Parameters:
    - f (function): Objective function for computing the gradient.
    - G_input (function): Function representing the target values of the nonlinear model.
    - Jacobian (function): Jacobian matrix of the model.
    - x0 (list): Initial guess for the minimum point.
    - e (float): Tolerance for stopping criteria based on the norm of the residual.
    - use_golden_ratio (bool): If True, use the golden ratio search for line search; otherwise, use a fixed step size.

    Returns:
    - Tuple[list, list, None/str]: A tuple containing the optimal point, the final values of the nonlinear model,
      and a warning message if the algorithm stagnates, or None if no warning.
    '''

    # Initialize the current point
    x = [i for i in x0]
    # List to store function values during optimization
    pob = []
    # Counter for the number of iterations
    brojac = 0

    while True:
        # Compute the Jacobian matrix
        J = Jacobian(x)
        # Compute the target values of the model
        G = G_input(x)
        # Transpose of the Jacobian matrix
        J_t = J.T()

        # Gauss-Newton update formula
        A = J_t * J
        g_pom = J_t * G
        g = -1 * g_pom

         # Solve the linear system using LUP decomposition
        U2_lup, L2_lup, P2_lup = A.LUP()

        y = L2_lup.supstitucija_unaprijed(P2_lup*g)
        delta_x_matrix = U2_lup.supstitucija_unatrag(y)

        delta_x = delta_x_matrix.u_listu()
        
        # Update the position using a line search  
        for i,x_i in enumerate(x):
            if use_golden_ratio:
                # Use the golden ratio search to find the optimal step size
                a, b = unimodalni_interval(0, 1.0, lambda delta: f([x[i] + delta * delta_x[i] for i in range(len(x))]))
                
                def modified_function(lambda_val):
                    return f([x[j] + lambda_val * delta_x[j] for j in range(len(x))])     #pretrazujem u smjeru vektora v
                _, _, hp = zlatni_rez(a, b, e, modified_function,x[i])
            else:
                # Use a fixed step size
                hp = 1

            # Update the position
            x[i] = x[i] + hp * delta_x[i]

        # Check for convergence based on the residual
        hp_list = [hp] * len(delta_x)
        result = [hp_i * delta_x_i for hp_i, delta_x_i in zip(hp_list, delta_x)]
        
        if (norma(result) < e):
            break
        
        # Check for lack of improvement in the objective function
        pob.append(f(x))
        if(poboljsanje(pob) or brojac == 100000):
            print("Upozorenje: Algoritam stagnira!")
            return x,pob[-1],"Upozorenje: Algoritam stagnira!"
        brojac+=1    


    return x,G_input(x).u_listu(),"Ne stagnira"

