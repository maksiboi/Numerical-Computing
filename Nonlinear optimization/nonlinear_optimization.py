import math
import copy
import random

def generator_x0() -> list:
    '''
    Generate a random initial point for optimization.

    Returns:
    - List[int]: A list containing two random integers representing the initial point coordinates.
    '''

    return [random.randint(-50, 50), random.randint(-50, 50)]

def norma(vector) -> float:
    '''
        Calculates the Euclidean norm (L2 norm) of a vector.

        Parameters:
        - vector (list or numeric): Input vector or a single numeric value.

        Returns:
        - float: Euclidean norm of the input vector.
    '''
    # Ensure that the input is treated as a vector (list)
    if(isinstance(vector,list)==False):
        vector=[vector]
    
    # Calculate the sum of squares of vector components
    sum_of_squares = sum(x**2 for x in vector)

    # Calculate the Euclidean norm (L2 norm)
    norm = sum_of_squares ** 0.5

    return norm

def unimodalni_interval(start_point: float, step_size: float, target_function: 'function') -> (float,float):
    '''
        Find a unimodal interval around the given start_point for the target function.

        Args:
        - start_point (float): The initial point for the search.
        - step_size (float): The step size used in the search.
        - target_function (class function): The target function to find the unimodal interval for.

        Returns:
        - Tuple([)float, float): A tuple containing the left and right boundaries of the unimodal interval.
    '''
    l = start_point - step_size
    r= start_point + step_size
    m = start_point
    print(l,r,m)
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
        - a (Optional[float]): The left boundary of the initial interval.
        - b (Optional[float]): The right boundary of the initial interval.
        - precision (float): The desired precision of the minimum.
        - target_function (Callable[[float], float]): The target function to minimize.
        - start_point (Optional[float]): If provided, the function will use this point to determine the initial interval.
        Returns:
        - Tuple(float, float, float): A tuple containing the left boundary, right boundary, and the minimum point.
    '''
    k = 0.5 * (math.sqrt(5) - 1)
     # Either use the predefined interval provided by arguments a and b or determine it from the start_point
    if  start_point != None:
        a, b = unimodalni_interval(start_point, 1.0, target_function)
        print(f"Koristeci unimodalni intrval dobili smo vrijednsoti a: {a}, b: {b}")
        c = b - k * (b - a)
        d = a + k * (b - a)
    else:
        c = b - k * (b - a)
        d = a + k * (b - a)

    fc = target_function(c)
    fd = target_function(d)

    while abs(b - a) > precision:
        print(f"a = {a}, b = {b}, c = {c}, d = {d}")
        print(f"f(a) = {target_function(a)}, f(b) = {target_function(b)}, f(c) = {fc}, f(d) = {fd}")

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

    # Return the result - two possibilities
    return a, b , (a + b) / 2  
    
def koordinatno_pretrazivanje(x0: list, e:float, target_function: 'function') -> list:
    '''
    Coordinate search algorithm for unconstrained optimization.

    Args:
    - x0 (List[float]): Initial guess for the minimum point.
    - e (float or List[float]): If a float, it is the precision for all dimensions.
                                If a list, it specifies the precision for each dimension separately.
    - target_function (function): The target function to minimize.
    Returns:
    - List[float]: The minimum point found by the algorithm.
    '''

    n = len(x0)
    # If e is a float, convert it to a list with the same precision for each dimension
    if(isinstance(e,list) == False):
        e = [e]*n
    
    x = [xi for xi in x0]
    
    while True:
        xs = x.copy()
        for i in range(n):
            e_i = [0.0] * n
            e_i[i] = 1.0
           
            def modified_function(lambda_val):
                return target_function([x[j] + lambda_val * e_i[j] for j in range(n)])
           
            _, _, lambda_opt = zlatni_rez(None, None, e[i], modified_function,x[i])
            
            x = [x[j] + lambda_opt * e_i[j] for j in range(n)]
        # Check if the change in the vector x is smaller than the specified precision 
        if(abs(norma(x)-norma(xs)) > norma(e)):
            break
    
    return x

def centroid(x: list,h: int) -> list:
    '''
        Calculates the centroid of a set of points excluding the point at index h.

        Parameters:
        - x (list): List of points where each point is represented as a list of coordinates.
        - h (int): Index of the point to be excluded when calculating the centroid.

        Returns:
        - list: Centroid coordinates.
    '''

    n = len(x) - 1  # Number of points excluding the one at index h
    dimenzionalnost = len(x[0])  # Dimensionality of the points

    suma_dimenzija = [0] * dimenzionalnost
    
    # Sum up the coordinates of all points (excluding the point at index h)
    for i, tocka in enumerate(x):
        if(i!=h):
            for j in range(dimenzionalnost):
                suma_dimenzija[j] += tocka[j]

    # Calculate the centroid coordinates
    centroid_dimenzije = [s / n for s in suma_dimenzija]

    return centroid_dimenzije

def pocetni_simpleks(x0: list, step: int) -> list:
    '''
    Generates an initial simplex for the Nelder-Mead optimization algorithm.

    Parameters:
    - x0 (list): Initial point around which the simplex is generated.
    - step (int): Step size for creating the simplex.

    Returns:
    - list: List of points forming the initial simplex.
    '''

    n = len(x0)
    simplex = [x0]

    # Generate points for the simplex by perturbing each coordinate of the initial point
    for i in range(n):
        point = x0.copy()
        point[i] += step
        simplex.append(point)

    return simplex

def najbolja_i_najgora_tocka(simplex: list, target_function: 'function') -> (list,int,list,int):
    '''
        Finds the best and worst points in the simplex based on the target function values.

        Parameters:
        - simplex (list): List of points representing the simplex.
        - target_function ('function'): Objective function to evaluate the points.

        Returns:
        - (list, int, list, int): Tuple containing the best point, its index, the worst point, and its index.
    '''
    najbolja_tocka = simplex[0]
    najbolja_vrijednost = target_function(najbolja_tocka)
    najbolja_indeks = 0
    najgora_tocka = simplex[0]
    najgora_vrijednost = target_function(najgora_tocka)
    najgora_indeks = 0

    for i, tocka in enumerate(simplex):
        vrijednost = target_function(tocka)

        # Check if the current point is better than the current best point
        if vrijednost < najbolja_vrijednost:
            najbolja_tocka = tocka
            najbolja_vrijednost = vrijednost
            najbolja_indeks = i

        # Check if the current point is worse than the current worst point
        if vrijednost > najgora_vrijednost:
            najgora_tocka = tocka
            najgora_vrijednost = vrijednost
            najgora_indeks = i


    return najbolja_tocka,najbolja_indeks, najgora_tocka, najgora_indeks

def simpleks_nelder_mead(x0=None,simpleks_step=1, alpha=1, beta=0.5, gamma=2, sigma=0.5, epsilon=1e-6,target_function=None):
    n = len(x0)
    brojac_zaustavljanja = 0
    if(isinstance(epsilon,list) == False):
        epsilon = [epsilon]

    simplex = pocetni_simpleks(x0, simpleks_step)

    while True:
        xl,l, xh,h = najbolja_i_najgora_tocka(simplex, target_function)

        centroid_point = centroid(simplex,h)
        print(f"Tocke simplexa su xh(najgora): {xh} i xl(najbolja): {xl}")
        print(f"Centroid: {centroid_point}, F(Centroid): {target_function(centroid_point)}")

        xr=[ (1+alpha)*centroid_point[i] - alpha*xh[i] for i in range(n)]                       #REFLEKSIJA

        if target_function(xr) < target_function(xl):
            xe = [centroid_point[i] + gamma * (xr[i] - centroid_point[i]) for i in range(n)]    #EKSPANZIJA
            if target_function(xe) < target_function(xl):
                simplex[h] = xe.copy()
            else:
                simplex[h] = xr.copy()
        else:
            if all(target_function(xr) > target_function(simplex[j]) for j in range(n) if j != h):
                if target_function(xr) < target_function(xh):
                    simplex[h] = xr.copy()
                xk = [centroid_point[i] + beta * (xh[i] - centroid_point[i]) for i in range(n)] #KONTRAKCIJA
                print("Tocka KONTRAKCIJA tj. xk",xk)
                if target_function(xk) < target_function(xh):
                    simplex[h] = xk.copy()
                else:
                    #pomakni sve tocke prema X[l];
                    for i in range(n + 1):
                        if(i==l):
                            continue
                        for j in range(len(simplex[i])):
                            simplex[i][j] = simplex[l][j] + sigma* (simplex[i][j]-simplex[l][j])
            else:
                simplex[h] = xr.copy()
        
        #uvjet zaustavljanja
        pom=0
        for j in range(len(simplex)):
            pom += (target_function(simplex[j]) - target_function(centroid_point))**2

        result = 1/2 * pom
        uvjet_zaust = math.sqrt(result)

        if uvjet_zaust < max(epsilon):
            suma = 0
            optimalno_rj = 0
            #računa se centroid svih točaka simpleksa
            for i,x_i in enumerate(simplex):
                    for j in x_i:
                        suma+=j
            optimalno_rj = suma/len(simplex)
            break
        brojac_zaustavljanja+=1
    print()
    return optimalno_rj,simplex[h]

def provjeri_uvjet(Dx: list, epsilon: list) -> bool:
    '''    
        Checks if any component of Dx (change in position) is greater than or equal to the corresponding component in epsilon.

        Parameters:
        - Dx (list): List representing the change in position.
        - epsilon (list): List representing the epsilon values for each component.
        Returns:
        - bool: True if any component of Dx is greater than or equal to the corresponding component in epsilon, else False.
    '''

    for dx_i,e_i in zip(Dx,epsilon):
        if(dx_i >= e_i):
            return True
    return False

def istrazi(xP: list, Dx: list, target_function: 'function', *args) -> list:
    '''
        Explore the neighborhood of the point xP by adjusting each component of xP based on the target function.

        Parameters:
        - xP (list): The current point.
        - Dx (list): The change in position.
        - target_function (function): The target function to minimize.
        - *args: Additional arguments to be passed to the target function.

        Returns:
        - list: The new point after exploration.
    '''
    # Create a copy to keep the original xP unchanged
    x = copy.deepcopy(xP)
    n = len(x)

    for i in range(n):
        P = target_function(x, *args)
        x[i] += Dx[i]
        N = target_function(x, *args)

        if N > P:
            x[i] -= 2 * Dx[i]
            N = target_function(x, *args)
            if N > P:
                x[i] += Dx[i]
    
    return x

def hooke_jeeves(x0: list, Dx: list, epsilon: float, target_function: 'function',*args) -> list:
    '''
    Hooke-Jeeves pattern search algorithm for unconstrained optimization.

    Parameters:
    - x0 (list): Initial guess.
    - Dx (list): Initial step size.
    - epsilon (float): Tolerance for stopping criteria.
    - target_function (function): The target function to minimize.
    - *args: Additional arguments to be passed to the target function.

    Returns:
    - list: The optimal point.
    '''
    
    xP = [xi for xi in x0]
    xB = [xi for xi in x0]
    epsilon_list = [epsilon] * len(x0)

    # Use a flag to enter the while loop
    while True:
        xN = istrazi(xP, Dx, target_function, *args)

        FP = target_function(xP,*args)
        FN = target_function(xN,*args)

        if FN < FP:
            xP = [2 * xN[i] - xB[i] for i in range(len(xN))]
            xB = [xi for xi in xN]
        else:
            Dx = [0.5 * dx for dx in Dx]
            xP = [xi for xi in xB]


        if(provjeri_uvjet(Dx,epsilon_list) == False):
            break
       
    return xB

