import math
import random
import sys
import os
from typing import List


path_Nonlinear_optimization = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Nonlinear_optimization'))
sys.path.append(path_Nonlinear_optimization)

from nonlinear_optimization import hooke_jeeves


def box_method(X0: list, F: 'function', g: List['function'], eksplicit: List['function'], a=1.3, e=1e-6) -> List[float]:
    '''
    Box method for optimizing a function with constraints.

    Args:
    - X0 (List[float]): Initial guess for the minimum point.
    - F (function): The target function to minimize.
    - g (List[function]): List of constraint functions.
    - eksplicit (List[function]): List representing explicit constraints [Xd, Xg].
    - a (float): Reflection parameter.
    - e (float): Stopping criterion for the algorithm.

    Returns:
    - List[float]: The minimum point found by the algorithm.
    '''

    # Extract lower and upper bounds from explicit constraints
    Xd = eksplicit[0]    
    Xg = eksplicit[1]    

    # Initialize the center point Xc as a copy of the starting point X0
    Xc = list(X0)

    # Initialize a list to store the points
    points = [-1] * (len(X0) * 2)

    # Generate the initial set of 2*n points (simplex)
    for j in range(2*len(X0)):
        dot = []
        for i in range(len(X0)):
            # Generate random points within the bounds defined by Xd and Xg
            r=random.uniform(0, 1)
            dot.append(Xd[0] + r*(Xg[0]-Xd[0]))

        points[j] = dot
        
        # Counter to track the number of iterations for constraint satisfaction
        brojac = 0
        while True:
            brojac = 0
            for g_i in g:
                # If the point does not satisfy inequality constraints, move it towards the center
                if g_i(points[j]) < 0:  
                    points[j] = [0.5 * (point + xc) for point, xc in zip(points[j], Xc)]
                    brojac = 0
                else:
                    brojac += 1

            if brojac == len(g):
                break

        # Update the center point Xc based on the accepted points   
        Xc = [sum(coord) / 2 for coord in zip(points[j], Xc)]


    while True:
        # Find indices h, h2: F(X[h]) = max, F(X[h2]) = second worst
        h, h2 = max(enumerate(map(F, points)), key=lambda x: x[1])[0], sorted(range(len(points)), key=lambda i: F(points[i]))[-2]
        # Index of the best point
        l = min(range(len(points)), key=lambda i: F(points[i]))

        # Calculate Xc (excluding X[h])
        Xc = [sum(coord) / (len(points)-1) for coord in zip(*[point for point in points if point != points[h]])]
        
        # Reflection
        Xr = [(1 + a) * xc - a * xh for xc, xh in zip(Xc, points[h])]

        # Move towards the bounds of explicit constraints
        for i in range(len(X0)):
            if(Xr[i] < Xd[i]):
                Xr[i] = Xd[i]
            elif(Xr[i] > Xg[i]):
                Xr[i] = Xg[i]
        

        # Move towards implicit constraints (inequality constraints)
        while True:
            brojac = 0
            for g_i in g:
                if g_i(Xr) < 0:
                    Xr = [0.5 * (xr + xc) for xr, xc in zip(Xr, Xc)]
                    brojac = 0
                else:
                    brojac += 1

            if brojac == len(g):
                break

        # If the reflected point is worse than the second worst point, move it towards the center
        if F(Xr) > F(points[h2]):
            Xr = [0.5 * (xr + xc) for xr, xc in zip(Xr, Xc)]

        # Update the point at index h with the reflected point Xr
        points[h] = Xr

        # Check the stopping condition
        suma = 0 
        for point in points:
            suma += (F(point)-F(Xc))**2
        
        res = 0.5*suma
        uvjet_zaust = math.sqrt(res)
        if(uvjet_zaust < e):
            break

    # Return the best point  
    return points[l]


def G(x: List[float],g: List['function']) -> float:
    '''
    Auxiliary function for satisfying inequality constraints.

    Args:
    - x (List[float]): Point in the search space.
    - g (List[function]): List of inequality constraint functions.

    Returns:
    - float: The sum of negative values of inequality constraint functions at the given point x.
    '''
    res = 0
    for g_i in g:
        if(g_i(x) < 0):
            res -= g_i(x)

    return res

def F_penalty_barrier(X: List[float],f: 'function',g: List['function'], t: float,h: List['function'] or None) -> float:
    '''
    Auxiliary function for the penalty barrier algorithm.

    Args:
    - X (List[float]): Point in the search space.
    - f (function): Target function to minimize.
    - g (List[function]): List of inequality constraint functions.
    - t (float): Barrier parameter.
    - h (List[function] or None): List of equality constraint functions or None if not present.

    Returns:
    - float: The augmented objective function value with penalty for constraints.
    '''
    F_veliko = f(X)

    # Augmented function for barrier method
    for g_i in g:
        if(g_i(X)<=0):
            return math.inf
        else:
            F_veliko += (1/t) * math.log(g_i(X))
    
    # Augmented function for penalty method
    if(h != None):
        for h_i in h:
            if(h_i(X) != 0):
                F_veliko += t* (h_i(X)**2)
                
    return F_veliko

def interior_point(g: List['function'],x0: list,epsilon: float) -> List[float]:
    '''
    Interior point method for solving constrained optimization problems.

    Args:
    - g (List[function]): List of inequality constraint functions.
    - x0 (List[float]): Initial guess for the solution.
    - epsilon (float): Convergence criterion.

    Returns:
    - List[float]: The solution point after applying the interior point method.
    '''
    x=x0
    Dx = [0.5] * len(x0)

    while True:
        xs = x.copy()
        
        # Use Hooke-Jeeves method to find a feasible point
        x = hooke_jeeves(x, Dx, epsilon, G, g)
        
        # Calculate the distance between the current and previous points
        distance = math.sqrt(sum((x_i - xs_i) ** 2 for x_i, xs_i in zip(x, xs)))
        
        # Check for convergence based on the distance
        if distance < epsilon:
            break

    return x    

def penalty_barrier(X0: list,f: 'function',g: List['function'],h: List['function'],t0=1,e=1e-6) -> List[float]:
    '''
    Penalty-barrier method for solving constrained optimization problems.

    Args:
    - X0 (List[float]): Initial guess for the solution.
    - f (function): Target function to minimize.
    - g (List[function]): List of inequality constraint functions.
    - h (List[function]): List of equality constraint functions.
    - t0 (float): Initial penalty parameter.
    - e (float): Convergence criterion.

    Returns:
    - List[float]: The solution point after applying the penalty-barrier method.
    '''
    X=X0
    t=t0

    # Check if the initial point satisfies all inequality constraints
    flag = True
    for g_i in g:
        if(g_i(X)<0):
            flag = False
    
    # If the initial point satisfies the inequality constraints, set it as the penalty point
    if flag:
        Xp = X0.copy()
    else:
        # If not, find a feasible point using the interior point method
        Xp = interior_point(g, X0, e)

    Dx = [0.5] * len(X0)
    while True:
        Xs = X.copy()

        # Use Hooke-Jeeves method to find a point minimizing the penalty-barrier function
        X = hooke_jeeves(Xp, Dx, e, F_penalty_barrier, f, g, t, h)
        
        # Increase the penalty parameter
        t = t * 10

        # Calculate the distance between the current and previous points
        distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(X, Xs)))

        # Check for convergence based on the distance
        if(distance < e):
            break

    return X




