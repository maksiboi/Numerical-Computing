import constrained_optimization

def write_to_file(file_path, data):
    with open(file_path, "w") as file:
        for line in data:
            file.write(str(line) + "\n")

# Primjer upotrebe
def f1(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

def f2(x):
    return (x[0]-4)**2 + 4*(x[1]-2)**2

def implicit1(X):
    return X[1] - X[0] 

def implicit2(X):
    return 2 - X[0] 

def implicit3(X):
    return 3 - X[0] - X[1]

def implicit4(X):
    return 3 + 1.5*X[0] - X[1]

def eksplicit1(X):
    return X[1] - 1

def f4(x):
    return (x[0]-3)**2 + x[1]**2

def prvi_zad():
    X0_1 = [-1.9, 2]
    eksplicit = [[-100, -100], [100, 100]]

    result_1 = constrained_optimization.box_method(X0_1, F=f1, g=[implicit1, implicit2], eksplicit=eksplicit)
    print("Funckija 1:")
    print("Optimal solution:", result_1)
    print("Objective value:", f1(result_1))

    print()
    X0_2 = [0.1,0.3]
    result_2 =  constrained_optimization.box_method(X0_2, F=f2, g=[implicit1, implicit2], eksplicit=eksplicit)
    print("Funckija 2:")
    print("Optimal solution:", result_2)
    print("Objective value:", f2(result_2),"\n")
    output_data = [
        "Funckija 1:",
        "Starting point: {}" .format(X0_1),
        "Optimal solution: {}".format(result_1),
        "Objective value: {}".format(f1(result_1)),
        "",
        "Funckija 2:",
        "Starting point: {}" .format(X0_2),
        "Optimal solution: {}".format(result_2),
        "Objective value: {}".format(f2(result_2))
    ]
    write_to_file("DZ4/results/1_zad_rj.txt", output_data)

def drugi_zad():
    X0_1 = [0,3]
    result_1 = constrained_optimization.penalty_barrier(X0_1, f=f1, g=[implicit1, implicit2], h=None)
    print("Funckija 1:")
    print("Optimal solution:", result_1)
    print("Objective value:", f1(result_1))

    X0_2 = [0.1,0.3]
    
    print()
    result_2 = constrained_optimization.penalty_barrier(X0_2, f=f2, g=[implicit1, implicit2], h=None)
    print("Funckija 2:")
    print("Optimal solution:", result_2)
    print("Objective value:", f2(result_2))

    output_data = [
        "Funckija 1:",
        "Starting point: {}" .format(X0_1),
        "Optimal solution: {}".format(result_1),
        "Objective value: {}".format(f1(result_1)),
        "",
        "Funckija 2:",
        "Starting point: {}" .format(X0_2),
        "Optimal solution: {}".format(result_2),
        "Objective value: {}".format(f2(result_2))
    ]
    write_to_file("DZ4/results/2_zad_rj.txt", output_data)

def treci_zad():
    X0 = [5,5]
    
    #za [5,5] koja ne zadovoljava ogranicenja nejednakosti
    #algoritam unutarnje tocke vrati [1.0000019073486328, 1.0000019073486328]
    result_1 = constrained_optimization.penalty_barrier(X0, f=f4, g=[implicit3, implicit4], h=[eksplicit1])
    print("Funckija 4:")
    print("Optimal solution:", result_1)
    print("Objective value:", f4(result_1))

    output_data = [
        "Funckija 4:",
        "Starting point: {}" .format(X0),
        "Optimal solution: {}".format(result_1),
        "Objective value: {}".format(f4(result_1)),
    ]
    write_to_file("DZ4/results/3_zad_rj.txt", output_data)

def main():
    prvi_zad()
    
    drugi_zad()

    treci_zad()

if __name__ == "__main__":
    main()