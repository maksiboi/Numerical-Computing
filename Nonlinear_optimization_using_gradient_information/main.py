import math
import sys
sys.path.append('')

from DZ1.matrica import Matrica
import nonlinear_optimization_using_gradient

broj_poziva_f1 = 0
broj_poziva_grad_f1 = 0
broj_poziva_hess_f1 = 0
broj_poziva_f2 = 0
broj_poziva_grad_f2 = 0
broj_poziva_hess_f2 = 0
broj_poziva_f3 = 0
broj_poziva_grad_f3 = 0
broj_poziva_hess_f3 = 0
broj_poziva_f4 = 0
broj_poziva_grad_f4 = 0
broj_poziva_hess_f4 = 0
broj_poziva_f5 = 0
broj_poziva_jacobiana_f1 = 0
broj_poziva_jacobiana_f5 = 0
broj_poziva_M = 0
broj_poziva_jacobiana_M = 0



def zapis_u_datoteku_gradijent(prvi,drugi,treci,cetvrti,peti,sesti,ime_datoteke):
        with open(ime_datoteke, 'w') as datoteka:
            datoteka.write(f"{prvi} \n{drugi}\n{treci}\n{cetvrti}\n{peti}\n{sesti} ")

def zapis_u_datoteku(prvi, drugi, treci, cetvrti, ime_datoteke):
    with open(ime_datoteke, 'w') as datoteka:
        datoteka.write(f"{prvi} {drugi}\n{treci} {cetvrti}")

def zapis_u_datoteku_jakobijan(input_data, ime_datoteke):
    with open(ime_datoteke, 'w') as datoteka:
        # Write elements in groups of three per line
        for i in range(0, len(input_data), 3):
            datoteka.write(f"{input_data[i]} {input_data[i + 1]} {input_data[i + 2]}\n")

def zapis_rezultata(lista_stringova, putanja_do_datoteke):
    try:
        with open(putanja_do_datoteke, 'w',encoding='UTF-8') as datoteka:
            for string in lista_stringova:
                datoteka.write(f"{string}\n")
        print(f"Podaci su uspješno zapisani u datoteku: {putanja_do_datoteke}")
    except Exception as e:
        print(f"Greška prilikom pisanja u datoteku: {e}")

def f1(x):
    global broj_poziva_f1
    broj_poziva_f1 += 1
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

def gradient_f1(x):
    global broj_poziva_grad_f1
    broj_poziva_grad_f1 += 1
    po_x1 = -400* x[0]* (-x[0]**2+x[1]) + 2*x[0] - 2
    po_x2 = -200*x[0]**2 + 200*x[1]

    return [po_x1,po_x2]

def hess_f1(x,dat):
    global broj_poziva_hess_f1
    broj_poziva_hess_f1 +=1
    prvi = -400 * (-3*x[0]**2 + x[1]) + 2
    drugi = -400 * x[0]
    treci = -400 * x[0]
    cetvrti = 200
    zapis_u_datoteku(prvi,drugi,treci,cetvrti,dat)

def jacobian_f1(x):
    global broj_poziva_jacobiana_f1
    broj_poziva_jacobiana_f1 += 1
    J = Matrica()
    prvi = -20 * x[0]
    drugi = 10
    treci = -1
    cetvrti = 0
    zapis_u_datoteku(prvi,drugi,treci,cetvrti,'DZ3/jacobian/jacobian_f1.txt')
    J.citaj_iz_datoteke('DZ3/jacobian/jacobian_f1.txt')
    return J

def G1(x):
    G = Matrica()
    prvi = 10*(x[1]-x[0]**2)
    treci = 1 - x[0]
    nonlinear_optimization_using_gradient.zapis_gradijent_u_datoteku(prvi,treci,'DZ3/vektorska_jednadzba/g1.txt')
    G.citaj_iz_datoteke('DZ3/vektorska_jednadzba/g1.txt')
    return G

def f2(x):
    global broj_poziva_f2 
    broj_poziva_f2 += 1
    return (x[0]-4)**2 + 4*(x[1]-2)**2

def gradient_f2(x):
    global broj_poziva_grad_f2
    broj_poziva_grad_f2 += 1
    po_x1 = 2* x[0] - 8
    po_x2 = 8* x[1] - 16

    return [po_x1,po_x2]

def hess_f2(x,dat):
    global broj_poziva_hess_f2
    broj_poziva_hess_f2 += 1
    prvi = 2
    drugi = 0
    treci = 0
    cetvrti = 8
    zapis_u_datoteku(prvi,drugi,treci,cetvrti,dat)

def f3(x):
    global broj_poziva_f3
    broj_poziva_f3 += 1
    return (x[0] - 2)**2 + (x[1] + 3)**2

def gradient_f3(x):
    global broj_poziva_grad_f3
    broj_poziva_grad_f3 += 1
    po_x1 = 2*x[0] - 4
    po_x2 = 2*x[1] + 6

    return [po_x1,po_x2]

def hess_f3(x,dat):
    global broj_poziva_hess_f3
    broj_poziva_hess_f3+=1
    prvi = 2
    drugi = 0
    treci = 0
    cetvrti = 2
    zapis_u_datoteku(prvi,drugi,treci,cetvrti,dat)
    
def f4(x):
    global broj_poziva_f4
    broj_poziva_f4 += 1
    return (1/4)*x[0]**4 - x[0]**2 + 2*x[0] + (x[1] - 1)**2

def gradient_f4(x):
    global broj_poziva_grad_f4
    broj_poziva_grad_f4 += 1
    po_x1 = x[0]**3 - 2*x[0] + 2
    po_x2 = 2*x[1] - 2
    
    return[po_x1,po_x2]

def hess_f4(x,dat):
    global broj_poziva_hess_f4
    broj_poziva_hess_f4 += 1
    prvi = 3*x[0]**2 - 2
    drugi = 0
    treci = 0
    cetvrti = 2
    zapis_u_datoteku(prvi,drugi,treci,cetvrti,dat)

def f5(x):
    global broj_poziva_f5
    broj_poziva_f5 += 1
    prvi_clan = (x[0]**2 + x[1]**2 -1)**2 
    drugi_clan = (x[1]-x[0]**2)**2
    return prvi_clan + drugi_clan

def G5(x):
    G = Matrica()
    prvi = x[0]**2 + x[1]**2 - 1
    treci = x[1] - x[0] ** 2
    nonlinear_optimization_using_gradient.zapis_gradijent_u_datoteku(prvi,treci,'DZ3/vektorska_jednadzba/g5.txt')
    G.citaj_iz_datoteke('DZ3/vektorska_jednadzba/g5.txt')
    return G

def jacobian_f5(x):
    global broj_poziva_jacobiana_f5
    broj_poziva_jacobiana_f5 += 1
    J = Matrica()
    prvi = 2 * x[0]
    drugi = 2 * x[1]
    treci = -2 * x[0]
    cetvrti = 1
    zapis_u_datoteku(prvi,drugi,treci,cetvrti,'DZ3/jacobian/jacobian_f1.txt')
    J.citaj_iz_datoteke('DZ3/jacobian/jacobian_f1.txt')
    return J

def M(x):
    global broj_poziva_M
    broj_poziva_M += 1
    prvi = x[0]*math.exp(x[1]) + x[2] - 3
    drugi =  x[0]*math.exp(x[1]*2) + x[2] - 4
    treci =  x[0]*math.exp(x[1]*3) + x[2] - 4
    cetvrti = x[0]*math.exp(x[1]*5) + x[2] - 5
    peti = x[0]*math.exp(x[1]*6) + x[2] - 6
    sesti = x[0]*math.exp(x[1]*7) + x[2] -8 

    return prvi**2 + drugi**2 + treci**2 + cetvrti**2 + peti**2 + sesti**2

def G_m(x):
    G = Matrica()
    prvi = x[0]*math.exp(x[1]) + x[2] - 3
    drugi =  x[0]*math.exp(x[1]*2) + x[2] - 4
    treci =  x[0]*math.exp(x[1]*3) + x[2] - 4
    cetvrti = x[0]*math.exp(x[1]*5) + x[2] - 5
    peti = x[0]*math.exp(x[1]*6) + x[2] - 6
    sesti = x[0]*math.exp(x[1]*7) + x[2] -8 
    zapis_u_datoteku_gradijent(prvi,drugi,treci,cetvrti,peti,sesti,'DZ3/vektorska_jednadzba/g6.txt')
    G.citaj_iz_datoteke('DZ3/vektorska_jednadzba/g6.txt')
    return G

def J_m(x):
    global broj_poziva_jacobiana_M
    broj_poziva_jacobiana_M += 1
    J = Matrica()
    prvi_1 = math.exp(x[1])
    drugi_1 = x[0]*math.exp(x[1])
    treci_1 = 1
    
    prvi_2 = math.exp(2 * x[1])
    drugi_2 = 2* x[0]*math.exp(2*x[1])
    treci_2 = 1

    prvi_3 = math.exp(3 * x[1])
    drugi_3 = 3* x[0]*math.exp(3*x[1])
    treci_3 = 1

    prvi_4 = math.exp(5 * x[1])
    drugi_4 = 5* x[0]*math.exp(5*x[1])
    treci_4 = 1

    prvi_5 = math.exp(6 * x[1])
    drugi_5 = 6* x[0]*math.exp(6*x[1])
    treci_5 = 1

    prvi_6 = math.exp(7 * x[1])
    drugi_6= 7* x[0]*math.exp(7*x[1])
    treci_6 = 1

    input  = [prvi_1,drugi_1,treci_1,prvi_2,drugi_2,treci_2,prvi_3,drugi_3,treci_3,prvi_4,drugi_4,treci_4,prvi_5,drugi_5,treci_5,prvi_6,drugi_6,treci_6]

    zapis_u_datoteku_jakobijan(input,'DZ3/jacobian/jacobian_M.txt')
    J.citaj_iz_datoteke('DZ3/jacobian/jacobian_M.txt')
    return J


def prvi():
    global broj_poziva_f3,broj_poziva_grad_f3
    # Set initial point, precision, and decide whether to use golden ratio method
    initial_point = [0, 0]
    precision = 1e-6
    use_golden_ratio = False

    broj_poziva_f3 = 0
    broj_poziva_grad_f3 = 0
    tocka,mini_f,stag = nonlinear_optimization_using_gradient.gradijetni_spust(f3,gradient_f3, initial_point, precision, use_golden_ratio)
    uz_1 = f"Određivanje minmuma funkcije 3 BEZ određivanje optimalnog iznosa koraka"
    stag = f"{stag}"
    prvi_1 = f"Optimal point: {tocka}"
    drugi_1 =f"Optimal value: {mini_f}"
    treci_1 = f"Broj poziva  funkcije cilja {broj_poziva_f3}"
    cet_1 = f"Broj poziva računanja gradijenta {broj_poziva_grad_f3}"

    ###################################################################################

    initial_point = [0, 0]
    precision = 1e-6
    use_golden_ratio = True

    broj_poziva_f3 = 0
    broj_poziva_grad_f3 = 0
    tocka,mini_f = nonlinear_optimization_using_gradient.gradijetni_spust(f3,gradient_f3, initial_point, precision, use_golden_ratio)

    novi_red = ""
    uz = f"Određivanje minmuma funkcije 3 UZ određivanje optimalnog iznosa koraka"
    prvi = f"Optimal point: {tocka}"
    drugi =f"Optimal value: {mini_f}"
    treci = f"Broj poziva  funkcije cilja {broj_poziva_f3}"
    cet = f"Broj poziva računanja gradijenta {broj_poziva_grad_f3}"
    rjesenje = [uz_1,stag,prvi_1,drugi_1,treci_1,cet_1,novi_red,uz,prvi,drugi,treci,cet]
    broj_poziva_f3 = 0
    broj_poziva_grad_f3 = 0
    zapis_rezultata(rjesenje,'DZ3/output/prvi_zad.txt')

def drugi():
    global broj_poziva_f1,broj_poziva_grad_f1,broj_poziva_hess_f1,broj_poziva_f2,broj_poziva_grad_f2,broj_poziva_hess_f2
    initial_point = [-1.9, 2]
    precision = 1e-6
    use_golden_ratio = True

    broj_poziva_f1 = 0
    broj_poziva_grad_f1 = 0
    tocka,mini_f,stagnira = nonlinear_optimization_using_gradient.gradijetni_spust(f1,gradient_f1, initial_point, precision, use_golden_ratio)
    
    uz_1 = (f"FUNKCIJA 1: Gradijentni spust metoda s korištenjem linijskog pretraživanja: ")
    stag = (f"{stagnira}")
    prvi_1 = ("Optimal point:", tocka)
    drugi_1 = ("Optimal value:", mini_f)
    treci_1 = (f"Broj poziva  funkcije cilja {broj_poziva_f1}")
    cet_1 = (f"Broj poziva računanja gradijenta {broj_poziva_grad_f1} \n")
    novi_red = ""

    broj_poziva_f1 = 0
    broj_poziva_grad_f1 = 0
    broj_poziva_hess_f1 = 0
    ime_datoteke = 'DZ3/input/zad2_ulaz.txt'
    initial_point = [-1.9, 2]
    tocka,mini_f,stagnira = nonlinear_optimization_using_gradient.newton_raphson(f1,gradient_f1,hess_f1, initial_point, precision,ime_datoteke, use_golden_ratio)

    uz_2 = (f"FUNKCIJA 1: Newton-Raphsonova metoda s korištenjem linijskog pretraživanja: ")
    prvi_2 = ("Optimal point:", tocka)
    drugi_2 = ("Optimal value:", mini_f)
    treci_2 = (f"Broj poziva  funkcije cilja: {broj_poziva_f1}")
    cet_2 = (f"Broj poziva računanja gradijenta: {broj_poziva_grad_f1}")
    peti_2 = (f"Broj poziva računanja Hesseove matrice: {broj_poziva_hess_f1} \n")

    broj_poziva_f1 = 0
    broj_poziva_grad_f1 = 0
    broj_poziva_hess_f1 = 0

    broj_poziva_f2 = 0
    broj_poziva_grad_f2 = 0
    broj_poziva_hess_f2 = 0
    initial_point = [0.1, 0.3]
    tocka,mini_f = nonlinear_optimization_using_gradient.gradijetni_spust(f2,gradient_f2, initial_point, precision, use_golden_ratio)

    uz_3 = (f"FUNKCIJA 2: Gradijentni spust metoda s korištenjem linijskog pretraživanja: ")
    prvi_3 = ("Optimal point:", tocka)
    drugi_3 = ("Optimal value:", mini_f)
    treci_3 = (f"Broj poziva  funkcije cilja: {broj_poziva_f2}")
    cet_3 = (f"Broj poziva računanja gradijenta: {broj_poziva_grad_f2}\n")


    broj_poziva_f2 = 0
    broj_poziva_grad_f2 = 0
    broj_poziva_hess_f2 = 0
    initial_point = [0.1, 0.3]
    ime_datoteke = 'DZ3/input/zad2_ulaz.txt'
    tocka,mini_f,stagnira = nonlinear_optimization_using_gradient.newton_raphson(f2,gradient_f2,hess_f2, initial_point, precision,ime_datoteke, use_golden_ratio)

    uz_4 = (f"FUNKCIJA 2: Newton-Raphsonova metoda s korištenjem linijskog pretraživanja: ")
    prvi_4 = ("Optimal point:", tocka)
    drugi_4 = ("Optimal value:", mini_f)
    treci_4 = (f"Broj poziva  funkcije cilja: {broj_poziva_f2}")
    cet_4 = (f"Broj poziva računanja gradijenta: {broj_poziva_grad_f2}")
    peti_4 = (f"Broj poziva računanja Hesseove matrice: {broj_poziva_hess_f2} \n")
    broj_poziva_f2 = 0
    broj_poziva_grad_f2 = 0
    broj_poziva_hess_f2 = 0

    rjesenje = [uz_1,stag,prvi_1,drugi_1,treci_1,cet_1,novi_red,uz_2,prvi_2,drugi_2,treci_2,cet_2,peti_2,novi_red,uz_3,prvi_3,drugi_3,treci_3,cet_3,novi_red,uz_4,prvi_4,drugi_4,treci_4,cet_4,peti_4]
    zapis_rezultata(rjesenje,'DZ3/output/drugi_zad.txt')
 
def treci():
    global broj_poziva_f4,broj_poziva_grad_f4,broj_poziva_hess_f4
    broj_poziva_f4 = 0
    broj_poziva_grad_f4 = 0
    broj_poziva_hess_f4 = 0
    precision = 1e-6
    use_golden_ratio = False
    ime_datoteke = 'DZ3/input/zad3_ulaz.txt'
    initial_point = [3,3]
    tocka,mini_f,stag = nonlinear_optimization_using_gradient.newton_raphson(f4,gradient_f4,hess_f4, initial_point, precision,ime_datoteke, use_golden_ratio)

    uz_1 = (f"FUNKCIJA 4: Newton-Raphsonova metoda BEZ korištenjem linijskog pretraživanja u početnoj točki : {initial_point}")
    prvi_1 = ("Optimal point:", tocka)
    drugi_1 = ("Optimal value:", mini_f)
    treci_1 = (f"Broj poziva  funkcije cilja: {broj_poziva_f4}")
    cet_1 = (f"Broj poziva računanja gradijenta: {broj_poziva_grad_f4}")
    peti_1 = (f"Broj poziva računanja Hesseove matrice: {broj_poziva_hess_f4} \n")
    novi_red = ""

    broj_poziva_f4 = 0
    broj_poziva_grad_f4 = 0
    broj_poziva_hess_f4 = 0
    precision = 1e-6
    use_golden_ratio = False
    ime_datoteke = 'DZ3/input/zad3_ulaz.txt'
    initial_point = [1,2]
    tocka,mini_f,stagi = nonlinear_optimization_using_gradient.newton_raphson(f4,gradient_f4,hess_f4, initial_point, precision,ime_datoteke, use_golden_ratio)

    uz_2 = (f"FUNKCIJA 4: Newton-Raphsonova metoda BEZ korištenjem linijskog pretraživanja u početnoj točki : {initial_point}")
    stag2 = f"{stagi}"
    prvi_2 = ("Optimal point:", tocka)
    drugi_2 = ("Optimal value:", mini_f)
    treci_2 = (f"Broj poziva  funkcije cilja: {broj_poziva_f4}")
    cet_2 = (f"Broj poziva računanja gradijenta: {broj_poziva_grad_f4}")
    peti_2 = (f"Broj poziva računanja Hesseove matrice: {broj_poziva_hess_f4} \n")

    #UZ KORISTENJE ZLATNOG REZA
    broj_poziva_f4 = 0
    broj_poziva_grad_f4 = 0
    broj_poziva_hess_f4 = 0
    precision = 1e-6
    use_golden_ratio = True
    ime_datoteke = 'DZ3/input/zad3_ulaz.txt'
    initial_point = [3,3]
    tocka,mini_f,stag = nonlinear_optimization_using_gradient.newton_raphson(f4,gradient_f4,hess_f4, initial_point, precision,ime_datoteke, use_golden_ratio)

    uz_3 = (f"FUNKCIJA 4: Newton-Raphsonova metoda UZ korištenjem linijskog pretraživanja u početnoj točki : {initial_point}")
    prvi_3 = ("Optimal point:", tocka)
    drugi_3 = ("Optimal value:", mini_f)
    treci_3 = (f"Broj poziva  funkcije cilja: {broj_poziva_f4}")
    cet_3 = (f"Broj poziva računanja gradijenta: {broj_poziva_grad_f4}")
    peti_3 = (f"Broj poziva računanja Hesseove matrice: {broj_poziva_hess_f4} \n")
    novi_red = ""

    broj_poziva_f4 = 0
    broj_poziva_grad_f4 = 0
    broj_poziva_hess_f4 = 0
    precision = 1e-6
    use_golden_ratio = True
    ime_datoteke = 'DZ3/input/zad3_ulaz.txt'
    initial_point = [1,2]
    tocka,mini_f,stag = nonlinear_optimization_using_gradient.newton_raphson(f4,gradient_f4,hess_f4, initial_point, precision,ime_datoteke, use_golden_ratio)

    uz_4 = (f"FUNKCIJA 4: Newton-Raphsonova metoda UZ korištenjem linijskog pretraživanja u početnoj točki : {initial_point}")
    prvi_4 = ("Optimal point:", tocka)
    drugi_4 = ("Optimal value:", mini_f)
    treci_4 = (f"Broj poziva  funkcije cilja: {broj_poziva_f4}")
    cet_4 = (f"Broj poziva računanja gradijenta: {broj_poziva_grad_f4}")
    peti_4 = (f"Broj poziva računanja Hesseove matrice: {broj_poziva_hess_f4} \n")


    rjesenje = [uz_1,prvi_1,drugi_1,treci_1,cet_1,peti_1,novi_red,uz_2,stag2,prvi_2,drugi_2,treci_2,cet_2,peti_2,novi_red,uz_3,prvi_3,drugi_3,treci_3,cet_3,peti_3,novi_red,uz_4,prvi_4,drugi_4,treci_4,cet_4,peti_4]
    zapis_rezultata(rjesenje,'DZ3/output/treci_zad.txt')

def cetvrti():
    global broj_poziva_jacobiana_f1
    broj_poziva_jacobiana_f1 = 0
    precision = 1e-6
    use_golden_ratio = False
    initial_point = [-1.9,2]
    tocka,mini_f,stag= nonlinear_optimization_using_gradient.gauss_newton(f1, G1, jacobian_f1, initial_point, precision, use_golden_ratio)
    
    uz_1 = f"Gauss-Newton uz čitav iznos pomaka"
    prvi_1 = f"Optimal point:{tocka}"
    drugi_1 = f"Optimal value: {mini_f}"
    treci_1 = f"Broj izracuna Jakobijana: {broj_poziva_jacobiana_f1}"
    broj_poziva_jacobiana_f1 = 0
    rjesenje = [uz_1,prvi_1,drugi_1,treci_1]
    zapis_rezultata(rjesenje,'DZ3/output/cetvrti_zad.txt')

def peti():
    global broj_poziva_f5,broj_poziva_jacobiana_f5

    broj_poziva_jacobiana_f5=0
    precision = 1e-6
    use_golden_ratio = False
    initial_point = [ [-2,2], [2,2], [2,-2] ]
    rjesenje = []
    rjesenje.append("Gauss-Newton BEZ određivanja optimalnog pomaka")
    for poc_toc in initial_point:
            tocka,mini_f,stag = nonlinear_optimization_using_gradient.gauss_newton(f5, G5, jacobian_f5, poc_toc, precision, use_golden_ratio)
            rjesenje.append(f"{stag}")
            rjesenje.append(f"Pocetna tocka:  {poc_toc}")
            rjesenje.append(f"Optimal point: {tocka}")
            rjesenje.append(f"Optimal value:  {mini_f}")
            rjesenje.append(f"Broj izracuna Jakobijana: {broj_poziva_jacobiana_f5}")
            rjesenje.append("")
            broj_poziva_jacobiana_f5=0
    
    rjesenje.append("Gauss-Newton UZ određivanja optimalnog pomaka")
    broj_poziva_f5 = 0
    use_golden_ratio = True
    for poc_toc in initial_point:
        tocka,mini_f,stag= nonlinear_optimization_using_gradient.gauss_newton(f5, G5, jacobian_f5, poc_toc, precision, use_golden_ratio)
        rjesenje.append(f"{stag}")
        rjesenje.append(f"Pocetna tocka:  {poc_toc}")
        rjesenje.append(f"Optimal point: {tocka}")
        rjesenje.append(f"Optimal value:  {mini_f}")
        rjesenje.append(f"Broj poziva  funkcije cilja: {broj_poziva_f5} ")
        rjesenje.append(f"Broj izracuna Jakobijana: {broj_poziva_jacobiana_f5}")
        rjesenje.append("")
        broj_poziva_f5 = 0
        broj_poziva_jacobiana_f5=0

    zapis_rezultata(rjesenje,'DZ3/output/peti_zad.txt')

def sesti():
    global broj_poziva_jacobiana_M
    broj_poziva_M=0
    broj_poziva_jacobiana_M = 0
    
    precision = 1e-6
    use_golden_ratio = False
    initial_point = [1,1,1]
    tocka,mini_f,stag = nonlinear_optimization_using_gradient.gauss_newton(M, G_m, J_m, initial_point, precision, use_golden_ratio)

    uz_1 = f"Optimizacija parametara model Gauss-Newtonovim postupakom"
    prvi_1 = f"Optimal point:{tocka}"
    drugi_1 = f"Optimal value: {mini_f}"
    cetvrti_1 = f"Broj izracuna Jakobijana: {broj_poziva_jacobiana_M}"
    
    rjesenje = [uz_1,"",prvi_1,drugi_1,"",cetvrti_1]
    zapis_rezultata(rjesenje,'DZ3/output/sesti_zad.txt')
    broj_poziva_jacobiana_M = 0


def main():
    
    prvi()
    
    #drugi()

    treci()

    cetvrti()

    peti()

    sesti()

if __name__ == "__main__":
    main()