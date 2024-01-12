import math
import csv
import nonlinear_optimization


broj_poziva_prvi_zad = 0
broj_poziva_f1 = 0
broj_poziva_f2 = 0
broj_poziva_f3 = 0
broj_poziva_f4 = 0
broj_poziva_schaffer_f6 = 0

def parse_text_file(file_path: str) -> dict:
    '''
        Parse a text file containing key-value pairs and return a dictionary.

        Parameters:
        - file_path (str): The path to the text file.
        Returns:
        - dict: A dictionary where keys are strings and values are floats, lists of floats, or None.
    '''

    result = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                key, value_str = line.split('=')
                key = key.strip()
                value_str = value_str.strip()
                if value_str.startswith('['):
                    value = [float(val.strip()) for val in value_str.strip('[]').split(',')]
                elif value_str.lower() == 'none':
                    value = None
                else:
                    value = float(value_str)
                result[key] = value
    return result

def save_to_txt(filename: str, data_list: list) -> None:
    '''
        Save a list of data to a text file.

        Parameters:
        - filename (str): The name of the file to which the data will be saved.
        - data_list (list): The list of data to be saved.
        Returns:
        - None
    '''

    with open(filename, 'a',encoding='UTF-8') as file:
        for i,item in enumerate(data_list):
            file.write(item + '\n')
            if(i%2==1):
                file.write('\n')

def save_to_csv(rezultati: tuple, filename:str) -> None:
    '''
        Save results to a CSV file.

        Parameters:
        - rezultati (List[Union[float, None]]): A list containing results for 'koordinatno_pretrazivanje',
        'nelder_meade', and 'hooke_jeves'. Each element can be a float or None.
        - filename (str): The name of the CSV file.
        Returns:
        - None
    '''

    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['koordinatno_pretrazivanje', 'nelder_meade', 'hooke_jeves']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow({
            'koordinatno_pretrazivanje': rezultati[0],
            'nelder_meade': rezultati[1],
            'hooke_jeves': rezultati[2]
        })

def izvrsi_metode_i_dohvati_rezultate(x0, e, funkcija,i):
    global broj_poziva_f1,broj_poziva_f2,broj_poziva_f3,broj_poziva_f4
    if i == 0:  #da znam koliko puta se koja funkcija pozvala          
        result_koordinatno = nonlinear_optimization.koordinatno_pretrazivanje(x0, e, funkcija)
        kp_str1 = (f"Optimalno rešenje: {result_koordinatno} s pocetnom tockom: {x0}")
        kp_str2 = (f"Funkcija se metodom koordinatnog pretraživanja pozvala {broj_poziva_f1} puta")
        str_kp = f"{result_koordinatno};{broj_poziva_f1}"
        print(kp_str1)
        print(kp_str2)
        broj_poziva_f1=0
        print()
        result_simpleks_nelder_mead,simplex = nonlinear_optimization.simpleks_nelder_mead(x0 = x0,epsilon = e ,target_function = funkcija)
        snm_str1 = (f"Optimalno rešenje Nelder-Mead: {result_simpleks_nelder_mead}, simpleks je {simplex}")
        snm_str2 = (f"Funkcija se simpleksom po Nelderu i Meadu pozvala {broj_poziva_f1} puta")
        str_snm = f"{simplex};{broj_poziva_f1}"
        print(snm_str1)
        print(snm_str2)
        broj_poziva_f1=0
        print()
        Dx = [0.5] * len(x0)
        result_hooke_jeeves = nonlinear_optimization.hooke_jeeves(x0, Dx, e, funkcija)
        hj_str1 = (f"Optimalno rješenje Hooke-Jeeves je: {result_hooke_jeeves}")
        hj_str2 = (f"Funkcija se algoritmom Hooke-Jeeves pozvala {broj_poziva_f1} puta")
        hj_snm = f"{result_hooke_jeeves};{broj_poziva_f1}"
        print(hj_str1)
        print(hj_str2)
        broj_poziva_f1=0
    if i == 1:
        result_koordinatno = nonlinear_optimization.koordinatno_pretrazivanje(x0, e, funkcija)
        kp_str1 = (f"Optimalno rešenje: {result_koordinatno} s pocetnom tockom: {x0}")
        kp_str2 = (f"Funkcija se metodom koordinatnog pretraživanja pozvala {broj_poziva_f2} puta")
        str_kp = f"{result_koordinatno};{broj_poziva_f2}"
        print(kp_str1)
        print(kp_str2)
        broj_poziva_f2=0
        print()
        result_simpleks_nelder_mead,simplex = nonlinear_optimization.simpleks_nelder_mead(x0 = x0,epsilon = e ,target_function = funkcija)
        snm_str1 = (f"Optimalno rešenje Nelder-Mead: {result_simpleks_nelder_mead}, simpleks je {simplex}")
        snm_str2 = (f"Funkcija se simpleksom po Nelderu i Meadu pozvala {broj_poziva_f2} puta")
        str_snm = f"{simplex};{broj_poziva_f2}"
        print(snm_str1)
        print(snm_str2)
        broj_poziva_f2=0
        print()
        Dx = [0.5] * len(x0)
        result_hooke_jeeves = nonlinear_optimization.hooke_jeeves(x0, Dx, e, funkcija)
        hj_str1 = (f"Optimalno rješenje Hooke-Jeeves je: {result_hooke_jeeves}")
        hj_str2 = (f"Funkcija se algoritmom Hooke-Jeeves pozvala {broj_poziva_f2} puta")
        hj_snm = f"{result_hooke_jeeves};{broj_poziva_f2}"
        print(hj_str1)
        print(hj_str2)
        broj_poziva_f2=0
    if i == 2:
        result_koordinatno = nonlinear_optimization.koordinatno_pretrazivanje(x0, e, funkcija)
        kp_str1 = (f"Optimalno rešenje: {result_koordinatno} s pocetnom tockom: {x0}")
        kp_str2 = (f"Funkcija se metodom koordinatnog pretraživanja pozvala {broj_poziva_f3} puta")
        str_kp = f"{result_koordinatno};{broj_poziva_f3}"
        print(kp_str1)
        print(kp_str2)
        broj_poziva_f3=0
        print()
        result_simpleks_nelder_mead,simplex = nonlinear_optimization.simpleks_nelder_mead(x0 = x0,epsilon = e ,target_function = funkcija)
        snm_str1 = (f"Optimalno rešenje Nelder-Mead: {result_simpleks_nelder_mead}, simpleks je {simplex}")
        snm_str2 = (f"Funkcija se simpleksom po Nelderu i Meadu pozvala {broj_poziva_f3} puta")
        str_snm = f"{simplex};{broj_poziva_f3}"
        print(snm_str1)
        print(snm_str2)
        broj_poziva_f3=0
        print()
        Dx = [0.5] * len(x0)
        result_hooke_jeeves = nonlinear_optimization.hooke_jeeves(x0, Dx, e, funkcija)
        hj_str1 = (f"Optimalno rješenje Hooke-Jeeves je: {result_hooke_jeeves}")
        hj_str2 = (f"Funkcija se algoritmom Hooke-Jeeves pozvala {broj_poziva_f3} puta")
        hj_snm = f"{result_hooke_jeeves};{broj_poziva_f3}"
        print(hj_str1)
        print(hj_str2)
        broj_poziva_f3=0    
    if i == 3:
        result_koordinatno = nonlinear_optimization.koordinatno_pretrazivanje(x0, e, funkcija)
        kp_str1 = (f"Optimalno rešenje: {result_koordinatno} s pocetnom tockom: {x0}")
        kp_str2 = (f"Funkcija se metodom koordinatnog pretraživanja pozvala {broj_poziva_f4} puta")
        str_kp = f"{result_koordinatno};{broj_poziva_f4}"
        print(kp_str1)
        print(kp_str2)
        broj_poziva_f4=0
        print()
        result_simpleks_nelder_mead,simplex = nonlinear_optimization.simpleks_nelder_mead(x0 = x0,epsilon = e ,target_function = funkcija)
        snm_str1 = (f"Optimalno rešenje Nelder-Mead: {result_simpleks_nelder_mead}, simpleks je {simplex}")
        snm_str2 = (f"Funkcija se simpleksom po Nelderu i Meadu pozvala {broj_poziva_f4} puta")
        str_snm = f"{simplex};{broj_poziva_f4}"
        print(snm_str1)
        print(snm_str2)
        broj_poziva_f4=0
        print()
        Dx = [0.5] * len(x0)
        result_hooke_jeeves = nonlinear_optimization.hooke_jeeves(x0, Dx, e, funkcija)
        hj_str1 = (f"Optimalno rješenje Hooke-Jeeves je: {result_hooke_jeeves}")
        hj_str2 = (f"Funkcija se algoritmom Hooke-Jeeves pozvala {broj_poziva_f4} puta")
        hj_snm = f"{result_hooke_jeeves};{broj_poziva_f4}"
        print(hj_str1)
        print(hj_str2)
        broj_poziva_f4=0
    
    return str_kp,str_snm,hj_snm


def f_prvi_zad(x):
    global broj_poziva_prvi_zad
    broj_poziva_prvi_zad+=1
    if(isinstance(x,list)):
        return (x[0]-3)**2
    else:
        return (x-3)**2

#Rosenbrockova 'banana' funkcija
def f1(x):
    global broj_poziva_f1
    broj_poziva_f1 += 1

    return 100 * (x[1]-x[0]**2)**2 + (1-x[0])**2

def f2(x):
    global broj_poziva_f2
    broj_poziva_f2 += 1
    
    return (x[0]-4)**2 + 4*(x[1]-2)**2

def f3(x):
    global broj_poziva_f3
    broj_poziva_f3 += 1
    res=0
    for i,x_i in enumerate(x):
        res += (x_i-(i+1))**2
    return res


def f4(x):
    global broj_poziva_f4
    broj_poziva_f4 += 1

    return abs(x[0]**2-x[1]**2) + math.sqrt((x[0]**2+x[1]**2))
    
def schaffer_f6(x):
    global broj_poziva_schaffer_f6
    broj_poziva_schaffer_f6 += 1   
    sigma = 0
    for i in x:
        sigma += (i**2)
    brojnik = (math.sin(math.sqrt(sigma)))**2 - 0.5
    nazivnik = (1.0 + 0.001 * (sigma))**2

    return brojnik / nazivnik + 0.5


def prvi_zadatak():
    global broj_poziva_prvi_zad
    prvi_zadatak_input = 'input/1_zad_input.txt'
    parsed_data = parse_text_file(prvi_zadatak_input)
    values_list = list(parsed_data.values())
    e = values_list[0]
    x0 = [values_list[1]]

    a,b,_ = nonlinear_optimization.zlatni_rez(None,None,e,f_prvi_zad,x0[0])
    zl_str1 = (f"Rješenje zlatnog reza je x € [{a},{b}]")
    zl_str2 = f"Funkcija se metodom zlatnog reza pozvala {broj_poziva_prvi_zad} puta"
    print(zl_str1)
    print(zl_str2)
    broj_poziva_prvi_zad=0
    print()
    ##############################################################################
    result_koordinatno = nonlinear_optimization.koordinatno_pretrazivanje(x0, e, f_prvi_zad)
    kp_str1 = (f"Optimalno rešenje: {result_koordinatno}")
    kp_str2 = (f"Funkcija se metodom koordinatnog pretraživanja pozvala {broj_poziva_prvi_zad} puta")
    print(kp_str1)
    print(kp_str2)
    broj_poziva_prvi_zad=0
    print()
    ##############################################################################
    optimalno_rj,simplex = nonlinear_optimization.simpleks_nelder_mead(x0 = x0,epsilon = e ,target_function = f_prvi_zad)
    snm_str1 = (f"Optimalno rešenje Nelder-Mead: {optimalno_rj}")
    snm_str2 = (f"Funkcija se simpleksom po Nelderu i Meadu pozvala {broj_poziva_prvi_zad} puta")
    print(snm_str1)
    print(snm_str2)
    broj_poziva_prvi_zad=0
    print()
    ##############################################################################
    Dx = [0.5] * len(x0)
    result_hooke_jeeves = nonlinear_optimization.hooke_jeeves(x0, Dx, e, f_prvi_zad)
    hj_str1 = (f"Optimalno rješenje Hooke-Jeeves je: {result_hooke_jeeves}")
    hj_str2 = (f"Funkcija se algoritmom Hooke-Jeeves pozvala {broj_poziva_prvi_zad} puta")
    print(hj_str1)
    print(hj_str2)
    broj_poziva_prvi_zad=0
    print()

    zapisi = [zl_str1,zl_str2,kp_str1,kp_str2,snm_str1,snm_str2,hj_str1,hj_str2]
    path = "results/1_zad_rj.txt"
    save_to_txt(path,zapisi)

def drugi_zadatak():
    global broj_poziva_f1, broj_poziva_f2 , broj_poziva_f3, broj_poziva_f4 
    drugi_zadatak_input = 'input/2_zad_input.txt'
    parsed_data = parse_text_file(drugi_zadatak_input)
    values_list = list(parsed_data.values())
    e = values_list[0]
    x0_f1 = values_list[1]
    x0_f2 = values_list[2]
    x0_f3 = values_list[3]
    x0_f4 = values_list[4]

    poc_tocke = [x0_f1,x0_f2,x0_f3,x0_f4]
    funckije = [f1,f2,f3,f4]
    drugi_zadatak_output = 'results/2_zad_rj.csv'

    #1 za danu funkciju i njezinu pocetnu tocku pozovem sva 3 postupka
    i = 0
    for x0, funkcija in zip(poc_tocke, funckije):
        rezultati = izvrsi_metode_i_dohvati_rezultate(x0, e, funkcija,i)
        save_to_csv(rezultati,drugi_zadatak_output)
        i+=1
        
def treci_zadatak():
    global broj_poziva_f4
    treci_zadatak_input = 'input/3_zad_input.txt'
    parsed_data = parse_text_file(treci_zadatak_input)
    values_list = list(parsed_data.values())
    e = values_list[0]
    x0 = values_list[1]
    
    optimalno_rj = nonlinear_optimization.simpleks_nelder_mead(x0 = x0,epsilon = e ,target_function = f4)
    snm_str1 = (f"Optimalno rešenje Nelder-Mead: {optimalno_rj}")
    snm_str2 = (f"Funkcija se simpleksom po Nelderu i Meadu pozvala {broj_poziva_f4} puta")
    print(snm_str1)
    print(snm_str2)
    broj_poziva_f4=0
    print()
    ##############################################################################
    Dx = [0.5] * len(x0)
    result_hooke_jeeves = nonlinear_optimization.hooke_jeeves(x0, Dx, e, f4)
    hj_str1 = (f"Optimalno rješenje Hooke-Jeeves je: {result_hooke_jeeves}")
    hj_str2 = (f"Funkcija se algoritmom Hooke-Jeeves pozvala {broj_poziva_f4} puta")
    print(hj_str1)
    print(hj_str2)
    broj_poziva_f4=0
    
    print()
    zapisi = [snm_str1,snm_str2,hj_str1,hj_str2]
    path = "results/3_zad_rj.txt"
    save_to_txt(path,zapisi)

def cetvrti_zadatak():
    global broj_poziva_f1
    cetvrti_zadatak_input = 'input/4_zad_input.txt'
    parsed_data = parse_text_file(cetvrti_zadatak_input)
    values_list = list(parsed_data.values())
    e = values_list[0]
    x0 = values_list[1]
    steps = values_list[2]
    
    for step in steps:
        optimalno_rj,simpleks = nonlinear_optimization.simpleks_nelder_mead(x0 = x0,simpleks_step=step, epsilon = e ,target_function = f1)
        snm_str1 = (f"{step} koraka za generiranje početnog simpleksa dalo je optimalno rješenje Nelder-Mead: {optimalno_rj} te simpleks {simpleks}")
        snm_str2 = (f"Funkcija se simpleksom po Nelderu i Meadu pozvala {broj_poziva_f1} puta")
        print(snm_str1)
        print(snm_str2)
        broj_poziva_f1=0
        zapisi = [snm_str1,snm_str2,]
        path = "results/4_zad_rj.txt"
        save_to_txt(path,zapisi)

def peti_zadatak():
    global broj_poziva_schaffer_f6
    peti_zadatak_input = 'input/5_zad_input.txt'
    parsed_data = parse_text_file(peti_zadatak_input)
    values_list = list(parsed_data.values())
    e = values_list[0]

    list_x0 = [nonlinear_optimization.generator_x0() for _ in range(10)]
    Dx = [0.5] * len(list_x0[0])

    for x0 in list_x0:
        optimalno_rj,simplex_rj = nonlinear_optimization.simpleks_nelder_mead(x0 = x0,epsilon = e ,target_function = schaffer_f6)
        snm_str1 = (f"S početnom točkom x0: {x0}, optimalno rešenje Nelder-Mead: {simplex_rj}")
        snm_str2 = (f"Funkcija se simpleksom po Nelderu i Meadu pozvala {broj_poziva_schaffer_f6} puta")
        print(snm_str1)
        print(snm_str2)
        broj_poziva_schaffer_f6=0
        print()
        zapisi = [snm_str1,snm_str2]
        path = "results/5_zad_rj.txt"
        save_to_txt(path,zapisi)

def main():
    global broj_poziva_f1,broj_poziva_f2,broj_poziva_f3,broj_poziva_f4,broj_poziva_schaffer_f6

    #1. zadatak
    prvi_zadatak()

    #2. zadatak
    drugi_zadatak()

    #3. zadatak
    treci_zadatak()

    #4. zadatak
    cetvrti_zadatak()

    #5. zadatak
    peti_zadatak()

if __name__ == "__main__":
    main()

