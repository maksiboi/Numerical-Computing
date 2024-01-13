from matrica import Matrica

def ispis_string_u_dat(ime_datoteke, sadrzaj) -> None:
    with open(ime_datoteke, 'w', encoding="utf-8") as file:
        file.write(sadrzaj)

if __name__ == "__main__":
    # 1. zadatak
    print("1. ZADATAK")
    A1 = Matrica()
    A1_POM = Matrica()
    A1.citaj_iz_datoteke("ulazi/zad1_ulaz.txt")
    print("Matrica A:")
    print(A1)
    A1_POM = A1 * 7
    print()
    print("Matrica A nakon množenja i djeljenja")
    A1_POM = A1 / 7

    A1_POM.ispis_u_datoteku("rezultati/prvi_zad")

    print(A1_POM)
    if A1 == A1_POM:
        print("Matrice su jednake\n")
    else:
        print("Matrice nisu jednake\n")

    # 2. zadatak
    # Rješavanje sustava lin. jednadžbi
    print("2. ZADATAK")
    A2_lu, A2_lup = Matrica(), Matrica()
    A2_lu.citaj_iz_datoteke("ulazi/zad2_ulaz.txt")
    A2_lup.citaj_iz_datoteke("ulazi/zad2_ulaz.txt")
    b2 = Matrica()
    b2.citaj_iz_datoteke("vektor_b/zad2_b.txt")
    print("Rješavamo sustav LU dekompozicijom")
    U2_lu, L2_lu = A2_lu.LU()
    print(U2_lu)
    if not isinstance(U2_lu, Matrica):
        ispis_string_u_dat("rezultati/drugi_zad_lu", "Stožerni element matrice je jako mali ili nula, dekompozicija nije moguća.")
    print()

    print("Rješavamo sustav LUP dekompozicijom")
    U2_lup, L2_lup, P2_lup = A2_lup.LUP()

    y = L2_lup.supstitucija_unaprijed(P2_lup*b2)
    x = U2_lup.supstitucija_unatrag(y)

    if isinstance(x, Matrica):
        print("Rješenje sustava LUP dekompozicijom je: \n", x)
        x.ispis_u_datoteku("rezultati/drugi_zad_lup")

    print("\n")
    print("3. ZADATAK")
    # 3. zadatak
    A3_lu, A3_lup = Matrica(), Matrica()
    A3_lu.citaj_iz_datoteke("ulazi/zad3_ulaz.txt")
    A3_lup.citaj_iz_datoteke("ulazi/zad3_ulaz.txt")

    print("Rastavljanje matrice LU dekompozicijom")
    U3_lu, L3_lu = A3_lu.LU()
    if isinstance(U3_lu, Matrica) and isinstance(L3_lu, Matrica):
        print("Matrica se može rastaviti LU dekompozicijom")
        U3_lu.ispis_u_datoteku("rezultati/treci_zad_U_lu")
        L3_lu.ispis_u_datoteku("rezultati/treci_zad_L_lu")

    print()
    print("Rastavljanje matrice LUP dekompozicijom")
    U3_lup, L3_lup, P3_lup = A3_lup.LUP()
    if isinstance(U3_lup, Matrica) and isinstance(L3_lup, Matrica):
        print("Matrica se može rastaviti LU dekompozicijom")
        U3_lup.ispis_u_datoteku("rezultati/treci_zad_U_lup")
        L3_lup.ispis_u_datoteku("rezultati/treci_zad_L_lup")
    else:
        ispis_string_u_dat("rezultati/treci_zad_LUP_dekompozicija",
                           "Stožerni element matrice je jako mali ili nula, dekompozicija nije moguća.")

    print("\n")
    print("4. ZADATAK")
    # 4. zadatak
    A4_lu, A4_lup = Matrica(), Matrica()
    A4_lu.citaj_iz_datoteke("ulazi/zad4_ulaz.txt")
    A4_lup.citaj_iz_datoteke("ulazi/zad4_ulaz.txt")
    b4 = Matrica()
    b4.citaj_iz_datoteke("vektor_b/zad4_b.txt")
    print("Rješavamo sustav LU dekompozicijom")
    U4_lu, L4_lu = A4_lu.LU()
    y_4_lu = L4_lu.supstitucija_unaprijed(b4)
    x_4_lu = U4_lu.supstitucija_unatrag(y_4_lu)

    if isinstance(x_4_lu, Matrica):
        print("Rješenje sustava LU dekompozicijom je: \n", x_4_lu)
        x_4_lu.ispis_u_datoteku("rezultati/cetvrti_zad_lu")
    print()

    print("Rješavamo sustav LUP dekompozicijom")
    U4_lup, L4_lup, P4_lup = A4_lup.LUP()
    y_4_lup = L4_lup.supstitucija_unaprijed(P4_lup * b4)
    x_4_lup = U4_lup.supstitucija_unatrag(y_4_lup)
    if isinstance(x_4_lup, Matrica):
        print("Rješenje sustava LUP dekompozicijom je: \n", x_4_lup)
        x_4_lup.ispis_u_datoteku("rezultati/cetvrti_zad_lup")

    print("\n")
    print("5. ZADATAK")
    # 5. zadatak
    A5 = Matrica()
    A5.citaj_iz_datoteke("ulazi/zad5_ulaz.txt")
    b5 = Matrica()
    b5.citaj_iz_datoteke("vektor_b/zad5_b.txt")

    print("Rješavamo sustav LUP dekompozicijom")
    U5_lup, L5_lup, P5_lup = A5.LUP()

    y_5_lup = L5_lup.supstitucija_unaprijed(P5_lup * b5)

    x_5_lup = U5_lup.supstitucija_unatrag(y_5_lup)

    if isinstance(x_5_lup, Matrica):
        print("Rješenje sustava LUP dekompozicijom je: \n", x_5_lup)
        x_5_lup.ispis_u_datoteku("rezultati/peti_zad_lup")

    print("\n")
    print("6. ZADATAK")
    # 6. zadatak
    A6 = Matrica()
    A6.citaj_iz_datoteke("ulazi/zad6_ulaz.txt")
    b6 = Matrica()
    b6.citaj_iz_datoteke("vektor_b/zad6_b.txt")

    print("Rješavamo sustav LUP dekompozicijom")
    U6_lup, L6_lup, P6_lup = A6.LUP()

    y_6_lup = L6_lup.supstitucija_unaprijed(P6_lup * b6)

    x_6_lup = U6_lup.supstitucija_unatrag(y_6_lup)

    if isinstance(x_6_lup, Matrica):
        print("Rješenje sustava LUP dekompozicijom je: \n", x_6_lup)
        x_6_lup.ispis_u_datoteku("rezultati/sesti_zad_lup")

    print("\n")
    print("7. ZADATAK")
    # 7. zadatak
    A7 = Matrica()
    A7.citaj_iz_datoteke("ulazi/zad7_ulaz.txt")

    inverz_A7 = A7.inverz()
    if inverz_A7 is None:
        print("Matrica A nema inverz")
        ispis_string_u_dat("rezultati/sedmi_zad_inverz", "Matrica A nema inverz")
    else:
        print()
        print("Inverz matrice A je")
        print(inverz_A7)
        inverz_A7.ispis_u_datoteku("rezultati/sedmi_zad_inverz")

    print("\n")
    print("8. ZADATAK")
    # 8. zadatak
    A8 = Matrica()
    A8.citaj_iz_datoteke("ulazi/zad8_ulaz.txt")

    print(A8)
    inverz_A8 = A8.inverz()
    if inverz_A8 is None:
        print("Determinanta je 0, stoga matrica A nema inverz")
        ispis_string_u_dat("rezultati/osmi_zad_inverz", "Matrica A nema inverz")
    else:
        print()
        print("Inverz matrice A je")
        print(inverz_A8)
        inverz_A8.ispis_u_datoteku("rezultati/osmi_zad_inverz")

    print("\n")
    print("9. ZADATAK")
    # 9. zadatak
    A9 = Matrica()
    A9.citaj_iz_datoteke("ulazi/zad9_ulaz.txt")

    det_A9 = A9.determinanta()
    print("Determinanta matrice je", det_A9)
    ispis_string_u_dat("rezultati/deveti_zad_determinanta", f"Determinanta matrice je {det_A9}")

    print("\n")
    print("10. ZADATAK")
    # 10. zadatak
    A10 = Matrica()
    A10.citaj_iz_datoteke("ulazi/zad10_ulaz.txt")

    det_A10 = A10.determinanta()
    print("Determinanta matrice je", det_A10)
    ispis_string_u_dat("rezultati/deseti_zad_determinanta", f"Determinanta matrice je {det_A10}")
