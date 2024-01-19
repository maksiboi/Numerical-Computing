epsilon = 1e-9

class Matrica:

    def __init__(self, redaka=0, stupaca=0):
        self.redaka = redaka
        self.stupaca = stupaca
        self.matrica = [[0.0 for _ in range(stupaca)] for _ in range(redaka)]
    
    def set_matricu(self, elements):
        self.matrica = elements
        
    def __str__(self):
        return "\n".join([" ".join(map(str, red)) for red in self.matrica])

    def __getitem__(self, indeks:int):
        return self.matrica[indeks]

    def __setitem__(self, indeks:int, vrijednost):
        self.matrica[indeks] = vrijednost

    def __add__(self, druga_matrica: 'Matrica') -> 'Matrica':
        try:
            if self.redaka != druga_matrica.redaka or self.stupaca != druga_matrica.stupaca:
                raise ValueError("Matrice moraju imati iste dimenzije za zbrajanje.")
            rezultat = Matrica(self.redaka, self.stupaca)
            for i in range(self.redaka):
                for j in range(self.stupaca):
                    rezultat[i][j] = self[i][j] + druga_matrica[i][j]
            return rezultat
        except ValueError as e:
            print("Upozorenje:", e)
            return None

    def __sub__(self, druga_matrica: 'Matrica') -> 'Matrica':
        try:
            if self.redaka != druga_matrica.redaka or self.stupaca != druga_matrica.stupaca:
                raise ValueError("Matrice moraju imati iste dimenzije za oduzimanje.")
            rezultat = Matrica(self.redaka, self.stupaca)
            for i in range(self.redaka):
                for j in range(self.stupaca):
                    rezultat[i][j] = self[i][j] - druga_matrica[i][j]
            return rezultat
        except ValueError as e:
            print("Upozorenje:", e)
            return None

    def __mul__(self, druga_matrica: 'Matrica') -> 'Matrica':
        try:
            if isinstance(druga_matrica, (int, float)):
                rezultat = Matrica(self.redaka, self.stupaca)
                for i in range(self.redaka):
                    for j in range(self.stupaca):
                        rezultat[i][j] = self[i][j] * druga_matrica
                return rezultat
            elif isinstance(druga_matrica, Matrica):
                if self.stupaca != druga_matrica.redaka:
                    raise ValueError("Broj stupaca prve matrice mora biti jednak broju redaka druge matrice za množenje.")
                    
                rezultat = Matrica(self.redaka, druga_matrica.stupaca)
                for i in range(self.redaka):
                    for j in range(druga_matrica.stupaca):
                        for k in range(self.stupaca):
                            rezultat[i][j] += self[i][k] * druga_matrica[k][j]
                return rezultat
            else:
                raise ValueError("Nedopušteni operand za množenje.")
        except ValueError as e:
            print("Upozorenje:", e)
            return None  

    def __rmul__(self, scalar: float):
        '''
            Handles the reverse operation for scalar * matrix multiplication.
        '''
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> 'Matrica':
        try:
            if isinstance(scalar, (int, float)):
                result = Matrica(self.redaka, self.stupaca)
                for i in range(self.redaka):
                    for j in range(self.stupaca):
                        result[i][j] = self[i][j] / scalar
                return result
            else:
                raise TypeError("Nemoguće podijeliti matricu s nenumeričkim skalarnim vrijednostima.")
        except TypeError as e:
            print(e)

    def __rtruediv__(self, scalar:float) -> 'Matrica':
        try:
            if isinstance(scalar, (int, float)):
                result = Matrica(self.redaka, self.stupaca)
                for i in range(self.redaka):
                    for j in range(self.stupaca):
                        result[i][j] = scalar / self[i][j]
                return result
            else:
                raise TypeError("Nemoguće podijeliti nenumeričke skalarnim vrijednostima s matricom.")
        except TypeError as e:
            print(e)

    def __iadd__(self, druga_matrica: 'Matrica') -> 'Matrica':
        try:
            if self.redaka != druga_matrica.redaka or self.stupaca != druga_matrica.stupaca:
                raise ValueError("Matrice moraju imati iste dimenzije za += operaciju.")
            for i in range(self.redaka):
                for j in range(self.stupaca):
                    self[i][j] += druga_matrica[i][j]
            return self
        except ValueError as e:
            print("Upozorenje:", e)
            return None  # Nastavite s izvođenjem programa

    def __isub__(self, druga_matrica: 'Matrica') -> 'Matrica':
        try:
            if self.redaka != druga_matrica.redaka or self.stupaca != druga_matrica.stupaca:
                raise ValueError("Matrice moraju imati iste dimenzije za -= operaciju.")
            for i in range(self.redaka):
                for j in range(self.stupaca):
                    self[i][j] -= druga_matrica[i][j]
            return self
        except ValueError as e:
            print("Upozorenje:", e)
            return None  # Nastavite s izvođenjem programa

    def __imul__(self, skalar:float) -> 'Matrica': 
        for i in range(self.redaka):
            for j in range(self.stupaca):
                self[i][j] *= skalar
        return self

    def __eq__(self, druga_matrica: 'Matrica') -> bool:
        if(self.redaka != 0 and druga_matrica == None):
            return False
        if self.redaka != druga_matrica.redaka or self.stupaca != druga_matrica.stupaca:
            return False
        for i in range(self.redaka):
            for j in range(self.stupaca):
                if self[i][j] != druga_matrica[i][j]:
                    return False
        return True
    
    def __round__(self, ndigits=0):
        rounded_matrica = Matrica(self.redaka, self.stupaca)
        for i in range(self.redaka):
            for j in range(self.stupaca):
                rounded_matrica.matrica[i][j] = round(self.matrica[i][j], ndigits)
        return rounded_matrica

    def citaj_iz_datoteke(self, ime_datoteke: str) -> None:
        '''
            Reads data from a file and updates the matrix attributes.

            Args:
            - ime_datoteke (str): The name of the file to read data from.
            Returns:
            - None: This method modifies the object's state in-place.
        '''
        with open(ime_datoteke, 'r') as datoteka:
            redovi = datoteka.readlines()
            self.redaka = len(redovi)
            self.stupaca = len(redovi[0].split())
            self.matrica = [[float(x) for x in red.split()] for red in redovi]

    def promijeni_dimenzije(self, redaka: int, stupaca: int) -> None:
        '''
            Changes the dimensions of the matrix.

            Args:
            - redaka (int): Number of rows for the new matrix.
            - stupaca (int): Number of columns for the new matrix.
            Returns:
            - None: This method modifies the object's state in-place.
        '''
        self.redaka = redaka
        self.stupaca = stupaca
        self.matrica = [[0.0 for _ in range(stupaca)] for _ in range(redaka)]

    def ispis_na_ekran(self) -> None:
        '''
            Prints the matrix to the console.

            Returns:
                - None
        '''
        for red in self.matrica:
            print(" ".join(map(str, red)))

    def ispis_u_datoteku(self, ime_datoteke: str) -> None:
        '''
            Writes the matrix to a file.

            Args:
            - ime_datoteke (str): The name of the file to write data to.

            Returns:
            - None

            Example:
            If the matrix is:
            [[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]]
            The content of the file would be:
            1.0 2.0 3.0
            4.0 5.0 6.0
        '''
        with open(ime_datoteke, 'w') as datoteka:
            for red in self.matrica:
                datoteka.write(" ".join(map(str, red)) + "\n")

    def u_listu(self) -> list:
        '''
            Converts the matrix to a flat list.

            Returns:
            - list: A flattened list containing all elements of the matrix.
        '''
        lista = []
        for red in self.matrica:
            for element in red:
                lista.append(element)
        return lista

    def T(self) -> 'Matrica':
        '''
            Returns the transpose of the matrix.

            Returns:
            - Matrica: A new instance representing the transposed matrix.
        '''
        transponirana = Matrica(self.stupaca, self.redaka)
        for i in range(self.redaka):
            for j in range(self.stupaca):
                transponirana[j][i] = self[i][j]
        return transponirana

    def supstitucija_unaprijed(self, b: 'Matrica') -> 'Matrica':
        '''
            Solves a system of linear equations using forward substitution.

            Args:
            - b (Matrica): The column vector on the right-hand side of the system.

            Returns:
            - Matrica: A column vector representing the solution of the system.

            Explanation:
            This method solves a system of linear equations using forward substitution,
            assuming the matrix is lower triangular. It checks for necessary conditions,
            such as a square matrix and matching dimensions with the vector 'b'. Then,
            it iterates through each row, calculating the corresponding element of the
            solution vector 'y' using the previously computed elements.
        '''
        try:
            if self.redaka != self.stupaca:
                raise ValueError("Matrica nije kvadratna.")
            if self.stupaca != b.redaka:
                raise ValueError("Dimenzije matrice i vektora ne odgovaraju.")
            
            n = self.redaka
            y = Matrica(n, 1)

            for i in range(n):
                if abs(self[i][i]) < epsilon:
                    raise ValueError("Nula ili jako mala vrijednost u stožernom elementu. Nema rješenja.")

                y[i][0] = b[i][0]
                for j in range(i):
                    y[i][0] -= self[i][j] * y[j][0]

            return y
        
        except ValueError as e:
            print("Upozorenje:", e)
            return None  

    def supstitucija_unatrag(self, b: 'Matrica') -> 'Matrica':
        '''
            Solves a system of linear equations using backward substitution.

            Args:
            - b (Matrica): The column vector on the right-hand side of the system.

            Returns:
            - Matrica: A column vector representing the solution of the system.

            Explanation:
            This method solves a system of linear equations using backward substitution,
            assuming the matrix is upper triangular. It checks for necessary conditions,
            such as a square matrix and matching dimensions with the vector 'b'. Then,
            it iterates through each row in reverse order, calculating the corresponding
            element of the solution vector 'x' using the previously computed elements.
        '''
        try:
            if self.redaka != self.stupaca:
                raise ValueError("Matrica nije kvadratna.")
            if self.stupaca != b.redaka:
                raise ValueError("Dimenzije matrice i vektora ne odgovaraju.")

            n = self.redaka
            x = Matrica(n, 1)

            for i in range(n - 1, -1, -1):
                if abs(self[i][i]) < epsilon:
                    raise ValueError("Stožerni element je jako mali ili nula, rješenje ne postoji.")

                x[i][0] = b[i][0] / self[i][i]  # Calculate the element in the last row of matrix A directly
                for j in range(i + 1, n):
                    x[i][0] -= self[i][j] * x[j][0] / self[i][i]

            return x
        
        except ValueError as e:
            print("Upozorenje:", e)
            return None  

    def LU(self) -> ('Matrica', 'Matrica'):
        '''
            Performs LU decomposition on the matrix.

            Returns:
            - Tuple(Matrica, Matrica): A tuple containing the U (upper triangular) and L (lower triangular) matrices.

            Explanation:
            This method performs LU decomposition on the matrix. It decomposes the matrix A into
            the product of lower triangular matrix L and upper triangular matrix U, such that A = LU.

            The LU decomposition is achieved through Gaussian elimination. It checks for necessary
            conditions, such as a square matrix, and performs the decomposition in-place.
        '''
        try:
            if self.redaka != self.stupaca:
                raise ValueError("Matrica mora biti kvadratna za LU dekompoziciju.")
            
            n = self.redaka
            
            for i in range(n - 1):
                if abs(self[i][i]) < epsilon:
                    raise ValueError("Stožerni element matrice je jako mali ili nula, dekompozicija nije moguća.")
                    
                for j in range(i + 1, n):
                    self[j][i] /= self[i][i]
                    for k in range(i + 1, n):
                        self[j][k] -= self[j][i] * self[i][k]
            
            L = Matrica(n, n)
            U = Matrica(n, n)
            
            for i in range(n):
                for j in range(n):
                    if i < j:
                        U[i][j] = self[i][j]
                    elif i == j:
                        U[i][j] = self[i][j]
                        L[i][j] = 1.0   # Set 1 on the diagonal for the L matrix
                    else:
                        L[i][j] = self[i][j]
            
            return U, L
        except ValueError as e:
            print("Upozorenje:", e)
            return None, None  

    def LUP(self) -> ('Matrica', 'Matrica', 'Matrica'):
        '''
            Performs LUP decomposition on the matrix.

            Returns:
            - Tuple(Matrica, Matrica, Matrica): A tuple containing the L (lower triangular),
            U (upper triangular), and P (permutation) matrices.

            Explanation:
            This method performs LUP decomposition on the matrix. It decomposes the matrix A into
            the product of lower triangular matrix L, upper triangular matrix U, and permutation
            matrix P, such that PA = LU.

            The LUP decomposition is achieved through Gaussian elimination with partial pivoting.
            It checks for necessary conditions, such as a square matrix, and performs the decomposition
            in-place.
        '''        
        try:
            if self.redaka != self.stupaca:
                raise ValueError("Matrica mora biti kvadratna za LUP dekompoziciju.")
            
            n = self.redaka
            P = list(range(n))  # List P represents the column order, i.e., column swaps

            for i in range(n):
                pivot = i
                for j in range(i + 1, n):
                    if abs(self[P[j]][i]) > abs(self[P[pivot]][i]):
                        pivot = j
                if abs(self[P[pivot]][i]) < epsilon:
                    raise ValueError("Stožerni element matrice je jako mali ili nula, dekompozicija nije moguća.")
                P[i], P[pivot] = P[pivot], P[i]

                for j in range(i + 1, n):
                    self[P[j]][i] /= self[P[i]][i]
                    for k in range(i + 1, n):
                        self[P[j]][k] -= self[P[j]][i] * self[P[i]][k]

            P_matrix = Matrica(n, n)  # Creating a matrix for permutation
            for i in range(n):
                P_matrix[i][P[i]] = 1.0  # Setting 1 at the position where the row is swapped

            L = Matrica(n, n)
            U = Matrica(n, n)

            for i in range(n):
                for j in range(n):
                    if i <= j:
                        U[i][j] = self[P[i]][j]
                    if i == j:
                        L[i][j] = 1.0  # Setting 1 on the diagonal for the L matrix
                    if i > j:
                        L[i][j] = self[P[i]][j]

            
            return U, L, P_matrix
        
        except ValueError as e:
            print("Upozorenje:", e)
            return None, None, None 

    def inverz(self) -> 'Matrica':
        '''
            Calculates the inverse of the matrix using LUP decomposition.

            Returns:
            - Matrica: A new instance representing the inverse matrix.

            Explanation:
            This method calculates the inverse of the matrix using LUP decomposition.
            It first checks if the matrix is square. Then, it performs LUP decomposition
            to obtain the matrices L, U, and P. After that, it solves the system of linear
            equations iteratively for each column vector of the identity matrix.
        '''
        try:
            n = self.redaka
            if self.stupaca != n:
                raise ValueError("Matrica mora biti kvadratna za računanje inverza.")
            
            U, L, P= self.LUP()  # Perform LUP decomposition

            if(U == None or L == None):
                return None

            inverz = Matrica(n, n)

            for i in range(n):
                e = Matrica(n, 1)
                e[i][0] = 1.0   # Set the vector e_i as the initial vector
                

                # Solve the system L * y = P * e_i using forward substitution
                y = L.supstitucija_unaprijed(P * e)

                # Solve the system L * y = P * e_i using forward substitution
                x = U.supstitucija_unatrag(y)


                # Add x as a column
                for j,elem in enumerate(x):
                    inverz[j][i] = elem[0]

            return inverz
        
        except ValueError as e:
            print("Upozorenje:", e)
            return None  

    def determinanta(self) -> float:
        '''
            Calculates the determinant of the matrix using LUP decomposition.

            Returns:
            - float: The determinant of the matrix.

            Explanation:
            This method calculates the determinant of the matrix using LUP decomposition.
            It first checks if the matrix is square. Then, it performs LUP decomposition to
            obtain the matrices L, U, and P. After that, it computes the determinants of P,
            L, and U and combines them to calculate the determinant of the original matrix A.
        '''
        try:
            n = self.redaka
            if self.stupaca != n:
                raise ValueError("Matrica mora biti kvadratna za računanje determinante.")

            U, L, P = self.LUP()   # Perform LUP decomposition

            if(U == None or L == None):
                return 0

            det_P = 1
            for i in range(n):
                if i != P[i]:
                    det_P *= -1  # Računamo determinantu permutacijske matrice

            det_L = 1
            det_U = 1

            for i in range(n):
                det_L *= L[i][i]  # Računamo umnožak elemenata na dijagonali matrice L
                det_U *= U[i][i]  # Računamo umnožak elemenata na dijagonali matrice U

            determinant = det_P * det_L * det_U  # Računamo ukupnu determinantu A

            return determinant * (-1)
        
        except ValueError as e:
            print("Upozorenje:", e)
            return None  # Nastavite s izvođenjem programa


      
