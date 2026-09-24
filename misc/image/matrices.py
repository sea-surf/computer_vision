import numpy as np # Matrices

def main():
    print("Matrix a:")
    matrix_a = np.arange(12)
    print(f'Shape: {matrix_a.shape}')
    print(f'Len: {len(matrix_a)}')
    print(matrix_a)

    print("Matrix b:")
    matrix_b = matrix_a.reshape((4,3))
    print(matrix_b)
    print(f'Shape: {matrix_b.shape}')

def matrix_2():
    m = np.arange(20).reshape((5,4))
    print(m)
    print(f'Elem[0][1]: {m[0][1]}')

    m = np.arange(20).reshape((2, 5, 2)) # 2 Matrices, 5 Rows x 2 Columns
    print(m)
    print(f'Elem[0][1]: {m[0, 1, 0]}') # [0] First Matrix, [1] First Row x [0] First Column

def matrix_3():
    c = np.arange(20).reshape((5,4))
    print(c)
    print(c[:2, :])    # 2 First Rows, : of all Columns
    print(c[0:2, 0:2]) # 2 First Rows, 2 First Columns

def matrix_color():
    m = np.arange(10 * 10 * 3).reshape((10, 10, 3))
    print(m[:5, :5, :])

    # Mask
    a = np.arange(20)
    print(a)
    print(a[a>5])
    a[a>5] = 0
    print(a)

if __name__ == "__main__":
    matrix_color()