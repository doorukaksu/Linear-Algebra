import copy

class Matrix:
    def __init__(self,data):
        if not self._validate_data(data):
            raise ValueError('Invalid matrix data. Matrix must be 2D list with equal length rows')
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])
        
    def _validate_data(self,data):
        if not all(isinstance(row, list) for row in data):
            return False
            
        return all(len(row) == len(data[0]) for row in data)
    
    def __repr__(self):
        """ String representation of the matrix for printing. """
        return '\n'.join([' '.join(map(str, row)) for row in self.data])
    
    def __add__(self, other):

        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError('Matrices must have same dimensions to be added')

        result = [
        [self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        
        return Matrix(result)


    def __subtract__(self,other):

        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError('Matrices must have same dimensions to be added')

        result = [
        [self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        
        return Matrix(result)

    def __multiply__(self,other):
        
        if self.cols != other.rows:
            raise ValueError('Matrix A columns must be equal to matrix B rows.')
        
        result = [[0 for j in range(other.cols)] for i in range(self.rows)]
        
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        
        return Matrix(result)

    def __scalar_multiply__(self,multiplier):

        result =  result = [[0 for j in range(self.cols)] for i in range(self.rows)]

        for i in range(self.rows):
            for j in range(self.cols):
                result[i][j] += self.data[i][j] * multiplier
    
        return result


    
    
