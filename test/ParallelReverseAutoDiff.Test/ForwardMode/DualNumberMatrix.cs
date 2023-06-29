using ParallelReverseAutoDiff.RMAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.ForwardMode
{
    public class DualNumberMatrix
    {
        private DualNumber[][] matrix;

        public DualNumberMatrix(int numRows, int numCols)
        {
            matrix = new DualNumber[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                matrix[i] = new DualNumber[numCols];
            }
        }

        public DualNumber this[int row, int col]
        {
            get { return matrix[row][col]; }
            set { matrix[row][col] = value; }
        }

        public int Length
        {
            get { return matrix.Length; }
        }

        public DualNumber[] this[int row]
        {
            get { return matrix[row]; }
        }

        public Matrix ToMatrix()
        {
            int numRows = matrix.Length;
            int numCols = matrix[0].Length;
            Matrix m = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
                DualNumber[] row = matrix[i];
                for (int j = 0; j < numCols; j++)
                {
                    m[i, j] = row[j].Real;
                }
            }
            return m;
        }
    }
}
