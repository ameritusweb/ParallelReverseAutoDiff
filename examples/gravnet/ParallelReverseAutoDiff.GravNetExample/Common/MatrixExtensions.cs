namespace ParallelReverseAutoDiff.GravNetExample.Common
{
    using ParallelReverseAutoDiff.RMAD;

    public static class MatrixExtensions
    {
        public static Matrix GetSection(this Matrix matrix, int startRow, int startCol, int endRow, int endCol)
        {
            int sectionRows = endRow - startRow + 1;
            int sectionColumns = endCol - startCol + 1;
            Matrix section = new Matrix(sectionRows, sectionColumns);

            for (int i = startRow; i <= endRow; i++)
            {
                for (int j = startCol; j <= endCol; j++)
                {
                    section[i - startRow, j - startCol] = matrix[i, j];
                }
            }

            return section;
        }
    }

}
