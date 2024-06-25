using Xunit;

namespace ParallelReverseAutoDiff.Test.Common
{
    public class DoubleArrayEqualityComparer : IEqualityComparer<double[]>
    {
        private readonly double _precision;

        public DoubleArrayEqualityComparer(double precision)
        {
            if (precision >= 1d)
            {
                precision = 1d / Math.Pow(10d, precision);
            }

            _precision = precision;
        }

        public bool Equals(double[] x, double[] y)
        {
            if (ReferenceEquals(x, y)) return true;
            if (x == null || y == null) return false;
            if (x.Length != y.Length) return false;

            return x.Zip(y, (a, b) => Math.Abs(a - b) <= _precision).All(equal => equal);
        }

        public int GetHashCode(double[] obj)
        {
            throw new NotImplementedException();
        }
    }
}
