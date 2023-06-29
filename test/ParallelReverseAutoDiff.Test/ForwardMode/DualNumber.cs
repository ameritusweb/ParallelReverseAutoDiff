using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.ForwardMode
{
    public class DualNumber
    {
        public double Real { get; set; }
        public double Dual { get; set; }

        public DualNumber(double real, double dual)
        {
            Real = real;
            Dual = dual;
        }

        public static DualNumber operator +(DualNumber a, DualNumber b)
            => new DualNumber(a.Real + b.Real, a.Dual + b.Dual);

        public static DualNumber operator -(DualNumber a, DualNumber b)
            => new DualNumber(a.Real - b.Real, a.Dual - b.Dual);

        public static DualNumber operator *(DualNumber a, DualNumber b)
            => new DualNumber(a.Real * b.Real, a.Real * b.Dual + a.Dual * b.Real);

        public static DualNumber operator /(DualNumber a, DualNumber b)
            => new DualNumber(a.Real / b.Real, (a.Dual * b.Real - a.Real * b.Dual) / (b.Real * b.Real));

        public static DualNumber Sqrt(DualNumber a)
            => new DualNumber(Math.Sqrt(a.Real), 0.5 * a.Dual / Math.Sqrt(a.Real));

        public static DualNumber Pow(DualNumber a, double power)
            => new DualNumber(Math.Pow(a.Real, power), power * Math.Pow(a.Real, power - 1) * a.Dual);

        public static DualNumber operator /(DualNumber a, int b)
            => new DualNumber(a.Real / b, a.Dual / b);

        public static DualNumber Sum(IEnumerable<DualNumber> collection)
        {
            DualNumber sum = new DualNumber(0, 0);
            foreach (DualNumber num in collection)
            {
                sum += num;
            }
            return sum;
        }

    }
}
