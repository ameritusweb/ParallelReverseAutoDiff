using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;

namespace GradientExplorer.Extensions
{
    public static class SolidColorBrushExtensions
    {
        public static string ToHex(this SolidColorBrush brush)
        {
            Color color = brush.Color;
            return $"#{color.R:X2}{color.G:X2}{color.B:X2}";
        }
    }
}
