using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Model
{
    public class Theme
    {
        public string Name { get; set; }
        public Guid Guid { get; set; }
        public System.Drawing.Color BackgroundColor { get; set; }

        public Theme DeepClone()
        {
            Theme copy = new Theme();
            copy.Name = Name;
            copy.Guid = Guid;
            copy.BackgroundColor = BackgroundColor;
            return copy;
        }

        public Microsoft.Msagl.Drawing.Color MsaglBackgroundColor
        {
            get
            {
                return new Microsoft.Msagl.Drawing.Color(BackgroundColor.A, BackgroundColor.R, BackgroundColor.G, BackgroundColor.B);
            }
        }

        public bool IsDark 
        { 
            get
            {
                if (Name == "Unknown")
                {
                    // Convert color to RGB components normalized to [0,1]
                    double r = BackgroundColor.R / 255.0;
                    double g = BackgroundColor.G / 255.0;
                    double b = BackgroundColor.B / 255.0;

                    // Calculate the luminance
                    double luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    if (luminance <= 0.5)
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    return Name.Contains("Dark");
                }
            }
        }
    }
}
