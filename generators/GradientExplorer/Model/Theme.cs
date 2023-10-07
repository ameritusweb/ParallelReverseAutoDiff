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

        public bool IsDark
        {
            get
            {
                return Name == "Dark";
            }
        }
    }
}
