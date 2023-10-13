using System;
using System.Windows.Markup;
using System.Windows.Media.Animation;

namespace GradientExplorer.Helpers
{
    public class EasingFunctionExtension : MarkupExtension
    {
        public string FunctionName { get; set; }

        public EasingFunctionExtension(string functionName)
        {
            FunctionName = functionName;
        }

        public override object ProvideValue(IServiceProvider serviceProvider)
        {
            switch (FunctionName)
            {
                case "CubicEase":
                    return new CubicEase { EasingMode = EasingMode.EaseInOut };
                case "BounceEase":
                    return new BounceEase { Bounces = 3, Bounciness = 4 };
                case "ElasticEase":
                    return new ElasticEase { Oscillations = 3, Springiness = 4 };
                // Add more easing functions as needed
                default:
                    return new CubicEase { EasingMode = EasingMode.EaseInOut };
            }
        }
    }
}
