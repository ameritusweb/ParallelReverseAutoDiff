using System.Text;
using System.Windows;

namespace GradientExplorer.Helpers
{
    public class PropertyPathBuilder
    {
        private readonly StringBuilder stringBuilder = new StringBuilder();

        public PropertyPathBuilder WithDependencyProperty(DependencyProperty property)
        {
            AppendSegment($"({property.OwnerType.Name}.{property.Name})");
            return this;
        }

        public PropertyPathBuilder WithDependencyArray(DependencyProperty property, int index)
        {
            AppendSegment($"({property.OwnerType.Name}.{property.Name})[{index}]");
            return this;
        }

        private void AppendSegment(string segment)
        {
            if (stringBuilder.Length > 0)
            {
                stringBuilder.Append('.');
            }
            stringBuilder.Append(segment);
        }

        public PropertyPath Build()
        {
            if (stringBuilder.Length == 0)
            {
                throw new InvalidOperationException("No segments added to PropertyPathBuilder.");
            }

            return new PropertyPath(stringBuilder.ToString());
        }
    }
}
