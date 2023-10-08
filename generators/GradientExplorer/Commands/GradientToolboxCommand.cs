using GradientExplorer;

namespace ToolWindow
{
    [Command(PackageIds.MyToolboxCommand)]
    internal sealed class GradientToolboxCommand : BaseCommand<GradientToolboxCommand>
    {
        protected override Task ExecuteAsync(OleMenuCmdEventArgs e)
        {
            return GradientToolbox.ShowAsync();
        }
    }
}
