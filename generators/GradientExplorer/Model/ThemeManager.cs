using Microsoft.VisualStudio.PlatformUI;
using Microsoft.VisualStudio.Settings;
using Microsoft.VisualStudio.Shell.Interop;
using Microsoft.VisualStudio.Shell.Settings;
using Microsoft.VisualStudio.Threading;
using System.Threading.Tasks;

namespace GradientExplorer.Model
{
    public class ThemeManager
    {
        private static readonly Lazy<ThemeManager> _instance = new Lazy<ThemeManager>(() => new ThemeManager());

        public static ThemeManager Instance => _instance.Value;

        private readonly AsyncLazy<IVsSettingsManager> _settingsManager;

        public Theme[] Themes { get; } = {
            new Theme { Name = "Dark", Guid = new Guid("{1DED0138-47CE-435E-84EF-9EC1F439B749}") },
            new Theme { Name = "Light", Guid = new Guid("{DE3DBBCD-F642-433C-8353-8F1DF4370ABA}") },
            new Theme { Name = "Blue", Guid = new Guid("{A4D6A176-B948-4B29-8C66-53C97A1ED7D0}") },
            new Theme { Name = "Blue (Extra Contrast)", Guid = new Guid("{CE94D289-8481-498B-8CA9-9B6191A315B9}") }
        };

        private ThemeManager()
        {
            _settingsManager = new AsyncLazy<IVsSettingsManager>(async () =>
            {
                await ThreadHelper.JoinableTaskFactory.SwitchToMainThreadAsync();
                return (IVsSettingsManager)await ServiceProvider.GetGlobalServiceAsync(typeof(SVsSettingsManager));
            });
        }
        public async Task<Theme?> GetCurrentThemeAsync()
        {
            const string COLLECTION_NAME = @"ApplicationPrivateSettings\Microsoft\VisualStudio";
            const string PROPERTY_NAME = "ColorTheme";

            IVsSettingsManager manager = await _settingsManager.GetValueAsync();
            SettingsStore store = new ShellSettingsManager(manager).GetReadOnlySettingsStore(SettingsScope.UserSettings);

            if (store.CollectionExists(COLLECTION_NAME))
            {
                if (store.PropertyExists(COLLECTION_NAME, PROPERTY_NAME))
                {
                    string[] parts = store.GetString(COLLECTION_NAME, PROPERTY_NAME).Split('*');
                    if (parts.Length == 3)
                    {
                        Guid themeGuid;
                        if (Guid.TryParse(parts[2], out themeGuid))
                        {
                            Theme theme = Array.Find(Themes, t => t.Guid == themeGuid);
                            return theme ?? null;
                        }
                    }
                }
            }
            return null;
        }
    }
}
