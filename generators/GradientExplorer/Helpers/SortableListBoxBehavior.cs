using System.Windows.Controls;
using System.Windows.Media.Animation;
using System.Windows.Media;
using System.Windows;
using GradientExplorer.Model;
using System.Collections.ObjectModel;
using System.Windows.Input;

namespace GradientExplorer.Helpers
{
    public static class SortableListBoxBehavior
    {
        private static SortableItem ghostItem = new SortableItem { Name = "Ghost", IsGhost = true };

        public static readonly DependencyProperty AllowSortProperty =
            DependencyProperty.RegisterAttached("AllowSort", typeof(bool), typeof(SortableListBoxBehavior),
                new PropertyMetadata(false, OnAllowSortChanged));

        public static bool GetAllowSort(DependencyObject d)
        {
            return (bool)d.GetValue(AllowSortProperty);
        }

        public static void SetAllowSort(DependencyObject d, bool value)
        {
            d.SetValue(AllowSortProperty, value);
        }

        public static readonly DependencyProperty AnimationDurationProperty =
            DependencyProperty.RegisterAttached("AnimationDuration", typeof(double), typeof(SortableListBoxBehavior),
                new PropertyMetadata(0.3));

        public static double GetAnimationDuration(DependencyObject d)
        {
            return (double)d.GetValue(AnimationDurationProperty);
        }

        public static void SetAnimationDuration(DependencyObject d, double value)
        {
            d.SetValue(AnimationDurationProperty, value);
        }

        public static readonly DependencyProperty EasingFunctionProperty =
    DependencyProperty.RegisterAttached("EasingFunction", typeof(EasingFunctionBase), typeof(SortableListBoxBehavior),
        new PropertyMetadata(new CubicEase { EasingMode = EasingMode.EaseInOut }));

        public static EasingFunctionBase GetEasingFunction(DependencyObject d)
        {
            return (EasingFunctionBase)d.GetValue(EasingFunctionProperty);
        }

        public static void SetEasingFunction(DependencyObject d, EasingFunctionBase value)
        {
            d.SetValue(EasingFunctionProperty, value);
        }


        private static void OnAllowSortChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is ListBox listBox)
            {
                if ((bool)e.NewValue)
                {
                    listBox.PreviewMouseLeftButtonDown += ListBox_PreviewMouseLeftButtonDown;
                    listBox.DragOver += ListBox_DragOver;
                    listBox.Drop += ListBox_Drop;
                }
                else
                {
                    listBox.PreviewMouseLeftButtonDown -= ListBox_PreviewMouseLeftButtonDown;
                    listBox.DragOver -= ListBox_DragOver;
                    listBox.Drop -= ListBox_Drop;
                }
            }
        }

        private static void ListBox_PreviewMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            // Sender is ListBox, so we need to find the ListBoxItem that is actually under the mouse
            if (sender is ListBox listBox)
            {
                var hitTestResult = VisualTreeHelper.HitTest(listBox, e.GetPosition(listBox));
                var listBoxItem = FindVisualParent<ListBoxItem>(hitTestResult.VisualHit);

                if (listBoxItem != null)
                {
                    ListBoxItem draggedItem = listBoxItem;
                    draggedItem.IsSelected = true;
                    DragDrop.DoDragDrop(draggedItem, draggedItem.DataContext, DragDropEffects.Move);
                }
            }
        }

        private static T FindVisualParent<T>(DependencyObject child) where T : DependencyObject
        {
            while (child != null && !(child is T))
            {
                child = VisualTreeHelper.GetParent(child);
            }
            return child as T;
        }

        private static void ListBox_DragOver(object sender, DragEventArgs e)
        {
            e.Handled = true;
            ListBox listBox = sender as ListBox;
            var itemsSource = listBox.ItemsSource as ObservableCollection<ISortableItem>;  // Cast to your specific item type

            if (itemsSource == null)
            {
                throw new InvalidOperationException("ItemsSource is not an ObservableCollection<Item>");
            }

            Point position = e.GetPosition(listBox);
            int? index = GetCurrentIndex(listBox, position);
            int selectedIndex = listBox.SelectedIndex;  // Get the index of the selected (dragged) item

            if (index.HasValue)
            {
                var ghostItemIndex = itemsSource.IndexOf(ghostItem);

                // Decide where to place the ghost item based on the direction of the drag
                if (index < selectedIndex)
                {
                    // Dragging upwards
                    if (index != ghostItemIndex)
                    {
                        UpdateGhostItemPosition(itemsSource, index.Value);
                    }
                }
                else if (index > selectedIndex)
                {
                    // Dragging downwards
                    index++;  // Adjust to place the ghost item below the current item
                    if (index != ghostItemIndex)
                    {
                        UpdateGhostItemPosition(itemsSource, index.Value);
                    }
                }
            }
        }

        private static void UpdateGhostItemPosition(ObservableCollection<ISortableItem> itemsSource, int index)
        {
            var ghostItemIndex = itemsSource.IndexOf(ghostItem);
            if (!itemsSource.Contains(ghostItem))
            {
                itemsSource.Insert(index, ghostItem);
            }
            else
            {
                itemsSource.Move(ghostItemIndex, index);
            }
        }

        private static int? GetCurrentIndex(ListBox listBox, Point position, double tolerance = 5.0)
        {
            for (int i = 0; i < listBox.Items.Count; i++)
            {
                ListBoxItem item = (ListBoxItem)listBox.ItemContainerGenerator.ContainerFromIndex(i);
                ISortableItem sortableItem = item.DataContext as ISortableItem;
                if (!item.IsSelected && !sortableItem.IsGhost)
                {
                    if (IsMouseOverTarget(listBox, item, position, tolerance))
                    {
                        return i;
                    }
                }
            }
            return default(int?);
        }

        private static bool IsMouseOverTarget(ListBox listBox, Visual target, Point point, double tolerance)
        {
            Rect bounds = VisualTreeHelper.GetDescendantBounds(target);

            // Transform the bounds to the coordinate space of the parent ListBox
            GeneralTransform transform = target.TransformToAncestor(listBox);
            bounds = transform.TransformBounds(bounds);
            bounds = new Rect(bounds.Left + tolerance, bounds.Top + tolerance,
                              bounds.Width - 2 * tolerance, bounds.Height - 2 * tolerance);
            var withinBounds = bounds.Contains(point);
            if (withinBounds)
            {
                
            }
            return withinBounds;
        }

        private static void ListBox_Drop(object sender, DragEventArgs e)
        {

            ListBox listBox = sender as ListBox;
            var itemsSource = listBox.ItemsSource as ObservableCollection<ISortableItem>;  // Cast to your specific item type

            if (itemsSource == null)
            {
                throw new InvalidOperationException("ItemsSource is not an ObservableCollection<Item>");
            }

            ISortableItem droppedData = e.Data.GetData(typeof(SortableItem)) as SortableItem;
            int removedIdx = itemsSource.IndexOf(droppedData);
            int targetIdx = itemsSource.IndexOf(ghostItem);

            ListBoxItem removedItem = (ListBoxItem)listBox.ItemContainerGenerator.ContainerFromIndex(removedIdx);
            ListBoxItem targetItem = (ListBoxItem)listBox.ItemContainerGenerator.ContainerFromIndex(targetIdx);

            removedItem.IsSelected = false;

            double animationDistance = CalculateAnimationDistance(listBox, removedItem, targetItem);

            // Animate and then perform data manipulation
            AnimateItemMove(sender as DependencyObject, removedItem, animationDistance, () =>
            {
                // Replace the ghostItem with the actual dragged item
                itemsSource[targetIdx] = droppedData;  // This replaces the ghost item
                                                       // Remove the original instance of the dragged item
                itemsSource.RemoveAt(removedIdx);
            });

            e.Handled = true;
        }

        private static double CalculateAnimationDistance(ListBox parent, ListBoxItem from, ListBoxItem to)
        {
            // Calculate distance between the Y positions of the two items
            // Logic can be customized based on layout and orientation
            return to.TransformToAncestor(parent).Transform(new Point(0, 0)).Y - from.TransformToAncestor(parent).Transform(new Point(0, 0)).Y;
        }

        private static void AnimateItemMove(DependencyObject d, ListBoxItem item, double to, Action onAnimationCompleted)
        {
            double durationInSeconds = GetAnimationDuration(d);

            TranslateTransform translateTransform = new TranslateTransform();
            item.RenderTransform = translateTransform;

            DoubleAnimation animation = new DoubleAnimation
            {
                To = to,
                Duration = new Duration(TimeSpan.FromSeconds(durationInSeconds)),
                EasingFunction = GetEasingFunction(d) // 1. Easing Functions
            };

            Storyboard.SetTarget(animation, translateTransform);
            Storyboard.SetTargetProperty(animation, new PropertyPath("Y"));

            Storyboard storyboard = new Storyboard();
            storyboard.Children.Add(animation);

            storyboard.Completed += (s, e) =>
            {
                // Reset transformations after animation
                item.RenderTransform = null;

                // Execute additional logic after animation is complete
                onAnimationCompleted?.Invoke();
            };

            storyboard.Begin();
        }
    }
}
