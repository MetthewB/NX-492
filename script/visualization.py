import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

def plot_detailed_cost_history(target_results, smoothing_window=10):
    """Plots Kinematic, Neural, and Force costs for each direction."""
    
    def smooth_data(data, window):
        if len(data) < window: return data 
        pad_head = np.full(window - 1, data[0])
        padded_data = np.concatenate([pad_head, data])
        kernel = np.ones(window) / window
        smoothed = np.convolve(padded_data, kernel, mode='valid')
        return smoothed

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=False, sharey=False)
    direction_names = ["Right", "Left", "Up", "Down", "Up-Right", "Up-Left", "Down-Left", "Down-Right"]
    spatial_map = {
        "Up-Left":    (0, 0), "Up":   (0, 1), "Up-Right":   (0, 2),
        "Left":       (1, 0),                 "Right":      (1, 2),
        "Down-Left":  (2, 0), "Down": (2, 1), "Down-Right": (2, 2)
    }
    cost_colors = {"kinematic": "tab:red", "neural": "tab:blue", "force": "tab:green"}
    
    for idx, data in target_results.items():
        if idx >= len(direction_names): continue
        dir_name = direction_names[idx]
        
        if dir_name in spatial_map:
            row, col = spatial_map[dir_name]
            ax = axes[row, col]
            raw_costs = {
                "kinematic": np.array(data["costs"]["kinematic"]),
                "neural":    np.array(data["costs"]["neural"]),
                "force":     np.array(data["costs"]["force"])
            }
            
            for cost_type, raw_data in raw_costs.items():
                if len(raw_data) > 0:
                    color = cost_colors[cost_type]
                    ax.plot(raw_data, color=color, alpha=0.15, linewidth=1.0)
                    smooth_y = smooth_data(raw_data, smoothing_window)
                    x_axis = np.arange(len(smooth_y))
                    ax.plot(x_axis, smooth_y, color=color, alpha=0.9, linewidth=2.0, label=cost_type.capitalize())

            ax.set_title(dir_name, fontsize=14, fontweight='bold')
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-4) 
            ax.set_xlabel("Trials", fontsize=11)
            ax.set_ylabel("Cost", fontsize=11)

    center_ax = axes[1, 1]
    center_ax.axis("off") 
    legend_elements = [
        Line2D([0], [0], color=cost_colors["kinematic"], lw=3, label='Kinematic Cost'),
        Line2D([0], [0], color=cost_colors["neural"],    lw=3, label='Neural Cost'),
        Line2D([0], [0], color=cost_colors["force"],     lw=3, label='Force Cost'),
        Line2D([0], [0], color='gray', lw=1, alpha=0.5, label='(Raw Data Shadow)')
    ]
    center_ax.legend(handles=legend_elements, loc='center', title="Cost Components", fontsize='x-large', title_fontsize='xx-large')

    plt.suptitle(f"Detailed Cost Evolution per Direction", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def plot_trajectory_evolution_grid_xy(epoch_qpos_over_time, epoch_metadata, dt=0.01):
    """Plots evolution of trajectories for all 8 directions (Early vs Late)."""
    fig = plt.figure(figsize=(16, 13))
    outer_grid = gridspec.GridSpec(3, 3, figure=fig, wspace=0.3, hspace=0.3)
    spatial_map = {
        "Up-Left":    (0, 0), "Up":   (0, 1), "Up-Right":   (0, 2),
        "Left":       (1, 0),                 "Right":      (1, 2),
        "Down-Left":  (2, 0), "Down": (2, 1), "Down-Right": (2, 2)
    }
    phases = [
        {"name": "Early",  "color": "tab:blue", "indices": lambda n: range(0, 10)},
        {"name": "Late",   "color": "tab:red",  "indices": lambda n: range(n - 10, n)}
    ]

    def get_trajectory_stats(indices):
        x_stack, y_stack = [], []
        min_len = float('inf')
        valid_trajs = []
        for idx in indices:
            traj = epoch_qpos_over_time[idx]
            x, y = zip(*traj)
            valid_trajs.append((x, y))
            if len(x) < min_len: min_len = len(x)
        for vx, vy in valid_trajs:
            x_stack.append(vx[:min_len])
            y_stack.append(vy[:min_len])
        x_stack = np.array(x_stack)
        y_stack = np.array(y_stack)
        return {
            "x_mean": np.mean(x_stack, axis=0), "x_std":  np.std(x_stack, axis=0),
            "y_mean": np.mean(y_stack, axis=0), "y_std":  np.std(y_stack, axis=0),
            "time":   np.arange(min_len) * dt
        }

    for row in range(3):
        for col in range(3):
            target_dir = None
            for name, coords in spatial_map.items():
                if coords == (row, col):
                    target_dir = name
                    break
            if row == 1 and col == 1: continue
            if target_dir is None: continue

            all_indices = [i for i, meta in enumerate(epoch_metadata) if meta["direction"] == target_dir]
            if len(all_indices) < 20: continue

            inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[row, col], hspace=0.05)
            ax_x = fig.add_subplot(inner_grid[0, 0])
            ax_y = fig.add_subplot(inner_grid[1, 0], sharex=ax_x)
            
            for phase in phases:
                idx_range = phase["indices"](len(all_indices))
                batch_indices = [all_indices[i] for i in idx_range]
                stats = get_trajectory_stats(batch_indices)
                t, c = stats["time"], phase["color"]
                
                ax_x.plot(t, stats["x_mean"], color=c, linewidth=2, label=phase["name"])
                ax_x.fill_between(t, stats["x_mean"] - stats["x_std"], stats["x_mean"] + stats["x_std"], color=c, alpha=0.2)
                ax_y.plot(t, stats["y_mean"], color=c, linewidth=2)
                ax_y.fill_between(t, stats["y_mean"] - stats["y_std"], stats["y_mean"] + stats["y_std"], color=c, alpha=0.2)
            
            ax_x.set_title(target_dir, fontsize=14, fontweight='bold', pad=5)
            for ax in [ax_x, ax_y]:
                ax.grid(True, linestyle='--', alpha=0.4)
                ax.axhline(0, color='k', linestyle=':', alpha=0.5)
            ax_x.set_ylabel("X (m)", fontsize=11)
            ax_y.set_ylabel("Y (m)", fontsize=11)
            ax_y.set_xlabel("Time (s)", fontsize=11)
            plt.setp(ax_x.get_xticklabels(), visible=False)

    ax_legend = fig.add_subplot(outer_grid[1, 1])
    ax_legend.axis("off")
    legend_elements = [
        Line2D([0], [0], color='tab:blue', lw=3, label='Early (First 10 Epochs)'),
        Line2D([0], [0], color='tab:red',  lw=3, label='Late (Last 10 Epochs)')
    ]
    ax_legend.legend(handles=legend_elements, loc='center', title="Training Progress\n(Mean $\pm$ Std)", fontsize='x-large', title_fontsize='xx-large')
    plt.suptitle("Trajectory Evolution: Early vs. Late Training", fontsize=20, y=0.95)
    plt.show()

def plot_detailed_trajectory_comparison(epoch_metadata, epoch_qpos_over_time, perturbation_directions):
    """Plots a 3x3 grid comparing High Load vs Low Load trajectories."""
    
    def plot_gradient_line(ax, x, y, color, alpha_start=0.1, alpha_end=1.0, linewidth=2.5):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        n_segments = len(segments)
        target_rgb = np.array(to_rgba(color)[:3])
        white_rgb = np.array([1.0, 1.0, 1.0]) 
        t = np.linspace(alpha_start, alpha_end, n_segments)[:, None]
        gradient_colors = white_rgb * (1 - t) + target_rgb * t
        lc = LineCollection(segments, colors=gradient_colors, linewidth=linewidth, linestyle='solid', zorder=3, capstyle='round')
        ax.add_collection(lc)
        return lc

    fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=False, sharey=False)
    spatial_map = {
        "Up-Left":    (0, 0), "Up":   (0, 1), "Up-Right":   (0, 2),
        "Left":       (1, 0),                 "Right":      (1, 2),
        "Down-Left":  (2, 0), "Down": (2, 1), "Down-Right": (2, 2)
    }
    direction_colors = {
        "Right": "tab:blue", "Left": "tab:orange", "Up": "tab:green", "Down": "tab:red",
        "Up-Right": "tab:purple", "Up-Left": "tab:brown", "Down-Left": "tab:pink", "Down-Right": "tab:gray"
    }

    last_indices = {"Low": {}, "High": {}}
    for i, meta in enumerate(epoch_metadata):
        d, l = meta["direction"], meta["load"]
        last_indices[l][d] = i

    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            current_dir = None
            for name, coords in spatial_map.items():
                if coords == (row, col):
                    current_dir = name
                    break
            
            if row == 1 and col == 1:
                ax.axis("off")
                continue
            if current_dir is None:
                ax.axis("off")
                continue

            raw_vec = perturbation_directions[current_dir]
            norm_vec = raw_vec / np.linalg.norm(raw_vec)
            arrow_len = 0.10 * 100  # Convert arrow length to cm
            ax.arrow(0, 0, norm_vec[0]*arrow_len, norm_vec[1]*arrow_len, color='lightgray', width=0.003*100, head_width=0.01*100, zorder=1)
            all_x_points, all_y_points = [0, norm_vec[0]*arrow_len], [0, norm_vec[1]*arrow_len]

            for load_type, style in [("Low", "--"), ("High", "-")]:
                if current_dir in last_indices[load_type]:
                    idx = last_indices[load_type][current_dir]
                    traj = epoch_qpos_over_time[idx]
                    x, y = zip(*traj)
                    x = np.array(x) * 100  # Convert x-coordinates to cm
                    y = np.array(y) * 100  # Convert y-coordinates to cm
                    color = direction_colors.get(current_dir, "black")
                    
                    if load_type == "High":
                        plot_gradient_line(ax, x, y, color, alpha_start=0.15, alpha_end=1.0)
                    else:
                        ax.plot(x, y, color=color, linestyle=style, linewidth=2.5, alpha=0.9, zorder=3)
                    all_x_points.extend(x)
                    all_y_points.extend(y)

            ax.scatter([0], [0], color='black', marker='x', s=100, linewidth=2, zorder=5)
            min_x, max_x = min(all_x_points), max(all_x_points)
            min_y, max_y = min(all_y_points), max(all_y_points)
            span_x, span_y = max_x - min_x, max_y - min_y
            max_span = max(span_x, span_y)
            padding = max_span * 0.15
            final_span = max_span + (padding * 2)
            center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
            
            ax.set_xlim(center_x - final_span/2, center_x + final_span/2)
            ax.set_ylim(center_y - final_span/2, center_y + final_span/2)
            ax.set_title(current_dir, fontsize=14, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_aspect('equal')
            ax.set_xlabel("X (cm)", fontsize=9)  # Update label to cm
            ax.set_ylabel("Y (cm)", fontsize=9)  # Update label to cm
            ax.tick_params(labelsize=8)

    center_ax = axes[1, 1]
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='High Load'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Low Load'),
        mpatches.FancyArrowPatch((0,0), (1,0), color='lightgray', label='Force Direction', mutation_scale=15)
    ]
    center_ax.legend(handles=legend_elements, loc='center', title="Comparison of stiffness\nstrategies per direction", fontsize='x-large', title_fontsize='xx-large')
    plt.suptitle("Trajectory Comparison: Low vs. High Load", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def plot_neural_activity_comparison(epoch_metadata, epoch_neural_activity, Tp_step, dt=0.01):
    """Plots the Motor RNN activity (y_m) for High vs Low background loads."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=False, sharey=False)
    spatial_map = {
        "Up-Left": (0, 0), "Up": (0, 1), "Up-Right": (0, 2),
        "Left": (1, 0), "Right": (1, 2),
        "Down-Left": (2, 0), "Down": (2, 1), "Down-Right": (2, 2)
    }
    last_indices = {"Low": {}, "High": {}}
    for i, meta in enumerate(epoch_metadata):
        d, l = meta["direction"], meta["load"]
        last_indices[l][d] = i

    window_start_ms, window_end_ms = 50, 100
    
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            target_dir = None
            for name, coords in spatial_map.items():
                if coords == (row, col):
                    target_dir = name
                    break
            
            if row == 1 and col == 1:
                ax.axis("off")
                continue
            if target_dir is None:
                ax.axis("off")
                continue

            for load_type, style, color in [("Low", "--", "tab:blue"), ("High", "-", "tab:red")]:
                if target_dir in last_indices[load_type]:
                    idx = last_indices[load_type][target_dir]
                    activity = np.array(epoch_neural_activity[idx])
                    time_axis = (np.arange(len(activity)) - Tp_step) * 1000 * dt
                    label = f"{load_type} Load" if row==0 and col==0 else None
                    ax.plot(time_axis, activity, color=color, linestyle=style, linewidth=2.5, label=label, alpha=0.9)

            ax.axvspan(window_start_ms, window_end_ms, color='yellow', alpha=0.15, zorder=0)
            ax.axvline(0, color='black', linestyle=':', alpha=0.5, linewidth=1.5)
            ax.set_title(target_dir, fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_xlim(-50, 300)
            ax.set_xlabel("Time post-perturbation (ms)", fontsize=11)
            ax.set_ylabel("RNN Activity", fontsize=11)

    center_ax = axes[1, 1]
    legend_elements = [
        Line2D([0], [0], color='tab:red', lw=3, label='High Load\n(High Spinal Gain)'),
        Line2D([0], [0], color='tab:blue', lw=3, linestyle='--', label='Low Load\n(Low Spinal Gain)'),
        mpatches.Patch(color='yellow', alpha=0.2, label='Reciprocal Window\n(50-100ms)')
    ]
    center_ax.legend(handles=legend_elements, loc='center', title="Neural Response", fontsize='x-large', title_fontsize='xx-large')
    plt.suptitle("Reciprocal Reduction: Cortical Activity vs Spinal Gain", fontsize=18, y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()