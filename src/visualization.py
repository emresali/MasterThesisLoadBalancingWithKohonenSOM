import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from typing import List, Dict
import networkx as nx

class SOMVisualizer:
    def __init__(self, balancer):
        self.balancer = balancer
        self.som_size = balancer.som_size
        
    def create_load_animation(self, save_path: str = "load_shift_animation.gif"):
        """Create animation showing load shifts over time"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        def update(frame):
            if frame >= len(self.balancer.load_history):
                return
            
            state = self.balancer.load_history[frame]
            
            # Clear axes
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            
            # 1. Load heatmap with activation levels
            self._plot_load_heatmap(ax1, state)
            
            # 2. Response time heatmap
            self._plot_response_time_heatmap(ax2, state)
            
            # 3. Queue lengths bar chart
            self._plot_queue_lengths(ax3, state)
            
            # 4. Load shift arrows
            self._plot_load_shifts(ax4, frame)
            
            fig.suptitle(f'Load Balancing Dynamics - Step {frame}', fontsize=14)
        
        anim = animation.FuncAnimation(fig, update, frames=len(self.balancer.load_history),
                                     interval=200, blit=False)
        anim.save(save_path, writer='pillow')
        plt.close()
    
    def _plot_load_heatmap(self, ax, state):
        """Plot VM utilization with activation overlay"""
        util_grid = np.zeros((self.som_size, self.som_size))
        activation_grid = np.zeros((self.som_size, self.som_size))
        
        for vm_state in state['vm_states']:
            pos = vm_state['pos']
            util_grid[pos] = vm_state['utilization']
            activation_grid[pos] = vm_state['activation_level']
        
        # Plot utilization as heatmap
        im = ax.imshow(util_grid, cmap='YlOrRd', vmin=0, vmax=1)
        
        # Overlay activation as circles
        for i in range(self.som_size):
            for j in range(self.som_size):
                if activation_grid[i, j] > 0.1:
                    circle = plt.Circle((j, i), activation_grid[i, j] * 0.4, 
                                      color='blue', alpha=0.3)
                    ax.add_patch(circle)
        
        ax.set_title('VM Load (color) + Activation (blue circles)')
        ax.set_xlabel('SOM X')
        ax.set_ylabel('SOM Y')
        
    def _plot_response_time_heatmap(self, ax, state):
        """Plot response times across VMs"""
        rt_grid = np.zeros((self.som_size, self.som_size))
        
        for vm_state in state['vm_states']:
            pos = vm_state['pos']
            rt_grid[pos] = vm_state['avg_response_time']
        
        sns.heatmap(rt_grid, ax=ax, cmap='viridis', 
                   cbar_kws={'label': 'Avg Response Time (ms)'})
        ax.set_title('Response Time Distribution')
        
    def _plot_queue_lengths(self, ax, state):
        """Bar chart of queue lengths"""
        positions = []
        queue_lengths = []
        colors = []
        
        for vm_state in state['vm_states']:
            if vm_state['queue_length'] > 0:
                positions.append(f"{vm_state['pos'][0]},{vm_state['pos'][1]}")
                queue_lengths.append(vm_state['queue_length'])
                colors.append('red' if vm_state['is_overloaded'] else 'blue')
        
        if positions:
            ax.bar(range(len(positions)), queue_lengths, color=colors)
            ax.set_xticks(range(len(positions)))
            ax.set_xticklabels(positions, rotation=45)
            ax.set_ylabel('Queue Length')
            ax.set_title('Request Queues (red=overloaded)')
        else:
            ax.text(0.5, 0.5, 'No queued requests', ha='center', va='center')
            ax.set_title('Request Queues')
        
    def _plot_load_shifts(self, ax, current_frame):
        """Show load shift events as arrows"""
        # Get recent load shifts
        recent_shifts = [e for e in self.balancer.load_shift_events 
                        if current_frame - 10 <= e['time'] <= current_frame]
        
        # Create grid positions
        for i in range(self.som_size):
            for j in range(self.som_size):
                ax.plot(j, i, 'o', color='lightgray', markersize=20)
                ax.text(j, i, f"{i},{j}", ha='center', va='center', fontsize=8)
        
        # Draw arrows for load shifts
        for shift in recent_shifts:
            from_pos = shift['from_pos']
            to_pos = shift['to_pos']
            
            # Calculate arrow properties
            alpha = 1 - (current_frame - shift['time']) / 10  # Fade out
            
            arrow = FancyArrowPatch(
                (from_pos[1], from_pos[0]),
                (to_pos[1], to_pos[0]),
                arrowstyle='->', mutation_scale=20,
                color='red', alpha=alpha, linewidth=2
            )
            ax.add_patch(arrow)
        
        ax.set_xlim(-0.5, self.som_size - 0.5)
        ax.set_ylim(-0.5, self.som_size - 0.5)
        ax.invert_yaxis()
        ax.set_title(f'Load Shifts (last 10 steps)')
        ax.set_xlabel('SOM X')
        ax.set_ylabel('SOM Y')
    
    def create_analysis_plots(self, save_path: str = "load_analysis.png"):
        """Create static analysis plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Load balance over time
        times = [s['time_step'] for s in self.balancer.load_history]
        avg_utils = [s['avg_utilization'] for s in self.balancer.load_history]
        overloaded = [s['overloaded_vms'] for s in self.balancer.load_history]
        
        ax1.plot(times, avg_utils, 'b-', label='Avg Utilization')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(times, overloaded, 'r-', label='Overloaded VMs')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Avg Utilization', color='b')
        ax1_twin.set_ylabel('Overloaded VMs', color='r')
        ax1.set_title('System Load Evolution')
        ax1.grid(True, alpha=0.3)
        
        # 2. Response time evolution
        if self.balancer.response_time_history:
            rt_times = [s['time_step'] for s in self.balancer.response_time_history]
            avg_rt = [s['avg_response_time'] for s in self.balancer.response_time_history]
            p95_rt = [s['p95_response_time'] for s in self.balancer.response_time_history]
            
            ax2.plot(rt_times, avg_rt, 'g-', label='Average')
            ax2.plot(rt_times, p95_rt, 'r--', label='P95')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Response Time (ms)')
            ax2.set_title('Response Time Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Load shift frequency heatmap
        shift_matrix = self.balancer.get_load_shift_matrix()
        if shift_matrix.max() > 0:
            sns.heatmap(shift_matrix[:20, :20], ax=ax3, cmap='Blues',
                       cbar_kws={'label': 'Shift Count'})
            ax3.set_title('Load Shift Patterns (first 20 VMs)')
            ax3.set_xlabel('To VM')
            ax3.set_ylabel('From VM')
        
        # 4. Neighbor activation correlation
        final_state = self.balancer.load_history[-1] if self.balancer.load_history else None
        if final_state:
            self._plot_neighbor_correlation(ax4, final_state)
        
        plt.suptitle('Load Balancing Analysis - Neighbor Activation Effects')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_neighbor_correlation(self, ax, state):
        """Plot correlation between neighbor load and activation"""
        utils = []
        activations = []
        
        for vm_state in state['vm_states']:
            utils.append(vm_state['utilization'])
            activations.append(vm_state['activation_level'])
        
        ax.scatter(utils, activations, alpha=0.6)
        ax.set_xlabel('VM Utilization')
        ax.set_ylabel('Activation Level')
        ax.set_title('Utilization vs Neighbor Activation')
        
        # Add trend line
        if len(utils) > 1:
            z = np.polyfit(utils, activations, 1)
            p = np.poly1d(z)
            ax.plot(sorted(utils), p(sorted(utils)), "r--", alpha=0.8)
        
        ax.grid(True, alpha=0.3)
    
    def create_comprehensive_analysis(self, filename: str = "som_analysis.png"):
        """Create comprehensive analysis similar to original but for VMs"""
        self.create_analysis_plots(filename)