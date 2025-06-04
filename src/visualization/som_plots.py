"""
Performance plotting utilities using efficient libraries
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Configure efficient plotting
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100

logger = logging.getLogger(__name__)


class PerformancePlotter:
    """Efficient performance visualizations using seaborn/pandas"""
    
    def __init__(self, style='whitegrid', palette='husl'):
        """Initialize with seaborn styling"""
        sns.set_theme(style=style, palette=palette)
        self.style = style
        self.palette = palette
    
    def plot_response_times(self, results: Dict[str, Any], filepath: Path) -> Optional[str]:
        """Response time analysis using seaborn"""
        try:
            # Convert to DataFrame for efficient plotting
            requests_data = results.get('completed_requests_data', [])
            if not requests_data:
                return None
            
            df = pd.DataFrame(requests_data)
            if 'response_time_ms' not in df.columns:
                return None
            
            # Remove invalid data
            df = df[df['response_time_ms'] > 0]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Response Time Analysis - {results.get("algorithm", "Unknown")}', fontsize=14)
            
            # 1. Distribution plot
            sns.histplot(data=df, x='response_time_ms', kde=True, ax=axes[0, 0])
            axes[0, 0].set_title('Response Time Distribution')
            axes[0, 0].axvline(df['response_time_ms'].mean(), color='red', linestyle='--', label='Mean')
            axes[0, 0].legend()
            
            # 2. Box plot
            sns.boxplot(data=df, y='response_time_ms', ax=axes[0, 1])
            axes[0, 1].set_title('Response Time Box Plot')
            
            # 3. CDF using empirical distribution
            sorted_times = np.sort(df['response_time_ms'])
            y_vals = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
            axes[1, 0].plot(sorted_times, y_vals * 100)
            axes[1, 0].set_xlabel('Response Time (ms)')
            axes[1, 0].set_ylabel('Cumulative Probability (%)')
            axes[1, 0].set_title('Cumulative Distribution')
            axes[1, 0].grid(True)
            
            # 4. Summary statistics
            stats = df['response_time_ms'].describe()
            axes[1, 1].axis('off')
            stats_text = f"""
            Count: {stats['count']:.0f}
            Mean: {stats['mean']:.2f} ms
            Std: {stats['std']:.2f} ms
            Min: {stats['min']:.2f} ms
            25%: {stats['25%']:.2f} ms
            50%: {stats['50%']:.2f} ms
            75%: {stats['75%']:.2f} ms
            Max: {stats['max']:.2f} ms
            """
            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                            verticalalignment='top', fontfamily='monospace')
            axes[1, 1].set_title('Summary Statistics')
            
            plt.tight_layout()
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Response time plot saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating response time plot: {e}")
            return None
    
    def plot_server_utilization(self, results: Dict[str, Any], filepath: Path) -> Optional[str]:
        """Server utilization using pandas time series"""
        try:
            metrics_history = results.get('metrics_history', [])
            if not metrics_history:
                return None
            
            # Convert to efficient DataFrame format
            data_rows = []
            for metric in metrics_history:
                timestamp = metric['timestamp']
                for server in metric.get('servers', []):
                    data_rows.append({
                        'timestamp': timestamp,
                        'server_id': server['server_id'],
                        'utilization': server['utilization_score'] * 100,
                        'cpu_usage': server.get('cpu_usage', 0),
                        'memory_usage': server.get('memory_usage', 0)
                    })
            
            df = pd.DataFrame(data_rows)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f'Server Utilization - {results.get("algorithm", "Unknown")}', fontsize=14)
            
            # 1. Individual server utilization over time
            sns.lineplot(data=df, x='timestamp', y='utilization', hue='server_id', ax=axes[0, 0])
            axes[0, 0].set_title('Individual Server Utilization')
            axes[0, 0].set_ylabel('Utilization (%)')
            
            # 2. Average utilization with confidence interval
            sns.lineplot(data=df, x='timestamp', y='utilization', 
                        estimator='mean', errorbar='sd', ax=axes[0, 1])
            axes[0, 1].set_title('Average Utilization ± Std Dev')
            axes[0, 1].set_ylabel('Utilization (%)')
            
            # 3. Final utilization distribution
            final_util = df.groupby('server_id')['utilization'].last()
            sns.barplot(x=final_util.index, y=final_util.values, ax=axes[1, 0])
            axes[1, 0].set_title('Final Server Utilization')
            axes[1, 0].set_ylabel('Utilization (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. Load balance quality over time
            balance_quality = df.groupby('timestamp')['utilization'].std()
            axes[1, 1].plot(balance_quality.index, balance_quality.values)
            axes[1, 1].set_title('Load Balance Quality (Lower = Better)')
            axes[1, 1].set_xlabel('Time (seconds)')
            axes[1, 1].set_ylabel('Utilization Std Dev')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Server utilization plot saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating server utilization plot: {e}")
            return None
    
    def plot_algorithm_comparison(self, results_list: List[Dict[str, Any]], filepath: Path) -> Optional[str]:
        """Algorithm comparison using seaborn"""
        try:
            if len(results_list) < 2:
                return None
            
            # Convert to DataFrame for efficient plotting
            comparison_data = []
            for result in results_list:
                comparison_data.append({
                    'Algorithm': result['algorithm'],
                    'Success Rate (%)': result.get('success_rate', 0),
                    'Avg Response Time (ms)': result.get('response_time_stats', {}).get('mean', 0),
                    'P95 Response Time (ms)': result.get('response_time_stats', {}).get('p95', 0),
                    'Load Balance Quality': 1 / (result.get('server_utilization_stats', {}).get('std', 1) + 0.001),
                    'Throughput (req/s)': result.get('throughput_rps', 0)
                })
            
            df = pd.DataFrame(comparison_data)
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('Algorithm Comparison', fontsize=16)
            
            # Use seaborn for efficient plotting
            metrics = ['Success Rate (%)', 'Avg Response Time (ms)', 'P95 Response Time (ms)', 
                      'Load Balance Quality', 'Throughput (req/s)']
            
            for i, metric in enumerate(metrics):
                row, col = divmod(i, 3)
                sns.barplot(data=df, x='Algorithm', y=metric, ax=axes[row, col])
                axes[row, col].set_title(metric)
                axes[row, col].tick_params(axis='x', rotation=45)
            
            # Overall ranking in last subplot
            axes[1, 2].axis('off')
            
            # Calculate normalized scores
            score_cols = ['Success Rate (%)', 'Load Balance Quality', 'Throughput (req/s)']
            penalty_cols = ['Avg Response Time (ms)', 'P95 Response Time (ms)']
            
            # Normalize to 0-100 scale
            for col in score_cols:
                df[f'{col}_norm'] = 100 * (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 0.001)
            
            for col in penalty_cols:
                df[f'{col}_norm'] = 100 * (1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 0.001))
            
            # Overall score
            df['Overall Score'] = (df['Success Rate (%)_norm'] * 0.3 + 
                                  df['Avg Response Time (ms)_norm'] * 0.3 +
                                  df['Load Balance Quality_norm'] * 0.2 + 
                                  df['Throughput (req/s)_norm'] * 0.2)
            
            # Display ranking
            ranking = df[['Algorithm', 'Overall Score']].sort_values('Overall Score', ascending=False)
            ranking_text = "Algorithm Ranking:\n" + "\n".join([
                f"{i+1}. {row['Algorithm']}: {row['Overall Score']:.1f}/100" 
                for i, (_, row) in enumerate(ranking.iterrows())
            ])
            axes[1, 2].text(0.1, 0.9, ranking_text, transform=axes[1, 2].transAxes, 
                           verticalalignment='top', fontfamily='monospace')
            axes[1, 2].set_title('Overall Ranking')
            
            plt.tight_layout()
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Algorithm comparison plot saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating algorithm comparison plot: {e}")
            return None
    
    def generate_report_plots(self, results: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
        """Generate all plots efficiently"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        
        # Generate plots using efficient methods
        plots['response_times'] = self.plot_response_times(results, output_dir / 'response_times.png')
        plots['server_utilization'] = self.plot_server_utilization(results, output_dir / 'server_utilization.png')
        
        return {k: v for k, v in plots.items() if v is not None}