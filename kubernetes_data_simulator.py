import pandas as pd
import numpy as np
import random
import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import seaborn as sns

class KubernetesDataSimulator:
    """
    A class to simulate Kubernetes cluster metrics and issue data for training
    predictive models.
    """
    
    def __init__(self, num_samples=5000, random_state=42):
        """
        Initialize the Kubernetes data simulator.
        
        Args:
            num_samples (int): Number of data points to generate
            random_state (int): Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Define possible issue types
        self.issue_types = [
            'None',  # No issue
            'node_failure',
            'pod_crash_loop',
            'memory_pressure',
            'cpu_throttling',
            'network_latency',
            'disk_pressure',
            'api_server_latency',
            'etcd_high_latency',
            'scheduler_failure'
        ]
        
        # Define node names
        self.node_names = [f'node-{i:03d}' for i in range(1, 21)]
        
        # Define namespaces
        self.namespaces = ['default', 'kube-system', 'monitoring', 'app-prod', 
                           'app-staging', 'app-dev', 'database', 'messaging']
        
        # Define workload types
        self.workload_types = ['deployment', 'statefulset', 'daemonset', 'job', 'cronjob']
        
        print(f"Initialized Kubernetes Data Simulator with {num_samples} samples")
    
    def generate_base_metrics(self):
        """
        Generate base metrics for normal operation.
        
        Returns:
            dict: Dictionary of base metrics
        """
        # Generate base metrics for normal operation
        base_metrics = {
            'node': random.choice(self.node_names),
            'namespace': random.choice(self.namespaces),
            'workload_type': random.choice(self.workload_types),
            'pod_count': random.randint(3, 50),
            'cpu_usage_percent': random.uniform(10, 60),
            'memory_usage_percent': random.uniform(20, 70),
            'disk_usage_percent': random.uniform(30, 70),
            'network_receive_bytes': random.uniform(1000, 10000000),
            'network_transmit_bytes': random.uniform(1000, 8000000),
            'pod_restart_count': random.randint(0, 2),
            'pod_ready_percent': random.uniform(98, 100),
            'node_ready_status': 1,
            'api_server_latency_ms': random.uniform(1, 50),
            'etcd_latency_ms': random.uniform(0.5, 10),
            'scheduler_latency_ms': random.uniform(1, 30),
            'container_runtime_errors': random.randint(0, 1),
            'kubelet_rpc_rate': random.uniform(10, 100),
            'pvc_usage_percent': random.uniform(10, 80),
            'time_of_day': random.randint(0, 23),
            'day_of_week': random.randint(0, 6),
            'is_weekend': 1 if random.randint(0, 6) >= 5 else 0,
            'cluster_age_days': random.randint(1, 365)
        }
        
        return base_metrics
    
    def inject_issue(self, base_metrics, issue_type):
        """
        Inject specific issue patterns into base metrics.
        
        Args:
            base_metrics (dict): Base metrics to modify
            issue_type (str): Type of issue to inject
            
        Returns:
            dict: Modified metrics reflecting the issue
        """
        metrics = base_metrics.copy()
        
        # Don't modify if no issue
        if issue_type == 'None':
            return metrics
        
        # Inject patterns based on issue type
        if issue_type == 'node_failure':
            metrics['node_ready_status'] = 0
            metrics['pod_ready_percent'] = random.uniform(0, 60)
            metrics['pod_restart_count'] = random.randint(5, 20)
            
        elif issue_type == 'pod_crash_loop':
            metrics['pod_restart_count'] = random.randint(10, 50)
            metrics['container_runtime_errors'] = random.randint(3, 10)
            metrics['pod_ready_percent'] = random.uniform(50, 90)
            
        elif issue_type == 'memory_pressure':
            metrics['memory_usage_percent'] = random.uniform(85, 100)
            metrics['pod_restart_count'] = random.randint(1, 10)
            metrics['container_runtime_errors'] = random.randint(1, 5)
            
        elif issue_type == 'cpu_throttling':
            metrics['cpu_usage_percent'] = random.uniform(80, 100)
            metrics['scheduler_latency_ms'] = random.uniform(50, 200)
            metrics['api_server_latency_ms'] = random.uniform(100, 300)
            
        elif issue_type == 'network_latency':
            metrics['network_receive_bytes'] = random.uniform(50000000, 100000000)
            metrics['network_transmit_bytes'] = random.uniform(40000000, 90000000)
            metrics['api_server_latency_ms'] = random.uniform(200, 500)
            metrics['etcd_latency_ms'] = random.uniform(50, 200)
            
        elif issue_type == 'disk_pressure':
            metrics['disk_usage_percent'] = random.uniform(85, 100)
            metrics['pvc_usage_percent'] = random.uniform(90, 100)
            metrics['pod_restart_count'] = random.randint(1, 5)
            
        elif issue_type == 'api_server_latency':
            metrics['api_server_latency_ms'] = random.uniform(300, 1000)
            metrics['kubelet_rpc_rate'] = random.uniform(5, 20)
            metrics['pod_restart_count'] = random.randint(0, 3)
            
        elif issue_type == 'etcd_high_latency':
            metrics['etcd_latency_ms'] = random.uniform(50, 500)
            metrics['api_server_latency_ms'] = random.uniform(100, 300)
            
        elif issue_type == 'scheduler_failure':
            metrics['scheduler_latency_ms'] = random.uniform(200, 1000)
            metrics['pod_ready_percent'] = random.uniform(60, 90)
            metrics['kubelet_rpc_rate'] = random.uniform(1, 10)
        
        return metrics
    
    def generate_dataset(self, output_file='kubernetes_metrics_dataset.csv', issue_probability=0.3):
        """
        Generate a complete dataset of simulated Kubernetes metrics.
        
        Args:
            output_file (str): Path to save the generated CSV dataset
            issue_probability (float): Probability of an issue occurring in a sample
            
        Returns:
            pandas.DataFrame: Generated dataset
        """
        print(f"Generating dataset with {self.num_samples} samples...")
        
        data = []
        
        # Generate timestamps covering a period
        start_date = datetime.datetime.now() - datetime.timedelta(days=30)
        timestamps = [start_date + datetime.timedelta(minutes=i*15) for i in range(self.num_samples)]
        
        for i in range(self.num_samples):
            # Decide if this sample has an issue
            if random.random() < issue_probability:
                issue_type = random.choice(self.issue_types[1:])  # Skip 'None'
            else:
                issue_type = 'None'
                
            # Generate base metrics
            base_metrics = self.generate_base_metrics()
            
            # Inject issue patterns if applicable
            metrics = self.inject_issue(base_metrics, issue_type)
            
            # Add timestamp and issue type
            metrics['timestamp'] = timestamps[i]
            metrics['issue_type'] = issue_type
            
            data.append(metrics)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Add some noise to make the dataset more realistic
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            if col not in ['timestamp', 'is_weekend', 'node_ready_status', 'day_of_week']:
                noise = np.random.normal(0, df[col].std() * 0.05, size=len(df))
                df[col] = df[col] + noise
                
                # Ensure values make sense (e.g., no negative counts)
                if col.endswith('count') or col.endswith('bytes') or col.endswith('rate'):
                    df[col] = df[col].clip(lower=0)
                elif col.endswith('percent'):
                    df[col] = df[col].clip(lower=0, upper=100)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Dataset saved to {output_file}")
        
        # Print distribution of issue types
        issue_counts = df['issue_type'].value_counts()
        print("\nIssue type distribution:")
        for issue, count in issue_counts.items():
            print(f"  {issue}: {count} samples ({count/len(df)*100:.1f}%)")
        
        return df
    
    def visualize_dataset(self, df, output_dir='visualizations'):
        """
        Create visualizations of the generated dataset.
        
        Args:
            df (pandas.DataFrame): The dataset to visualize
            output_dir (str): Directory to save visualizations
            
        Returns:
            None
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGenerating visualizations...")
        
        # Issue type distribution
        plt.figure(figsize=(12, 6))
        issue_counts = df['issue_type'].value_counts()
        issue_counts.plot(kind='bar')
        plt.title('Distribution of Issue Types')
        plt.xlabel('Issue Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'issue_distribution.png'))
        
        # CPU vs Memory usage by issue type
        plt.figure(figsize=(12, 8))
        for issue in df['issue_type'].unique():
            subset = df[df['issue_type'] == issue]
            plt.scatter(subset['cpu_usage_percent'], subset['memory_usage_percent'], 
                        alpha=0.5, label=issue)
        plt.title('CPU vs Memory Usage by Issue Type')
        plt.xlabel('CPU Usage (%)')
        plt.ylabel('Memory Usage (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cpu_vs_memory.png'))
        
        # API server latency distribution
        plt.figure(figsize=(12, 6))
        for issue in df['issue_type'].unique():
            subset = df[df['issue_type'] == issue]
            if len(subset) > 10:  # Only plot if enough samples
                plt.hist(subset['api_server_latency_ms'], alpha=0.5, label=issue, bins=20)
        plt.title('API Server Latency Distribution by Issue Type')
        plt.xlabel('API Server Latency (ms)')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'api_latency_distribution.png'))
        
        # Pod restart count by issue type
        plt.figure(figsize=(12, 6))
        sns_data = df[['issue_type', 'pod_restart_count']].copy()
        sns_data = sns_data[sns_data['issue_type'] != 'None'].copy()  # Exclude 'None' for better visualization
        sns.boxplot(x='issue_type', y='pod_restart_count', data=sns_data)
        plt.title('Pod Restart Count by Issue Type')
        plt.xlabel('Issue Type')
        plt.ylabel('Pod Restart Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pod_restart_by_issue.png'))
        
        # Correlation heatmap of numeric features
        plt.figure(figsize=(14, 12))
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix of Numeric Features')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
        
        print(f"Visualizations saved to {output_dir} directory")

# Example usage
if __name__ == "__main__":
    # Create simulator
    simulator = KubernetesDataSimulator(num_samples=10000, random_state=42)
    
    # Generate dataset
    dataset = simulator.generate_dataset(output_file="kubernetes_metrics_dataset.csv", issue_probability=0.3)
    
    # Create visualizations
    simulator.visualize_dataset(dataset, output_dir="visualizations")
    
    print("Data generation and visualization complete!")