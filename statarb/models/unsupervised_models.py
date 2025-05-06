import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
import os
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AssetClustering:
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.clusters = None
        self.cluster_centers = None
        self.feature_names = None
    
    def generate_return_features(
        self, 
        price_data: pd.DataFrame, 
        lookback_periods: List[int] = [5, 10, 20, 60, 120]
    ) -> pd.DataFrame:
        """
        Generate return-based features for clustering.
        
        Args:
            price_data: DataFrame of price data
            lookback_periods: List of lookback periods for features
            
        Returns:
            DataFrame with features
        """
        features = pd.DataFrame()
        
        # Calculate daily returns
        returns = price_data.pct_change().dropna()
        
        # Calculate log returns
        log_returns = np.log(price_data / price_data.shift(1)).dropna()
        
        # For each ticker
        for ticker in price_data.columns:
            ticker_features = {}
            
            # Add mean and std of returns
            ticker_features[f'{ticker}_mean_return'] = returns[ticker].mean()
            ticker_features[f'{ticker}_std_return'] = returns[ticker].std()
            
            # Add skewness and kurtosis
            ticker_features[f'{ticker}_skew'] = returns[ticker].skew()
            ticker_features[f'{ticker}_kurt'] = returns[ticker].kurt()
            
            # Add annualized Sharpe ratio
            sharpe = np.sqrt(252) * returns[ticker].mean() / returns[ticker].std()
            ticker_features[f'{ticker}_sharpe'] = sharpe
            
            # Add autocorrelation
            ticker_features[f'{ticker}_autocorr_1'] = returns[ticker].autocorr(lag=1)
            ticker_features[f'{ticker}_autocorr_5'] = returns[ticker].autocorr(lag=5)
            
            # Calculate mean and std for different lookback periods
            for period in lookback_periods:
                # Calculate rolling statistics
                rolling_returns = returns[ticker].rolling(window=period)
                
                # Store mean of the rolling means and stds
                ticker_features[f'{ticker}_mean_{period}d'] = rolling_returns.mean().mean()
                ticker_features[f'{ticker}_std_{period}d'] = rolling_returns.std().mean()
                
                # Calculate cumulative returns for periods
                cum_return = (1 + returns[ticker]).rolling(window=period).apply(
                    lambda x: np.prod(x) - 1
                )
                
                ticker_features[f'{ticker}_cum_return_{period}d'] = cum_return.mean()
            
            # Add to features DataFrame
            features = pd.concat([features, pd.DataFrame([ticker_features])])
        
        # Set index to be the ticker
        features.index = price_data.columns
        
        return features
    
    def generate_correlation_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate correlation-based features for clustering.
        
        Args:
            price_data: DataFrame of price data
            
        Returns:
            DataFrame with correlation features
        """
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Extract features from correlation matrix
        features = pd.DataFrame(index=corr_matrix.index)
        
        # Average correlation with all other assets
        for ticker in corr_matrix.index:
            features.loc[ticker, 'avg_corr'] = corr_matrix[ticker].mean()
            features.loc[ticker, 'max_corr'] = corr_matrix[ticker].drop(ticker).max()
            features.loc[ticker, 'min_corr'] = corr_matrix[ticker].drop(ticker).min()
            features.loc[ticker, 'std_corr'] = corr_matrix[ticker].std()
        
        return features
    
    def fit(
        self, 
        features: pd.DataFrame, 
        use_pca: bool = True, 
        n_components: int = 5
    ) -> Dict:
        """
        Fit clustering model to the data.
        
        Args:
            features: DataFrame with features
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components
            
        Returns:
            Dictionary with clustering results
        """
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA if requested
        if use_pca:
            self.pca = PCA(n_components=min(n_components, features.shape[1]))
            features_reduced = self.pca.fit_transform(features_scaled)
            explained_var = self.pca.explained_variance_ratio_
            logger.info(f"PCA explained variance: {explained_var.sum():.2f}")
        else:
            features_reduced = features_scaled
        
        # Fit KMeans clustering
        self.kmeans.fit(features_reduced)
        
        # Get cluster labels
        cluster_labels = self.kmeans.labels_
        
        # Store clusters
        self.clusters = pd.DataFrame({
            'ticker': features.index,
            'cluster': cluster_labels
        })
        
        # Get cluster centers
        if use_pca:
            # Transform cluster centers back to original feature space
            self.cluster_centers = self.pca.inverse_transform(self.kmeans.cluster_centers_)
        else:
            self.cluster_centers = self.kmeans.cluster_centers_
        
        # Calculate silhouette score
        silhouette = silhouette_score(features_reduced, cluster_labels)
        logger.info(f"Silhouette score: {silhouette:.4f}")
        
        # Create results dictionary
        results = {
            'clusters': self.clusters,
            'cluster_centers': self.cluster_centers,
            'silhouette_score': silhouette,
            'pca_explained_variance': explained_var if use_pca else None,
            'feature_names': self.feature_names
        }
        
        return results
    
    def find_pairs_within_clusters(
        self, 
        return_data: pd.DataFrame, 
        min_correlation: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """
        Find highly correlated pairs within each cluster.
        
        Args:
            return_data: DataFrame with return data
            min_correlation: Minimum correlation for pairs
            
        Returns:
            List of tuples with (ticker1, ticker2, correlation)
        """
        if self.clusters is None:
            raise ValueError("Clusters not found. Run fit() first.")
        
        pairs = []
        
        # Calculate correlation matrix
        corr_matrix = return_data.corr()
        
        # For each cluster
        for cluster_id in self.clusters['cluster'].unique():
            # Get tickers in this cluster
            cluster_tickers = self.clusters[self.clusters['cluster'] == cluster_id]['ticker'].tolist()
            
            # Skip if cluster has fewer than 2 tickers
            if len(cluster_tickers) < 2:
                continue
            
            # Find high correlation pairs
            for i, ticker1 in enumerate(cluster_tickers):
                for ticker2 in cluster_tickers[i+1:]:
                    correlation = corr_matrix.loc[ticker1, ticker2]
                    
                    if correlation >= min_correlation:
                        pairs.append((ticker1, ticker2, correlation))
        
        # Sort pairs by correlation (descending)
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs
    
    def plot_clusters_2d(self, features: pd.DataFrame) -> plt.Figure:
        """
        Plot clusters in 2D using PCA for dimensionality reduction.
        
        Args:
            features: DataFrame with features
            
        Returns:
            Matplotlib figure
        """
        if self.clusters is None:
            raise ValueError("Clusters not found. Run fit() first.")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Apply PCA for visualization (2 components)
        pca_viz = PCA(n_components=2)
        features_2d = pca_viz.fit_transform(features_scaled)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get cluster labels
        labels = self.clusters['cluster'].values
        
        # Create scatter plot with clusters
        scatter = ax.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.7,
            s=100
        )
        
        # Add ticker labels
        for i, ticker in enumerate(self.clusters['ticker']):
            ax.annotate(
                ticker,
                (features_2d[i, 0], features_2d[i, 1]),
                fontsize=8
            )
        
        # Add legend and labels
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        
        ax.set_xlabel(f'PCA Component 1 ({pca_viz.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PCA Component 2 ({pca_viz.explained_variance_ratio_[1]:.2%})')
        ax.set_title('Asset Clusters')
        
        plt.tight_layout()
        
        return fig
    
    def plot_dendrogram(self, features: pd.DataFrame) -> plt.Figure:
        """
        Plot hierarchical clustering dendrogram.
        
        Args:
            features: DataFrame with features
            
        Returns:
            Matplotlib figure
        """
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Apply hierarchical clustering
        linked = linkage(features_scaled, method='ward')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot dendrogram
        dendrogram(
            linked,
            labels=features.index,
            orientation='top',
            leaf_rotation=90,
            ax=ax
        )
        
        ax.set_title('Hierarchical Clustering Dendrogram')
        ax.set_xlabel('Assets')
        ax.set_ylabel('Distance')
        
        plt.tight_layout()
        
        return fig
    
    def plot_correlation_network(
        self, 
        return_data: pd.DataFrame, 
        min_correlation: float = 0.7
    ) -> plt.Figure:
        """
        Plot correlation network of assets.
        
        Args:
            return_data: DataFrame with return data
            min_correlation: Minimum correlation for connecting assets
            
        Returns:
            Matplotlib figure
        """
        # Calculate correlation matrix
        corr_matrix = return_data.corr().abs()
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for ticker in corr_matrix.index:
            G.add_node(ticker)
        
        # Add edges for correlations above threshold
        for i, ticker1 in enumerate(corr_matrix.index):
            for ticker2 in corr_matrix.index[i+1:]:
                correlation = corr_matrix.loc[ticker1, ticker2]
                
                if correlation >= min_correlation:
                    G.add_edge(ticker1, ticker2, weight=correlation)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Get positions using spring layout
        pos = nx.spring_layout(G, seed=self.random_state)
        
        # Get cluster colors for nodes
        if self.clusters is not None:
            cluster_dict = dict(zip(self.clusters['ticker'], self.clusters['cluster']))
            node_colors = [cluster_dict.get(node, 0) for node in G.nodes()]
        else:
            node_colors = 'skyblue'
        
        # Get edge weights for line thickness
        edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        
        # Draw network
        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=True,
            node_color=node_colors,
            cmap='viridis',
            node_size=300,
            font_size=8,
            width=edge_weights,
            alpha=0.7,
            ax=ax
        )
        
        ax.set_title(f'Asset Correlation Network (Min. Correlation: {min_correlation})')
        ax.axis('off')
        
        plt.tight_layout()
        
        return fig
    
    def save_model(self, model_dir: str, model_name: str):
        """
        Save the clustering model to disk.
        
        Args:
            model_dir: Directory to save model
            model_name: Name of model file
        """
        if self.clusters is None:
            raise ValueError("Model not fitted. Run fit() first.")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save KMeans model
        kmeans_path = os.path.join(model_dir, f"{model_name}_kmeans.pkl")
        joblib.dump(self.kmeans, kmeans_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        # Save PCA if used
        if self.pca is not None:
            pca_path = os.path.join(model_dir, f"{model_name}_pca.pkl")
            joblib.dump(self.pca, pca_path)
        
        # Save clusters
        clusters_path = os.path.join(model_dir, f"{model_name}_clusters.csv")
        self.clusters.to_csv(clusters_path, index=False)
        
        # Save feature names
        features_path = os.path.join(model_dir, f"{model_name}_features.pkl")
        joblib.dump(self.feature_names, features_path)
        
        logger.info(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir: str, model_name: str):
        """
        Load a clustering model from disk.
        
        Args:
            model_dir: Directory with model
            model_name: Name of model file
        """
        # Load KMeans model
        kmeans_path = os.path.join(model_dir, f"{model_name}_kmeans.pkl")
        self.kmeans = joblib.load(kmeans_path)
        
        # Load scaler
        scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
        self.scaler = joblib.load(scaler_path)
        
        # Load PCA if exists
        pca_path = os.path.join(model_dir, f"{model_name}_pca.pkl")
        if os.path.exists(pca_path):
            self.pca = joblib.load(pca_path)
        
        # Load clusters
        clusters_path = os.path.join(model_dir, f"{model_name}_clusters.csv")
        self.clusters = pd.read_csv(clusters_path)
        
        # Load feature names
        features_path = os.path.join(model_dir, f"{model_name}_features.pkl")
        self.feature_names = joblib.load(features_path)
        
        logger.info(f"Model loaded from {model_dir}")
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict clusters for new data.
        
        Args:
            features: DataFrame with features
            
        Returns:
            DataFrame with predictions
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted. Run fit() first.")
        
        # Ensure all required features are present
        if not all(feature in features.columns for feature in self.feature_names):
            raise ValueError("Features do not match the fitted model")
        
        # Scale features
        features_scaled = self.scaler.transform(features[self.feature_names])
        
        # Apply PCA if used in fitting
        if self.pca is not None:
            features_reduced = self.pca.transform(features_scaled)
        else:
            features_reduced = features_scaled
        
        # Predict clusters
        cluster_labels = self.kmeans.predict(features_reduced)
        
        # Create DataFrame with predictions
        predictions = pd.DataFrame({
            'ticker': features.index,
            'cluster': cluster_labels
        })
        
        return predictions 