import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import yfinance as yf
from datetime import datetime, timedelta
import time

class StockPCAAnalyzer:
    """PCA analysis for stock market data"""
    
    def __init__(self, tickers, start_date, end_date, min_observations=50):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.min_observations = min_observations
        self.data = None
        self.returns = None
        self.scaled_returns = None
        self.pca = None
        self.components = None
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance using batch download"""
        print(f"Fetching data for {len(self.tickers)} stocks...")
        
        try:
            # Batch download is more efficient
            df = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False,
                threads=False
            )
            
            if df is None or df.empty:
                raise ValueError("Yahoo Finance returned no data.")
            
            price_dict = {}
            
            # Handle both single and multiple ticker downloads
            if len(self.tickers) == 1:
                # Single ticker case
                if 'Close' in df.columns:
                    close_series = df['Close'].dropna()
                    if len(close_series) >= self.min_observations:
                        price_dict[self.tickers[0]] = close_series
            else:
                # Multiple tickers case
                for ticker in self.tickers:
                    try:
                        if ('Close', ticker) in df.columns:
                            close_series = df['Close'][ticker].dropna()
                        elif ticker in df.columns.get_level_values(0):
                            close_series = df[ticker]["Close"].dropna()
                        else:
                            print(f"✗ Ticker {ticker} not found in download")
                            continue
                            
                        if len(close_series) >= self.min_observations:
                            price_dict[ticker] = close_series
                        else:
                            print(f"✗ Insufficient data for {ticker} ({len(close_series)} obs)")
                    except Exception as e:
                        print(f"✗ Error processing {ticker}: {str(e)[:50]}")
            
            if len(price_dict) < 3:
                raise ValueError(f"Only {len(price_dict)} stocks with valid data. Need at least 3 for PCA.")
            
            self.data = pd.DataFrame(price_dict)
            self.data = self.data.dropna()
            
            # Update tickers list to only successful ones
            self.tickers = list(self.data.columns)
            
            print(f"✓ Successfully loaded data for {self.data.shape[1]} stocks: {self.data.shape}")
            
        except Exception as e:
            raise ValueError(f"Failed to fetch data: {str(e)}")
        
        return self
    
    def calculate_returns(self, method='log'):
        """Calculate stock returns"""
        if method == 'log':
            self.returns = np.log(self.data / self.data.shift(1))
        else:
            self.returns = self.data.pct_change()
        
        self.returns = self.returns.dropna()
        print(f"Calculated {method} returns: {self.returns.shape}")
        return self
    
    def standardize_data(self):
        """Standardize the returns data"""
        scaler = StandardScaler()
        self.scaled_returns = pd.DataFrame(
            scaler.fit_transform(self.returns),
            columns=self.returns.columns,
            index=self.returns.index
        )
        return self
    
    def perform_pca(self, n_components=None):
        """Perform PCA on the standardized returns"""
        if n_components is None:
            n_components = min(len(self.tickers), len(self.returns))
        
        self.pca = PCA(n_components=n_components)
        self.components = self.pca.fit_transform(self.scaled_returns)
        
        print(f"\nPCA completed with {n_components} components")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_[:5]}")
        return self
    
    def plot_explained_variance(self):
        """Plot explained variance and cumulative variance"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Explained variance by component
        axes[0].bar(range(1, len(self.pca.explained_variance_ratio_) + 1),
                    self.pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Principal Component', fontsize=12)
        axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
        axes[0].set_title('Explained Variance by Principal Component', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative explained variance
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        axes[1].plot(range(1, len(cumsum) + 1), cumsum, 
                    marker='o', linestyle='-', linewidth=2, markersize=8, color='darkred')
        axes[1].axhline(y=0.95, color='green', linestyle='--', 
                       label='95% Variance', linewidth=2)
        axes[1].axhline(y=0.90, color='orange', linestyle='--', 
                       label='90% Variance', linewidth=2)
        axes[1].set_xlabel('Number of Components', fontsize=12)
        axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
        axes[1].set_title('Cumulative Explained Variance', 
                         fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        n_95 = np.argmax(cumsum >= 0.95) + 1
        n_90 = np.argmax(cumsum >= 0.90) + 1
        print(f"\nComponents needed for 90% variance: {n_90}")
        print(f"Components needed for 95% variance: {n_95}")
    
    def plot_component_loadings(self, n_components=3):
        """Plot loadings for the first n principal components"""
        loadings = pd.DataFrame(
            self.pca.components_[:n_components].T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=self.tickers
        )
        
        fig, axes = plt.subplots(1, n_components, figsize=(6*n_components, 6))
        if n_components == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            loadings_sorted = loadings[f'PC{i+1}'].sort_values()
            colors = ['red' if x < 0 else 'green' for x in loadings_sorted]
            loadings_sorted.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
            ax.set_title(f'PC{i+1} Loadings\n(Variance: {self.pca.explained_variance_ratio_[i]:.2%})',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Loading Value', fontsize=11)
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
        
        return loadings
    
    def plot_biplot(self, pc_x=0, pc_y=1, top_n=10):
        """Create a biplot showing both samples and features"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot samples (daily returns) in PC space
        xs = self.components[:, pc_x]
        ys = self.components[:, pc_y]
        
        scatter = ax.scatter(xs, ys, alpha=0.3, s=30, c=range(len(xs)), cmap='viridis')
        
        # Plot feature vectors (stocks)
        scale_factor = 3
        for i, ticker in enumerate(self.tickers):
            ax.arrow(0, 0, 
                    self.pca.components_[pc_x, i] * scale_factor,
                    self.pca.components_[pc_y, i] * scale_factor,
                    head_width=0.1, head_length=0.1, 
                    fc='red', ec='red', alpha=0.6, linewidth=2)
            ax.text(self.pca.components_[pc_x, i] * scale_factor * 1.1,
                   self.pca.components_[pc_y, i] * scale_factor * 1.1,
                   ticker, fontsize=10, fontweight='bold')
        
        ax.set_xlabel(f'PC{pc_x+1} ({self.pca.explained_variance_ratio_[pc_x]:.2%} variance)',
                     fontsize=12)
        ax.set_ylabel(f'PC{pc_y+1} ({self.pca.explained_variance_ratio_[pc_y]:.2%} variance)',
                     fontsize=12)
        ax.set_title('PCA Biplot: Stock Relationships in Principal Component Space',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        
        plt.colorbar(scatter, ax=ax, label='Time (Days)')
        plt.tight_layout()
        plt.show()
    
    def cluster_stocks(self, n_clusters=3):
        """Cluster stocks based on their PCA representation"""
        # Use first few components that explain 90% variance
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= 0.90) + 1
        
        pca_features = self.pca.components_[:n_components].T
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(pca_features)
        
        cluster_df = pd.DataFrame({
            'Ticker': self.tickers,
            'Cluster': clusters,
            'PC1': self.pca.components_[0],
            'PC2': self.pca.components_[1]
        })
        
        # Visualize clusters
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for cluster in range(n_clusters):
            mask = clusters == cluster
            ax.scatter(pca_features[mask, 0], pca_features[mask, 1],
                      label=f'Cluster {cluster}', s=100, alpha=0.6)
            
            for i, ticker in enumerate(self.tickers):
                if clusters[i] == cluster:
                    ax.annotate(ticker, (pca_features[i, 0], pca_features[i, 1]),
                              fontsize=9, fontweight='bold')
        
        ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
        ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
        ax.set_title('Stock Clustering Based on PCA', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return cluster_df
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix of stock returns"""
        corr_matrix = self.returns.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        ax.set_title('Stock Returns Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return corr_matrix


def main():
    """Main function to run PCA analysis on stock data"""
    
    # Define stock tickers (major tech and financial stocks)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
               'TSLA', 'NVDA', 'JPM', 'BAC', 'GS',
               'V', 'MA', 'DIS', 'NFLX', 'PYPL']
    
    # Date range (last 2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    print("="*70)
    print("PCA ANALYSIS ON STOCK MARKET DATA")
    print("="*70)
    
    # Initialize analyzer
    analyzer = StockPCAAnalyzer(tickers, start_date, end_date)
    
    # Run analysis pipeline
    analyzer.fetch_data()
    analyzer.calculate_returns(method='log')
    analyzer.standardize_data()
    analyzer.perform_pca(n_components=len(analyzer.tickers))
    
    # Visualizations
    print("\n" + "="*70)
    print("1. EXPLAINED VARIANCE ANALYSIS")
    print("="*70)
    analyzer.plot_explained_variance()
    
    print("\n" + "="*70)
    print("2. CORRELATION MATRIX")
    print("="*70)
    analyzer.plot_correlation_matrix()
    
    print("\n" + "="*70)
    print("3. PRINCIPAL COMPONENT LOADINGS")
    print("="*70)
    loadings = analyzer.plot_component_loadings(n_components=3)
    print("\nTop 5 stocks for PC1:")
    print(loadings['PC1'].abs().sort_values(ascending=False).head())
    
    print("\n" + "="*70)
    print("4. BIPLOT VISUALIZATION")
    print("="*70)
    analyzer.plot_biplot(pc_x=0, pc_y=1)
    
    print("\n" + "="*70)
    print("5. STOCK CLUSTERING")
    print("="*70)
    cluster_df = analyzer.cluster_stocks(n_clusters=3)
    print("\nStock Clusters:")
    for cluster in sorted(cluster_df['Cluster'].unique()):
        stocks = cluster_df[cluster_df['Cluster'] == cluster]['Ticker'].tolist()
        print(f"Cluster {cluster}: {', '.join(stocks)}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    # Summary statistics
    print("\nKey Insights:")
    print(f"- Total variance explained by PC1: {analyzer.pca.explained_variance_ratio_[0]:.2%}")
    print(f"- Total variance explained by PC1+PC2: {sum(analyzer.pca.explained_variance_ratio_[:2]):.2%}")
    print(f"- Number of components for 95% variance: {np.argmax(np.cumsum(analyzer.pca.explained_variance_ratio_) >= 0.95) + 1}")


if __name__ == "__main__":
    main()