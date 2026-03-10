import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA CREATION AND PREPARATION
# ============================================================================

def create_sample_credit_data(n_samples=1000):
    """Create sample credit data for demonstration"""
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'employment_years': np.random.randint(0, 40, n_samples),
        'loan_amount': np.random.normal(20000, 10000, n_samples),
        'credit_history_years': np.random.randint(0, 30, n_samples),
        'num_credit_lines': np.random.randint(1, 10, n_samples),
        'late_payments_12m': np.random.poisson(1, n_samples),
        'debt_to_income': np.random.uniform(0, 0.5, n_samples),
        'default_flag': np.random.binomial(1, 0.2, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Make default more realistic based on features
    df['default_flag'] = ((df['late_payments_12m'] > 2) | 
                          (df['debt_to_income'] > 0.4) | 
                          (df['credit_history_years'] < 2)).astype(int)
    
    return df

def engineer_features(df):
    """Create additional features for credit scoring"""
    df_processed = df.copy()
    
    # Risk ratios
    df_processed['loan_to_income'] = df_processed['loan_amount'] / (df_processed['income'] + 1)
    df_processed['utilization_rate'] = df_processed['loan_amount'] / (df_processed['income'] * 0.3)
    
    # Employment stability
    df_processed['employment_stability'] = df_processed['employment_years'] / (df_processed['age'] - 18 + 1)
    
    # Credit history ratio
    df_processed['credit_history_ratio'] = df_processed['credit_history_years'] / (df_processed['age'] - 18 + 1)
    
    return df_processed

def prepare_data_for_modeling(df):
    """Prepare data for credit scoring model"""
    feature_cols = ['age', 'income', 'employment_years', 'loan_amount', 
                   'credit_history_years', 'num_credit_lines', 
                   'late_payments_12m', 'debt_to_income',
                   'loan_to_income', 'utilization_rate', 
                   'employment_stability', 'credit_history_ratio']
    
    X = df[feature_cols]
    y = df['default_flag']
    
    return X, y

# ============================================================================
# 2. CREDIT SCORECARD CLASS
# ============================================================================

class CreditScorecard:
    """
    Credit scoring model that converts logistic regression coefficients to scorecard points
    """
    def __init__(self, target_points=600, pdo=20, odds=50):
        """
        target_points: Score at target odds
        pdo: Points to double the odds
        odds: Odds at target points (e.g., 50:1 good:bad)
        """
        self.target_points = target_points
        self.pdo = pdo
        self.odds = odds
        self.factor = pdo / np.log(2)
        self.offset = target_points - self.factor * np.log(odds)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def fit(self, X, y):
        """Fit logistic regression and create scorecard"""
        self.feature_names = X.columns.tolist()
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train logistic regression
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict_proba(self, X):
        """Predict probability of default"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def probability_to_score(self, prob_default):
        """Convert default probability to credit score"""
        # odds = (1 - p) / p
        odds = (1 - prob_default) / (prob_default + 1e-10)
        score = self.offset + self.factor * np.log(odds)
        return score
    
    def predict_score(self, X):
        """Predict credit scores for new data"""
        prob_default = self.predict_proba(X)[:, 1]
        scores = self.probability_to_score(prob_default)
        # Cap scores to reasonable range
        scores = np.clip(scores, 300, 850)
        return scores
    
    def get_scorecard_points(self):
        """Calculate points for each feature based on coefficients"""
        coefficients = self.model.coef_[0]
        intercept = self.model.intercept_[0]
        
        # Calculate points
        points = []
        for i, (feature, coef) in enumerate(zip(self.feature_names, coefficients)):
            # Points per one standard deviation change
            point_value = -coef * self.factor
            points.append({
                'feature': feature,
                'coefficient': coef,
                'points_per_std': point_value
            })
        
        # Base points from intercept
        base_points = -intercept * self.factor
        
        return pd.DataFrame(points), base_points

# ============================================================================
# 3. MODEL EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(y_true, y_pred_proba, scores):
    """Evaluate credit scoring model performance"""
    
    # Classification metrics
    y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # AUC-ROC
    auc_roc = roc_auc_score(y_true, y_pred_proba[:, 1])
    print(f"\nAUC-ROC: {auc_roc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # Score distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist([scores[y_true==0], scores[y_true==1]], 
             label=['Good', 'Bad'], bins=30, alpha=0.7)
    plt.xlabel('Credit Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution by Class')
    plt.legend()
    
    # Default rate by score bin
    plt.subplot(1, 2, 2)
    score_bins = pd.cut(scores, bins=10)
    default_rate = pd.Series(y_true).groupby(score_bins).mean()
    default_rate.plot(kind='bar', rot=45)
    plt.xlabel('Score Range')
    plt.ylabel('Default Rate')
    plt.title('Default Rate by Score Bin')
    plt.tight_layout()
    plt.show()
    
    return auc_roc

# ============================================================================
# 4. CREDIT DECISION ENGINE
# ============================================================================

class CreditDecisionEngine:
    """
    Makes credit decisions based on scores and business rules
    """
    def __init__(self, scorecard, approval_threshold=650, review_threshold=580):
        self.scorecard = scorecard
        self.approval_threshold = approval_threshold
        self.review_threshold = review_threshold
        
    def make_decision(self, X):
        """Make credit decision for applicants"""
        scores = self.scorecard.predict_score(X)
        prob_default = self.scorecard.predict_proba(X)[:, 1]
        
        decisions = []
        for score, prob in zip(scores, prob_default):
            if score >= self.approval_threshold:
                decision = 'APPROVED'
                reason = 'Good credit score'
            elif score >= self.review_threshold:
                decision = 'REVIEW'
                reason = 'Borderline score - requires manual review'
            else:
                decision = 'DECLINED'
                reason = 'Credit score below minimum threshold'
            
            decisions.append({
                'score': score,
                'probability_default': prob,
                'decision': decision,
                'reason': reason
            })
        
        return pd.DataFrame(decisions)
    
    def suggest_loan_terms(self, X, base_rate=0.05):
        """Suggest loan terms based on credit score"""
        scores = self.scorecard.predict_score(X)
        
        # Risk-based pricing
        terms = []
        for score in scores:
            if score >= 750:
                rate = base_rate * 0.8  # 20% discount
                max_ltv = 0.85
            elif score >= 700:
                rate = base_rate
                max_ltv = 0.80
            elif score >= 650:
                rate = base_rate * 1.2  # 20% premium
                max_ltv = 0.75
            elif score >= 600:
                rate = base_rate * 1.5  # 50% premium
                max_ltv = 0.70
            else:
                rate = None  # Not eligible
                max_ltv = None
            
            terms.append({
                'score': score,
                'interest_rate': rate,
                'max_ltv': max_ltv
            })
        
        return pd.DataFrame(terms)

# ============================================================================
# 5. CREDIT SCORE MONITOR
# ============================================================================

class CreditScoreMonitor:
    """
    Monitor credit scoring model performance over time
    """
    def __init__(self, scorecard, reference_scores=None):
        self.scorecard = scorecard
        self.reference_scores = reference_scores
        self.performance_history = []
        
    def calculate_psi(self, current_scores, reference_scores=None, bins=10):
        """Calculate Population Stability Index"""
        if reference_scores is None:
            reference_scores = self.reference_scores
        
        if reference_scores is None:
            raise ValueError("Reference scores not provided")
        
        # Create bins
        score_min = min(reference_scores.min(), current_scores.min())
        score_max = max(reference_scores.max(), current_scores.max())
        bins_edges = np.linspace(score_min, score_max, bins + 1)
        
        # Calculate distributions
        ref_dist, _ = np.histogram(reference_scores, bins=bins_edges)
        curr_dist, _ = np.histogram(current_scores, bins=bins_edges)
        
        # Convert to percentages
        ref_pct = ref_dist / len(reference_scores)
        curr_pct = curr_dist / len(current_scores)
        
        # Calculate PSI (add small epsilon to avoid division by zero)
        epsilon = 1e-10
        psi = np.sum((ref_pct - curr_pct) * np.log((ref_pct + epsilon) / (curr_pct + epsilon)))
        
        return psi
    
    def monitor_performance(self, X, y, period_name):
        """Monitor model performance for a period"""
        scores = self.scorecard.predict_score(X)
        proba = self.scorecard.predict_proba(X)[:, 1]
        
        # Calculate metrics
        auc = roc_auc_score(y, proba)
        
        # Calculate PSI if reference scores available
        if self.reference_scores is not None:
            psi = self.calculate_psi(scores)
        else:
            psi = None
            self.reference_scores = scores
        
        # Store performance
        performance = {
            'period': period_name,
            'auc': auc,
            'psi': psi,
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'default_rate': y.mean(),
            'sample_size': len(y)
        }
        
        self.performance_history.append(performance)
        
        return performance
    
    def generate_report(self):
        """Generate monitoring report"""
        report_df = pd.DataFrame(self.performance_history)
        
        print("Credit Score Model Monitoring Report")
        print("=" * 50)
        print(report_df.to_string(index=False))
        
        # Plot trends
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # AUC trend
        axes[0, 0].plot(report_df['period'], report_df['auc'], marker='o')
        axes[0, 0].set_title('AUC-ROC Trend')
        axes[0, 0].set_ylabel('AUC')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # PSI trend
        if report_df['psi'].notna().any():
            axes[0, 1].plot(report_df['period'], report_df['psi'], marker='o', color='red')
            axes[0, 1].axhline(y=0.1, color='green', linestyle='--', label='Stable')
            axes[0, 1].axhline(y=0.25, color='orange', linestyle='--', label='Warning')
            axes[0, 1].set_title('PSI Trend')
            axes[0, 1].set_ylabel('PSI')
            axes[0, 1].legend()
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Mean score trend
        axes[1, 0].plot(report_df['period'], report_df['mean_score'], marker='o', color='green')
        axes[1, 0].fill_between(report_df['period'], 
                                report_df['mean_score'] - report_df['std_score'],
                                report_df['mean_score'] + report_df['std_score'],
                                alpha=0.3)
        axes[1, 0].set_title('Mean Score Trend (+/- 1 std)')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Default rate trend
        axes[1, 1].plot(report_df['period'], report_df['default_rate'], marker='o', color='purple')
        axes[1, 1].set_title('Default Rate Trend')
        axes[1, 1].set_ylabel('Default Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return report_df

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("Credit Scoring System in Python")
    print("=" * 50)
    
    # Create sample data
    print("\n1. Creating sample credit data...")
    df = create_sample_credit_data(10000)
    
    # Engineer features
    print("\n2. Engineering features...")
    df = engineer_features(df)
    
    # Prepare data
    print("\n3. Preparing data for modeling...")
    X, y = prepare_data_for_modeling(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train scorecard
    print("\n4. Training credit scorecard...")
    scorecard = CreditScorecard(target_points=600, pdo=20, odds=50)
    scorecard.fit(X_train, y_train)
    
    # Get scorecard points
    points_df, base_points = scorecard.get_scorecard_points()
    print(f"\nBase points: {base_points:.2f}")
    print("Top 5 most influential features:")
    print(points_df.nlargest(5, 'points_per_std')[['feature', 'coefficient', 'points_per_std']])
    
    # Evaluate model
    print("\n5. Evaluating model performance...")
    test_proba = scorecard.predict_proba(X_test)
    test_scores = scorecard.predict_score(X_test)
    auc = evaluate_model(y_test, test_proba, test_scores)
    
    # Make decisions
    print("\n6. Making credit decisions...")
    decision_engine = CreditDecisionEngine(scorecard)
    decisions = decision_engine.make_decision(X_test)
    print("\nDecision distribution:")
    print(decisions['decision'].value_counts())
    
    # Suggest loan terms
    print("\n7. Suggesting loan terms...")
    terms = decision_engine.suggest_loan_terms(X_test)
    decisions_df = decision_engine.make_decision(X_test)
    merged_df = pd.concat([decisions_df, terms], axis=1)
    print("\nAverage interest rate by decision:")
    print(merged_df.groupby('decision')['interest_rate'].mean())
    
    # Monitor model
    print("\n8. Monitoring model performance...")
    monitor = CreditScoreMonitor(scorecard)
    
    # Simulate monitoring
    for i, period in enumerate(['2023-Q1', '2023-Q2', '2023-Q3', '2023-Q4']):
        start_idx = i * len(X_test) // 4
        end_idx = (i + 1) * len(X_test) // 4
        X_period = X_test.iloc[start_idx:end_idx]
        y_period = y_test.iloc[start_idx:end_idx]
        monitor.monitor_performance(X_period, y_period, period)
    
    # Generate report
    print("\n9. Generating monitoring report...")
    monitor.generate_report()
    
    print("\nCredit scoring system completed successfully!")

if __name__ == "__main__":
    main()