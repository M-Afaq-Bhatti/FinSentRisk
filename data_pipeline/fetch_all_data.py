import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Make directories if they don't exist
os.makedirs("datasets", exist_ok=True)

def create_complete_financial_dataset():
    """
    Create a comprehensive time-series dataset for financial risk modeling
    This version guarantees data output for model training
    """
    print("🔄 Creating comprehensive financial time-series dataset...")
    
    # Define companies and time range
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    # Create quarterly dates from 2000 to 2022 (92 quarters)
    start_date = "2000-01-01"
    end_date = "2022-12-31"
    quarters = pd.date_range(start=start_date, end=end_date, freq='QE')
    
    print(f"📅 Generating data for {len(quarters)} quarters from {quarters[0].date()} to {quarters[-1].date()}")
    
    # Generate comprehensive dataset
    all_data = []
    
    for ticker in tickers:
        print(f"📊 Creating data for {ticker}...")
        
        # Set consistent seed for each ticker to get realistic patterns
        np.random.seed(hash(ticker) % 2147483647)
        
        # Base characteristics for each company (realistic ranges)
        base_configs = {
            "AAPL": {"pe_base": 25, "roe_base": 0.20, "debt_ratio": 0.3},
            "MSFT": {"pe_base": 30, "roe_base": 0.18, "debt_ratio": 0.2}, 
            "GOOGL": {"pe_base": 22, "roe_base": 0.16, "debt_ratio": 0.1}
        }
        
        config = base_configs.get(ticker, {"pe_base": 20, "roe_base": 0.15, "debt_ratio": 0.5})
        
        for i, quarter in enumerate(quarters):
            # Create realistic time-series patterns
            
            # Market cycle (7-year cycles)
            market_cycle = np.sin(2 * np.pi * i / 28) * 0.3
            
            # Seasonal effects (quarterly)
            seasonal = np.sin(2 * np.pi * i / 4) * 0.1
            
            # Long-term trend
            trend = 0.02 * (i / len(quarters))
            
            # Random shock (financial crises, etc.)
            shock = 0
            if i in range(28, 32):  # 2008 crisis period
                shock = -0.4
            elif i in range(80, 84):  # 2020 covid period  
                shock = -0.2
            
            # Economic indicator base values with realistic patterns
            base_gdp = 2.5 + trend * 100 + market_cycle * 2 + seasonal * 0.5 + shock * 5
            base_inflation = 2.0 + trend * 50 + np.random.normal(0, 0.5)
            base_interest = max(0, 2.0 + market_cycle * 3 + shock * 2 + np.random.normal(0, 0.3))
            base_unemployment = max(3, 5.0 - market_cycle * 2 + abs(shock) * 8 + np.random.normal(0, 0.5))
            
            # Financial ratios with realistic correlations
            noise_factor = np.random.normal(1, 0.1)
            
            pe_ratio = max(5, config["pe_base"] * (1 + market_cycle + seasonal + shock) * noise_factor)
            current_ratio = max(0.5, 1.5 + seasonal * 0.3 + np.random.normal(0, 0.2))
            debt_to_equity = max(0.05, config["debt_ratio"] * (1 - market_cycle * 0.5) + np.random.normal(0, 0.1))
            roe = max(0.01, config["roe_base"] * (1 + market_cycle + trend) * noise_factor)
            roa = max(0.005, roe * 0.5 * (1 + np.random.normal(0, 0.1)))
            
            # Sentiment score (correlated with market conditions)
            sentiment_score = max(0, min(1, 0.5 + market_cycle * 0.3 + seasonal * 0.1 - abs(shock) * 0.8 + np.random.normal(0, 0.1)))
            
            # Market cap (growing over time with cycles)
            market_cap_base = {"AAPL": 500e9, "MSFT": 400e9, "GOOGL": 300e9}
            market_cap = market_cap_base.get(ticker, 200e9) * (1 + trend * 5) * (1 + market_cycle * 0.5) * noise_factor
            
            # Revenue (correlated with market cap)
            revenue = market_cap * 0.1 * (1 + np.random.normal(0, 0.2))
            net_income = revenue * (roe * 2) * (1 + np.random.normal(0, 0.3))
            
            # Calculate Altman Z-Score
            wc_ta = current_ratio * 0.3  # Working capital / Total assets proxy
            re_ta = roe * 0.5  # Retained earnings / Total assets proxy
            ebit_ta = roa * 1.5  # EBIT / Total assets proxy
            mve_tl = market_cap / (market_cap * debt_to_equity)  # Market value equity / Total liabilities
            sales_ta = 0.8  # Sales / Total assets proxy
            
            altman_z = 1.2*wc_ta + 1.4*re_ta + 3.3*ebit_ta + 0.6*mve_tl + 1.0*sales_ta
            
            # Create final data record
            record = {
                'Date': quarter.strftime('%Y-%m-%d'),
                'Quarter': quarter,
                'Ticker': ticker,
                'PE_Ratio': round(pe_ratio, 2),
                'PB_Ratio': round(np.random.uniform(2, 8), 2),
                'Debt_to_Equity': round(debt_to_equity, 3),
                'Current_Ratio': round(current_ratio, 3),
                'ROE': round(roe, 4),
                'ROA': round(roa, 4),
                'Market_Cap': int(market_cap),
                'Total_Revenue': int(revenue),
                'Net_Income': int(net_income),
                'Altman_Z': round(altman_z, 2),
                
                # Macroeconomic indicators
                'GDP_Growth': round(base_gdp, 2),
                'CPI_Inflation': round(base_inflation, 2), 
                'Fed_Funds_Rate': round(base_interest, 2),
                'Unemployment_Rate': round(base_unemployment, 1),
                
                # Sentiment indicators
                'Sentiment_Score': round(sentiment_score, 3),
                'Sentiment_Volatility': round(abs(np.random.normal(0.3, 0.1)), 3),
                
                # Target variables
                'Risk_Score': round(altman_z, 2),
                'Financial_Distress': 1 if altman_z < 1.8 else 0,
                'Risk_Category': 'High' if altman_z < 1.8 else 'Medium' if altman_z < 2.99 else 'Low'
            }
            
            all_data.append(record)
    
    # Create final DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"✅ Dataset created successfully!")
    print(f"📊 Shape: {df.shape}")
    print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"🏢 Companies: {df['Ticker'].unique()}")
    print(f"📈 Total quarters per company: {len(df) // len(tickers)}")
    
    # Save the dataset
    output_file = "datasets/financial_risk_timeseries.csv"
    df.to_csv(output_file, index=False)
    print(f"💾 Saved to: {output_file}")
    
    # Print sample data
    print("\n📋 Sample data (first 5 rows):")
    print(df[['Date', 'Ticker', 'PE_Ratio', 'ROE', 'Sentiment_Score', 'Risk_Score', 'Financial_Distress']].head())
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   Risk Distribution:")
    risk_dist = df['Risk_Category'].value_counts()
    for category, count in risk_dist.items():
        print(f"     {category}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n   Financial Distress Distribution:")
    distress_dist = df['Financial_Distress'].value_counts()
    for status, count in distress_dist.items():
        label = "Distressed" if status == 1 else "Healthy"
        print(f"     {label}: {count} ({count/len(df)*100:.1f}%)")
    
    return df

def create_additional_datasets(main_df):
    """Create additional datasets for different model approaches"""
    
    print("\n🔄 Creating additional datasets...")
    
    # 1. Company-specific datasets for individual ARIMAX models
    for ticker in main_df['Ticker'].unique():
        company_data = main_df[main_df['Ticker'] == ticker].copy()
        company_data = company_data.sort_values('Quarter').reset_index(drop=True)
        
        filename = f"datasets/{ticker}_timeseries.csv"
        company_data.to_csv(filename, index=False)
        print(f"   📁 {ticker}: {filename} ({len(company_data)} quarters)")
    
    # 2. Cross-sectional dataset (latest quarter for each company)
    latest_data = main_df.groupby('Ticker').last().reset_index()
    latest_data.to_csv("datasets/financial_risk_crosssectional.csv", index=False)
    print(f"   📁 Cross-sectional: datasets/financial_risk_crosssectional.csv ({len(latest_data)} companies)")
    
    # 3. Aggregated monthly sentiment data
    sentiment_data = main_df.groupby('Quarter').agg({
        'Sentiment_Score': 'mean',
        'Sentiment_Volatility': 'mean'
    }).reset_index()
    sentiment_data['Date'] = sentiment_data['Quarter']
    sentiment_data.to_csv("datasets/sentiment_timeseries.csv", index=False)
    print(f"   📁 Sentiment: datasets/sentiment_timeseries.csv ({len(sentiment_data)} quarters)")
    
    # 4. Macro indicators dataset
    macro_data = main_df[['Quarter', 'GDP_Growth', 'CPI_Inflation', 'Fed_Funds_Rate', 'Unemployment_Rate']].drop_duplicates()
    macro_data['Date'] = macro_data['Quarter'] 
    macro_data.to_csv("datasets/macro_indicators.csv", index=False)
    print(f"   📁 Macro: datasets/macro_indicators.csv ({len(macro_data)} quarters)")
    
    print("✅ All additional datasets created!")

def main():
    """Main execution function"""
    print("🚀 Financial Risk Time-Series Dataset Generator")
    print("=" * 60)
    
    # Create main dataset
    main_dataset = create_complete_financial_dataset()
    
    # Create additional datasets
    create_additional_datasets(main_dataset)
    
    print("\n" + "=" * 60)
    print("🎯 DATASET CREATION COMPLETED SUCCESSFULLY!")
    print("\nFiles created in datasets/ folder:")
    print("  📊 financial_risk_timeseries.csv (Main dataset - 276 records)")
    print("  📊 AAPL_timeseries.csv (AAPL only - 92 quarters)")
    print("  📊 MSFT_timeseries.csv (MSFT only - 92 quarters)")
    print("  📊 GOOGL_timeseries.csv (GOOGL only - 92 quarters)")
    print("  📊 financial_risk_crosssectional.csv (Latest data)")
    print("  📊 sentiment_timeseries.csv (Sentiment data)")
    print("  📊 macro_indicators.csv (Economic indicators)")
    
    print("\n🔬 Ready for model training:")
    print("  ✅ ARIMAX: Use individual company files or main dataset")
    print("  ✅ LSTM: Use main dataset with sequence windows")
    print("  ✅ Cross-sectional models: Use crosssectional dataset")
    
    return main_dataset

# Run the data generation
if __name__ == "__main__":
    final_dataset = main()