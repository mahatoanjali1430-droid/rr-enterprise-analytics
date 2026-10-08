import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(
    page_title="RR Enterprise Analytics",
    page_icon="⚙️",
    layout="wide"
)

# Styling setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Title & Header
st.title("⚙️ RR Enterprise - Business Analytics Suite")
st.markdown("**Course Code:** MGNM801 | **Client:** RR Enterprise (Gear Parts & Equipment)")

# Load Data Function with Caching
@st.cache_data
def load_data():
    sales_df = pd.read_csv("rr_enterprise_sales.csv")
    inventory_df = pd.read_csv("rr_enterprise_inventory.csv")
    sales_df['Invoice_Date'] = pd.to_datetime(sales_df['Invoice_Date'])
    return sales_df, inventory_df

try:
    sales_df, inventory_df = load_data()
except Exception as e:
    st.error("Please ensure 'rr_enterprise_sales.csv' and 'rr_enterprise_inventory.csv' exist in the root folder.")
    st.stop()

# Sidebar Navigation
st.sidebar.header("Navigation")
service_choice = st.sidebar.radio(
    "Select Service Module:",
    [
        "Overview & Sales Dashboard",
        "Service 1: Customer Retention (Churn Model)",
        "Service 2: Inventory & Dead-Stock Optimization"
    ]
)

# ==============================================================================
# SERVICE MODULE 3: OVERVIEW & AUTOMATED SALES DASHBOARD
# ==============================================================================
if service_choice == "Overview & Sales Dashboard":
    st.header("📊 Service 3: Automated Sales & Revenue Dashboard")
    
    total_revenue = sales_df['Total_Amount'].sum()
    total_orders = sales_df['Invoice_No'].nunique()
    avg_order = sales_df['Total_Amount'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"₹{total_revenue:,.2f}")
    col2.metric("Total Invoices", f"{total_orders:,}")
    col3.metric("Avg Order Value", f"₹{avg_order:,.2f}")
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Revenue by Equipment Category")
        fig, ax = plt.subplots(figsize=(6, 4))
        cat_rev = sales_df.groupby('Category')['Total_Amount'].sum().sort_values()
        cat_rev.plot(kind='barh', ax=ax, color='#2b5c8f')
        ax.set_xlabel("Revenue (₹)")
        st.pyplot(fig)
        
    with col_right:
        st.subheader("Monthly Sales Trend")
        sales_df['Month'] = sales_df['Invoice_Date'].dt.to_period('M').astype(str)
        monthly_rev = sales_df.groupby('Month')['Total_Amount'].sum()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        monthly_rev.plot(kind='line', marker='o', ax=ax2, color='#2b5c8f')
        ax2.set_ylabel("Revenue (₹)")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

# ==============================================================================
# SERVICE MODULE 1: CUSTOMER RETENTION & AT-RISK CHURN MODEL
# ==============================================================================
elif service_choice == "Service 1: Customer Retention (Churn Model)":
    st.header("🎯 Service 1: Customer Retention & At-Risk Churn Prediction")
    
    ref_dt = pd.to_datetime('2026-10-08')
    
    rfm = sales_df.groupby('Customer_ID').agg({
        'Invoice_Date': lambda x: (ref_dt - x.max()).days,
        'Invoice_No': 'nunique',
        'Total_Amount': 'sum',
        'Quantity_Sold': 'sum'
    }).reset_index()
    
    rfm.columns = ['Customer_ID', 'Recency_Days', 'Frequency', 'Monetary_Value', 'Total_Units']
    rfm['Avg_Order_Value'] = rfm['Monetary_Value'] / rfm['Frequency']
    
    # Threshold slider for interactive churn analysis
    recency_threshold = st.slider("Select At-Risk Inactivity Threshold (Days):", 30, 90, 60)
    rfm['Is_Churned'] = np.where(rfm['Recency_Days'] > recency_threshold, 1, 0)
    
    churn_count = rfm['Is_Churned'].sum()
    st.warning(f"**At-Risk Clients:** {churn_count} out of {len(rfm)} clients haven't ordered in >{recency_threshold} days.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Retention vs. At-Risk Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = rfm['Is_Churned'].value_counts()
        ax.bar(['Active', 'At-Risk'], [counts.get(0, 0), counts.get(1, 0)], color=['#2b5c8f', '#d9534f'])
        ax.set_ylabel("Number of Clients")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Recency vs Total Spend")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors = rfm['Is_Churned'].map({0: '#2b5c8f', 1: '#d9534f'})
        ax2.scatter(rfm['Recency_Days'], rfm['Monetary_Value'], c=colors, alpha=0.7)
        ax2.set_xlabel("Days Since Last Purchase")
        ax2.set_ylabel("Total Spend (₹)")
        st.pyplot(fig2)
        
    st.subheader("📋 Actionable At-Risk Client List")
    st.dataframe(rfm[rfm['Is_Churned'] == 1][['Customer_ID', 'Recency_Days', 'Frequency', 'Monetary_Value']])

# ==============================================================================
# SERVICE MODULE 2: INVENTORY & DEAD-STOCK OPTIMIZATION
# ==============================================================================
elif service_choice == "Service 2: Inventory & Dead-Stock Optimization":
    st.header("📦 Service 2: Inventory & Dead-Stock Optimization")
    
    inv = inventory_df.copy()
    inv['Total_Stock_Value'] = inv['Stock_Quantity'] * inv['Purchase_Cost']
    
    sales_qty = sales_df.groupby('SKU')['Quantity_Sold'].sum().reset_index()
    inv = inv.merge(sales_qty, on='SKU', how='left').fillna({'Quantity_Sold': 0})
    
    inv = inv.sort_values(by='Total_Stock_Value', ascending=False)
    inv['Cumulative_Value'] = inv['Total_Stock_Value'].cumsum()
    total_val = inv['Total_Stock_Value'].sum()
    inv['Cumulative_Perc'] = (inv['Cumulative_Value'] / total_val) * 100
    
    inv['ABC_Category'] = np.where(inv['Cumulative_Perc'] <= 70, 'Class A (High Value)',
                           np.where(inv['Cumulative_Perc'] <= 90, 'Class B (Medium)', 'Class C (Low/Dead Stock)'))
    
    st.info(f"**Total Capital Tied Up in Inventory:** ₹{total_val:,.2f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ABC Classification (Capital Value)")
        fig, ax = plt.subplots(figsize=(6, 4))
        abc_sum = inv.groupby('ABC_Category')['Total_Stock_Value'].sum() / 1000
        abc_sum.plot(kind='bar', ax=ax, color=['#2b5c8f', '#f0ad4e', '#d9534f'])
        ax.set_ylabel("Value (₹ Thousands)")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Reorder Threshold vs. Stock Level")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.boxplot([inv['Stock_Quantity'], inv['Reorder_Level']], labels=['Current Stock', 'Reorder Level'], patch_artist=True)
        ax2.set_ylabel("Units")
        st.pyplot(fig2)
        
    st.subheader("🚨 Stock Reorder & Clearance Audit")
    reorder_needed = inv[inv['Stock_Quantity'] <= inv['Reorder_Level']]
    st.dataframe(reorder_needed[['SKU', 'Item_Name', 'Category', 'Stock_Quantity', 'Reorder_Level', 'ABC_Category']])
