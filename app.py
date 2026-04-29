import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load pre-segmented dataset
st.set_page_config(layout="wide")
df = pd.read_excel("Segmented_Customers.xlsx")

# Create Loyalty Buckets (if not already there)
if 'Loyalty_Bucket' not in df.columns:
    loyalty_bins = [0, 25, 50, 75, 100]
    loyalty_labels = ['0-25', '25-50', '50-75', '75-100']
    df['Loyalty_Bucket'] = pd.cut(df['Loyalty_Score'], bins=loyalty_bins, labels=loyalty_labels, include_lowest=True)

# Sidebar navigation
st.sidebar.title("Navigation")
options = [
    "Overview",
    "Churn Funnel",
    "CLV Prediction",
    "Churn Prediction",
    "Customer Segmentation",
    "Boxplot: CLV by Segment",
    "Churn Rate by Loyalty Buckets",
    "Average Churn by Loyalty Group",
    "Heat Burn Chart"
]
choice = st.sidebar.radio("Go to", options)

if choice == "Overview":
    st.title("Customer Analytics Dashboard")
    st.write("This dashboard uses pre-computed churn, CLV, and segments from the uploaded dataset.")
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

elif choice == "Churn Funnel":
    st.title("Churn Funnel: Risk Segmentation")
    total_customers = df.shape[0]
    churn_25 = df[df['Churn'] > 0.25].shape[0]
    churn_50 = df[df['Churn'] > 0.5].shape[0]
    churn_75 = df[df['Churn'] > 0.75].shape[0]

    funnel_data = {
        "Segment": ["Total Customers", "Churn > 0.25", "Churn > 0.5", "Churn > 0.75"],
        "Count": [total_customers, churn_25, churn_50, churn_75],
    }
    funnel_df = pd.DataFrame(funnel_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(funnel_df)):
        width = funnel_df['Count'][i]
        label = funnel_df['Segment'][i]
        percent = int((width / funnel_df['Count'][0]) * 100)
        ax.barh(y=i, width=width, height=0.6, color='black')
        ax.text(width / 2, i, f"{width}\n{percent}%", va='center', ha='center', color='white')
        ax.text(-100, i, label, va='center', ha='right')
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlim(0, funnel_df['Count'][0] * 1.1)
    ax.set_title("Churn Funnel: Risk Segmentation", fontsize=16)
    ax.axis('off')
    st.pyplot(fig)

elif choice == "CLV Prediction":
    st.title("CLV Prediction")
    st.write(df[['Age', 'Purchase_Amount', 'Rating', 'Loyalty_Score', 'Customer_Lifetime_Value']].head())

elif choice == "Churn Prediction":
    st.title("📊 Churn Prediction")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    rating = st.slider("Rating (1.0 - 5.0)", 1.0, 5.0, 3.5)
    loyalty = st.slider("Loyalty Score", 0, 100, 50)

    if st.button("⚡ Predict Now"):
        # Dummy prediction logic (replace with model.predict)
        churn_prob = round(0.6 - (loyalty / 200) + (5 - rating) / 10, 4)
        churn_prob = np.clip(churn_prob, 0.0, 1.0)
        predicted_clv = round(2000 + (loyalty * 10) - (rating * 100), 2)
        segment = "Segment 1" if loyalty > 75 else "Segment 2" if loyalty > 50 else "Segment 3"

        st.markdown("### 📈 Prediction Results")
        st.metric("Predicted CLV", f"${predicted_clv:,.2f}")
        st.metric("Churn Probability", f"{churn_prob*100:.2f}%")
        st.metric("Customer Segment", segment)

        # Pie Chart
        fig, ax = plt.subplots()
        labels = ['Churn', 'No Churn']
        values = [churn_prob, 1 - churn_prob]
        colors = ['#4da6ff', '#006bb3']
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')
        st.subheader("Churn Breakdown")
        st.pyplot(fig)

elif choice == "Customer Segmentation":
    st.title("Customer Segmentation")
    st.write(df[['Age', 'Purchase_Amount', 'Customer_Lifetime_Value', 'Customer_Segment']].head())

elif choice == "Boxplot: CLV by Segment":
    st.title("Boxplot of CLV by Segment")
    fig, ax = plt.subplots()
    sns.boxplot(x='Customer_Segment', y='Customer_Lifetime_Value', data=df, ax=ax)
    ax.set_title("CLV Distribution by Customer Segment")
    st.pyplot(fig)

elif choice == "Churn Rate by Loyalty Buckets":
    st.title("Churn Rate by Loyalty Buckets")
    churn_by_loyalty = df.groupby('Loyalty_Bucket')['Churn'].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x='Loyalty_Bucket', y='Churn', data=churn_by_loyalty, ax=ax)
    ax.set_title("Churn Rate by Loyalty Buckets")
    st.pyplot(fig)

elif choice == "Average Churn by Loyalty Group":
    st.title("Average Churn Probability by Loyalty Group")
    loyalty_grouped = df.groupby('Loyalty_Bucket')['Churn'].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x='Loyalty_Bucket', y='Churn', data=loyalty_grouped, ax=ax, palette='coolwarm')
    ax.set_title("Average Churn Probability by Loyalty Group")
    st.pyplot(fig)

elif choice == "Heat Burn Chart":
    st.title("Customer Heat Burn Chart")
    burn_matrix = pd.pivot_table(
        df, values='Customer_Lifetime_Value',
        index='Customer_Segment', columns='Loyalty_Bucket',
        aggfunc='mean')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(burn_matrix, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
    ax.set_title("Heat Burn: Avg CLV by Segment and Loyalty")
    st.pyplot(fig)
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa;
        text-align: center;
        color: #888;
        font-size: 13px;
        padding: 5px 0;
        z-index: 100;
    }
    </style>
    <div class="footer">© 2025 | Developed by Rajesh</div>
    """,
    unsafe_allow_html=True
)
