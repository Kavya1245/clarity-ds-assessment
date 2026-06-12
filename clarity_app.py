import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import json
import urllib.request
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLE
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Clarity TTS — DS Assessment",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d1b3e 0%, #1a3a6e 100%);
}
section[data-testid="stSidebar"] * { color: #e8edf5 !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 15px; padding: 6px 0; }

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1a3a6e, #0d1b3e);
    border-radius: 12px;
    padding: 20px 24px;
    color: white;
    text-align: center;
    margin-bottom: 8px;
    border: 1px solid #2a4a8e;
}
.metric-card .val { font-size: 2.2rem; font-weight: 700; color: #60a5fa; }
.metric-card .lbl { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }

/* ── Section header ── */
.section-header {
    background: linear-gradient(90deg, #0d1b3e, #1a3a6e);
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 20px 0 12px 0;
    border-left: 4px solid #60a5fa;
}

/* ── Urgency badges ── */
.badge-high   { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:20px; font-weight:700; font-size:0.82rem; }
.badge-medium { background:#fef3c7; color:#92400e; padding:3px 10px; border-radius:20px; font-weight:700; font-size:0.82rem; }
.badge-low    { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:20px; font-weight:700; font-size:0.82rem; }

/* ── Complaint summary card ── */
.summary-card {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
    background: #f8fafc;
    border-left: 5px solid #60a5fa;
}
.summary-card.high   { border-left-color: #ef4444; background:#fff8f8; }
.summary-card.medium { border-left-color: #f59e0b; background:#fffdf5; }
.summary-card.low    { border-left-color: #10b981; background:#f5fff9; }

/* ── Prediction result ── */
.pred-high { background:#fee2e2; border:2px solid #ef4444; border-radius:12px; padding:20px; text-align:center; }
.pred-low  { background:#d1fae5; border:2px solid #10b981; border-radius:12px; padding:20px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, 'data', 'clarity_bookings_dataset.csv')
    df = pd.read_csv(csv_path)
    df['booking_date']   = pd.to_datetime(df['booking_date'])
    df['departure_date'] = pd.to_datetime(df['departure_date'])
    df['ancillary_revenue_inr'] = df['ancillary_revenue_inr'].fillna(0)
    df['payment_method'] = df['payment_method'].fillna(df['payment_method'].mode()[0])
    df['fare_basis']     = df['fare_basis'].fillna('Unknown')
    df['is_cancelled']   = df['booking_status'].isin(['Cancelled','Refunded']).astype(int)
    df['booking_month']  = df['booking_date'].dt.month
    df['is_repeat']      = (df['prior_bookings'] > 0).astype(int)
    df['fare_per_pax']   = df['total_fare_inr'] / df['pax_count']
    df['route']          = df['origin'] + ' → ' + df['destination']
    return df

df = load_data()

# ══════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ✈️ Clarity TTS")
    st.markdown("**Data Science Assessment**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠  Home & Overview",
            "📊  EDA Dashboard",
            "📈  Revenue Analysis",
            "🚫  Cancellation Patterns",
            "👥  Customer Behaviour",
            "🤖  Cancellation Predictor",
            "💬  Complaint Summariser",
            "📋  Priority Action Board",
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Dataset:** <code style='background:#2a4a8e;color:#93c5fd;padding:2px 6px;border-radius:4px;font-size:0.82rem'>clarity_bookings_dataset.csv</code>", unsafe_allow_html=True)
    st.markdown(f"**Records:** {len(df):,}")
    st.markdown(f"**Columns:** {df.shape[1]}")
    st.markdown("**Period:** Jan–Dec 2025")

# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME & OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠  Home & Overview":
    st.title("✈️ Clarity Travel Technology Solutions")
    st.subheader("Data Science Assessment — Interactive Dashboard")
    st.markdown("""
    Welcome to the **Clarity TTS DS Assessment App**. This Streamlit application brings together
    all three parts of the assessment — EDA, Predictive Modelling, and NLP/GenAI — into one
    interactive interface for the Clarity operations and data teams.
    """)

    st.markdown("---")
    st.markdown("### 📦 Dataset at a Glance")

    total_rev = df['total_fare_inr'].sum()
    cancel_rate = df['is_cancelled'].mean() * 100
    avg_fare = df['total_fare_inr'].mean()
    complaint_count = df['customer_complaint'].notna().sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="val">2,000</div>
            <div class="lbl">Total Bookings</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="val">₹{total_rev/1e7:.1f}Cr</div>
            <div class="lbl">Total Revenue</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{cancel_rate:.1f}%</div>
            <div class="lbl">Cancellation Rate</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{complaint_count}</div>
            <div class="lbl">Complaints Logged</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗺️ App Navigation Guide")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**📊 EDA Dashboard**
Monthly booking trends, channel mix, and booking status breakdown.

**📈 Revenue Analysis**
Top airlines, cabin class revenue, NDC vs GDS comparison, top routes.

**🚫 Cancellation Patterns**
Cancellation rates by cabin, channel, lead time and fare buckets.

**👥 Customer Behaviour**
New vs repeat customer comparison on fare, cancellations, satisfaction.
        """)
    with col2:
        st.markdown("""
**🤖 Cancellation Predictor**
Enter booking details and get a real-time cancellation risk prediction.

**💬 Complaint Summariser**
AI-powered (Groq/demo) structured summaries for support agents.

**📋 Priority Action Board**
Full urgency-ranked complaint dashboard for the support team lead.
        """)

    st.markdown("---")
    st.markdown("### 🔑 Key Findings Summary")
    st.info("""
**Revenue:** Air France leads at ₹3.1Cr total revenue. Business class generates the highest total (38%) while First class commands the highest average fare (₹4.2L). GDS accounts for 68% of bookings.

**Cancellations:** Overall rate is 22.6%. Premium Economy cancels most (24.3%). Cancelled bookings have longer average lead times (69.7 days vs 58.8 days), suggesting early planners change their mind.

**Complaints:** 248 complaints across 7 categories. Ticketing Issues (25.8%) and Refund Issues (19.8%) dominate. Schedule Change complaints correlate with the highest cancellation rate (58%).
    """)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — EDA DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "📊  EDA Dashboard":
    st.title("📊 Exploratory Data Analysis Dashboard")
    st.markdown("Monthly trends, booking channel mix, and status breakdown across the 2025 booking period.")

    # Monthly bookings
    st.markdown('<div class="section-header">📅 Monthly Booking Volume & Revenue</div>', unsafe_allow_html=True)
    monthly = df.groupby('booking_month').agg(
        bookings=('booking_id','count'),
        revenue=('total_fare_inr','sum')
    ).reset_index()
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthly['month_name'] = monthly['booking_month'].apply(lambda x: month_names[x-1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].bar(monthly['month_name'], monthly['bookings'], color='#3b82f6', edgecolor='white')
    axes[0].set_title('Monthly Booking Volume', fontweight='bold')
    axes[0].set_ylabel('Number of Bookings')
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(monthly['month_name'], monthly['revenue']/1e6, color='#10b981', edgecolor='white')
    axes[1].set_title('Monthly Revenue (₹ Millions)', fontweight='bold')
    axes[1].set_ylabel('Revenue (₹M)')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Booking channel mix
    st.markdown('<div class="section-header">📡 Booking Channel Mix Over the Year</div>', unsafe_allow_html=True)
    channel_monthly = df.groupby(['booking_month','booking_channel']).size().unstack(fill_value=0)
    channel_monthly.index = [month_names[i-1] for i in channel_monthly.index]

    fig, ax = plt.subplots(figsize=(14, 4))
    channel_monthly.plot(kind='bar', stacked=True, ax=ax,
        color=['#3b82f6','#10b981','#f59e0b','#ef4444'], edgecolor='white')
    ax.set_title('Booking Channel Mix by Month', fontweight='bold')
    ax.set_ylabel('Number of Bookings')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Booking status
    st.markdown('<div class="section-header">📋 Booking Status Breakdown</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        status_counts = df['booking_status'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#3b82f6','#10b981','#ef4444','#f59e0b','#8b5cf6']
        ax.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=140)
        ax.set_title('Booking Status Distribution', fontweight='bold')
        st.pyplot(fig)
        plt.close()
    with col2:
        st.dataframe(
            status_counts.reset_index().rename(columns={'booking_status':'Status','count':'Count'}),
            use_container_width=True, hide_index=True
        )
        st.markdown(f"""
- **Ticketed:** {status_counts.get('Ticketed',0)} bookings (fully confirmed + issued)
- **Cancelled:** {status_counts.get('Cancelled',0)} bookings
- **Refunded:** {status_counts.get('Refunded',0)} bookings
- **Combined cancel rate: {df['is_cancelled'].mean()*100:.1f}%**
        """)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — REVENUE ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "📈  Revenue Analysis":
    st.title("📈 Revenue Analysis")
    st.markdown("Top airlines, cabin class performance, booking source comparison, and highest-value routes.")

    # Airline revenue
    st.markdown('<div class="section-header">✈️ Top Airlines by Total Revenue</div>', unsafe_allow_html=True)
    airline_rev = df.groupby('airline')['total_fare_inr'].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(airline_rev.index, airline_rev.values/1e6,
                   color='#3b82f6', edgecolor='white')
    ax.set_xlabel('Total Revenue (₹ Millions)')
    ax.set_title('Total Revenue by Airline', fontweight='bold')
    for bar, val in zip(bars, airline_rev.values/1e6):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f'₹{val:.1f}M', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Cabin class
    st.markdown('<div class="section-header">💺 Revenue by Cabin Class</div>', unsafe_allow_html=True)
    cabin = df.groupby('cabin_class').agg(
        total_rev=('total_fare_inr','sum'),
        avg_fare=('total_fare_inr','mean'),
        bookings=('booking_id','count')
    ).reset_index().sort_values('total_rev', ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#1e40af','#3b82f6','#60a5fa','#93c5fd']
        ax.pie(cabin['total_rev'], labels=cabin['cabin_class'], autopct='%1.1f%%',
               colors=colors, startangle=140)
        ax.set_title('Revenue Share by Cabin', fontweight='bold')
        st.pyplot(fig)
        plt.close()
    with c2:
        cabin_display = cabin.copy()
        cabin_display['total_rev'] = cabin_display['total_rev'].apply(lambda x: f'₹{x/1e6:.1f}M')
        cabin_display['avg_fare']  = cabin_display['avg_fare'].apply(lambda x: f'₹{x:,.0f}')
        cabin_display.columns = ['Cabin','Total Revenue','Avg Fare','Bookings']
        st.dataframe(cabin_display, use_container_width=True, hide_index=True)

    # NDC vs GDS
    st.markdown('<div class="section-header">🔗 NDC vs GDS Revenue Comparison</div>', unsafe_allow_html=True)
    source = df.groupby('booking_source').agg(
        total_rev=('total_fare_inr','sum'),
        avg_fare=('total_fare_inr','mean'),
        bookings=('booking_id','count'),
        avg_ancillary=('ancillary_revenue_inr','mean')
    ).reset_index()

    c1, c2, c3, c4 = st.columns(4)
    for i, row in source.iterrows():
        label = row['booking_source']
        with [c1,c2,c3,c4][i*2]:
            st.markdown(f"""<div class="metric-card">
                <div class="val">₹{row['total_rev']/1e6:.0f}M</div>
                <div class="lbl">{label} — Total Revenue</div></div>""", unsafe_allow_html=True)
        with [c1,c2,c3,c4][i*2+1]:
            st.markdown(f"""<div class="metric-card">
                <div class="val">{row['bookings']}</div>
                <div class="lbl">{label} — Bookings</div></div>""", unsafe_allow_html=True)

    # Top routes
    st.markdown('<div class="section-header">🗺️ Top 10 Revenue-Generating Routes</div>', unsafe_allow_html=True)
    routes = df.groupby('route')['total_fare_inr'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.barh(routes.index[::-1], routes.values[::-1]/1e6, color='#8b5cf6', edgecolor='white')
    ax.set_xlabel('Total Revenue (₹ Millions)')
    ax.set_title('Top 10 Routes by Revenue', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE 4 — CANCELLATION PATTERNS
# ══════════════════════════════════════════════════════════════
elif page == "🚫  Cancellation Patterns":
    st.title("🚫 Cancellation Patterns")
    st.markdown("Identifying which factors most strongly associate with cancellations.")

    cancel_rate = df['is_cancelled'].mean() * 100
    st.metric("Overall Cancellation Rate", f"{cancel_rate:.1f}%",
              help="Includes both Cancelled and Refunded bookings")

    # By cabin and channel
    st.markdown('<div class="section-header">📊 Cancellation Rate by Cabin Class & Channel</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        cabin_cancel = df.groupby('cabin_class')['is_cancelled'].mean().sort_values(ascending=True) * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#10b981' if v < 22 else '#ef4444' for v in cabin_cancel.values]
        ax.barh(cabin_cancel.index, cabin_cancel.values, color=colors, edgecolor='white')
        ax.axvline(cancel_rate, color='#f59e0b', linestyle='--', linewidth=1.5, label=f'Overall avg ({cancel_rate:.1f}%)')
        ax.set_xlabel('Cancellation Rate (%)')
        ax.set_title('By Cabin Class', fontweight='bold')
        ax.legend(fontsize=8)
        for i, v in enumerate(cabin_cancel.values):
            ax.text(v + 0.1, i, f'{v:.1f}%', va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with c2:
        ch_cancel = df.groupby('booking_channel')['is_cancelled'].mean().sort_values(ascending=True) * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#10b981' if v < 22 else '#ef4444' for v in ch_cancel.values]
        ax.barh(ch_cancel.index, ch_cancel.values, color=colors, edgecolor='white')
        ax.axvline(cancel_rate, color='#f59e0b', linestyle='--', linewidth=1.5, label=f'Overall avg ({cancel_rate:.1f}%)')
        ax.set_xlabel('Cancellation Rate (%)')
        ax.set_title('By Booking Channel', fontweight='bold')
        ax.legend(fontsize=8)
        for i, v in enumerate(ch_cancel.values):
            ax.text(v + 0.1, i, f'{v:.1f}%', va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Lead time
    st.markdown('<div class="section-header">📅 Lead Time vs Cancellation</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    cancelled  = df[df['is_cancelled']==1]['lead_time_days']
    not_cancel = df[df['is_cancelled']==0]['lead_time_days']
    axes[0].hist(not_cancel, bins=40, alpha=0.6, color='#3b82f6', label='Not Cancelled')
    axes[0].hist(cancelled,  bins=40, alpha=0.6, color='#ef4444', label='Cancelled')
    axes[0].axvline(not_cancel.mean(), color='#3b82f6', linestyle='--', linewidth=2)
    axes[0].axvline(cancelled.mean(),  color='#ef4444', linestyle='--', linewidth=2)
    axes[0].set_title('Lead Time Distribution', fontweight='bold')
    axes[0].set_xlabel('Lead Time (days)')
    axes[0].legend()

    df['lead_bucket'] = pd.cut(df['lead_time_days'],
        bins=[0,14,30,60,120,365], labels=['0-14d','15-30d','31-60d','61-120d','120d+'])
    bucket_cancel = df.groupby('lead_bucket', observed=True)['is_cancelled'].mean() * 100
    axes[1].bar(bucket_cancel.index.astype(str), bucket_cancel.values,
                color='#8b5cf6', edgecolor='white')
    axes[1].set_title('Cancel Rate by Lead Time Bucket', fontweight='bold')
    axes[1].set_ylabel('Cancellation Rate (%)')
    axes[1].set_xlabel('Lead Time Bucket')
    for i, v in enumerate(bucket_cancel.values):
        axes[1].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Fare buckets
    st.markdown('<div class="section-header">💰 Fare Amount vs Cancellation</div>', unsafe_allow_html=True)
    df['fare_bucket'] = pd.qcut(df['total_fare_inr'], q=4,
        labels=['Low (<₹30K)','Mid (₹30-70K)','High (₹70-200K)','Premium (>₹200K)'])
    fare_cancel = df.groupby('fare_bucket', observed=True)['is_cancelled'].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(fare_cancel.index.astype(str), fare_cancel.values, color='#f59e0b', edgecolor='white')
    ax.axhline(cancel_rate, color='#ef4444', linestyle='--', linewidth=1.5, label=f'Overall avg')
    ax.set_title('Cancellation Rate by Fare Bucket', fontweight='bold')
    ax.set_ylabel('Cancellation Rate (%)')
    ax.legend()
    for i, v in enumerate(fare_cancel.values):
        ax.text(i, v + 0.2, f'{v:.1f}%', ha='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE 5 — CUSTOMER BEHAVIOUR
# ══════════════════════════════════════════════════════════════
elif page == "👥  Customer Behaviour":
    st.title("👥 Customer Behaviour Analysis")
    st.markdown("Comparing **New customers** (prior_bookings = 0) vs **Repeat customers** (prior_bookings > 0).")

    new_df    = df[df['is_repeat']==0]
    repeat_df = df[df['is_repeat']==1]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{len(new_df)}</div>
            <div class="lbl">New Customers</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{len(repeat_df)}</div>
            <div class="lbl">Repeat Customers</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{new_df['is_cancelled'].mean()*100:.1f}%</div>
            <div class="lbl">New Cancel Rate</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{repeat_df['is_cancelled'].mean()*100:.1f}%</div>
            <div class="lbl">Repeat Cancel Rate</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Side-by-Side Comparison</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Fare
    axes[0].bar(['New','Repeat'],
                [new_df['total_fare_inr'].mean(), repeat_df['total_fare_inr'].mean()],
                color=['#3b82f6','#10b981'], edgecolor='white', width=0.5)
    axes[0].set_title('Average Fare (₹)', fontweight='bold')
    axes[0].set_ylabel('Average Total Fare (₹)')
    for i, v in enumerate([new_df['total_fare_inr'].mean(), repeat_df['total_fare_inr'].mean()]):
        axes[0].text(i, v + 500, f'₹{v:,.0f}', ha='center', fontsize=9)

    # Cancel rate
    axes[1].bar(['New','Repeat'],
                [new_df['is_cancelled'].mean()*100, repeat_df['is_cancelled'].mean()*100],
                color=['#ef4444','#f59e0b'], edgecolor='white', width=0.5)
    axes[1].set_title('Cancellation Rate (%)', fontweight='bold')
    axes[1].set_ylabel('Cancel Rate (%)')
    for i, v in enumerate([new_df['is_cancelled'].mean()*100, repeat_df['is_cancelled'].mean()*100]):
        axes[1].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontsize=10)

    # Satisfaction
    new_sat    = new_df['satisfaction_score'].dropna().mean()
    repeat_sat = repeat_df['satisfaction_score'].dropna().mean()
    axes[2].bar(['New','Repeat'], [new_sat, repeat_sat],
                color=['#8b5cf6','#06b6d4'], edgecolor='white', width=0.5)
    axes[2].set_title('Avg Satisfaction Score', fontweight='bold')
    axes[2].set_ylabel('Score (1–5)')
    axes[2].set_ylim(0, 5)
    for i, v in enumerate([new_sat, repeat_sat]):
        axes[2].text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">🔍 Key Insights</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"""
**Repeat customers are more valuable:**
- Cancel {repeat_df['is_cancelled'].mean()*100:.1f}% vs {new_df['is_cancelled'].mean()*100:.1f}% for new customers
- Higher satisfaction score ({repeat_sat:.2f} vs {new_sat:.2f})
        """)
    with col2:
        st.info(f"""
**New customers book higher fares:**
- Avg fare ₹{new_df['total_fare_inr'].mean():,.0f} vs ₹{repeat_df['total_fare_inr'].mean():,.0f} for repeat
- But cancel more — suggests impulse/exploratory bookings
        """)

# ══════════════════════════════════════════════════════════════
# PAGE 6 — CANCELLATION PREDICTOR
# ══════════════════════════════════════════════════════════════
elif page == "🤖  Cancellation Predictor":
    st.title("🤖 Cancellation Risk Predictor")
    st.markdown("Enter booking details below to get an instant cancellation risk prediction.")

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    @st.cache_resource
    def train_model():
        dft = df.copy()
        dft['fare_per_pax']         = dft['total_fare_inr'] / dft['pax_count']
        dft['tax_ratio']            = dft['taxes_inr'] / (dft['base_fare_inr'] + 1)
        dft['has_ancillary']        = (dft['ancillary_revenue_inr'] > 0).astype(int)
        dft['is_group_booking']     = (dft['pax_count'] >= 3).astype(int)
        dft['is_weekend']           = dft['booking_date'].dt.dayofweek.isin([5,6]).astype(int)
        route_avg = dft.groupby('route')['total_fare_inr'].mean()
        dft['fare_to_route_avg']    = dft.apply(
            lambda r: r['total_fare_inr'] / route_avg.get(r['route'], r['total_fare_inr']), axis=1)

        cat_cols = ['airline','haul_type','cabin_class','trip_type',
                    'booking_channel','booking_source','payment_method']
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            dft[col+'_enc'] = le.fit_transform(dft[col].astype(str))
            encoders[col] = le

        features = ['lead_time_days','pax_count','fare_per_pax','tax_ratio',
                    'is_repeat','has_ancillary','is_group_booking','is_weekend',
                    'fare_to_route_avg','booking_month'] + [c+'_enc' for c in cat_cols]

        X = dft[features]
        y = dft['is_cancelled']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        model = GradientBoostingClassifier(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)
        return model, encoders, features, route_avg

    with st.spinner("Training model..."):
        model, encoders, features, route_avg = train_model()

    st.markdown('<div class="section-header">🔢 Enter Booking Details</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        airline  = st.selectbox("Airline", sorted(df['airline'].unique()))
        cabin    = st.selectbox("Cabin Class", df['cabin_class'].unique())
        trip     = st.selectbox("Trip Type", df['trip_type'].unique())
        haul     = st.selectbox("Haul Type", df['haul_type'].unique())

    with col2:
        channel  = st.selectbox("Booking Channel", df['booking_channel'].unique())
        source   = st.selectbox("Booking Source", df['booking_source'].unique())
        payment  = st.selectbox("Payment Method", df['payment_method'].dropna().unique())
        month    = st.slider("Booking Month", 1, 12, 6)

    with col3:
        lead     = st.slider("Lead Time (days)", 0, 365, 30)
        pax      = st.slider("Passenger Count", 1, 9, 1)
        fare     = st.number_input("Total Fare (₹)", min_value=5000, max_value=2000000,
                                    value=80000, step=5000)
        taxes    = st.number_input("Taxes (₹)", min_value=0, max_value=500000,
                                    value=12000, step=1000)
        ancillary = st.checkbox("Has Ancillary Services")
        repeat    = st.checkbox("Repeat Customer")

    if st.button("🔮 Predict Cancellation Risk", type="primary", use_container_width=True):
        base_fare = fare - taxes
        fare_per_pax = fare / max(pax, 1)
        tax_ratio    = taxes / (base_fare + 1)
        is_group     = int(pax >= 3)
        route_avg_val = fare
        fare_to_route = fare / route_avg_val

        row = {}
        for col_name in ['airline','haul_type','cabin_class','trip_type',
                          'booking_channel','booking_source','payment_method']:
            val_map = {'airline':airline,'haul_type':haul,'cabin_class':cabin,
                       'trip_type':trip,'booking_channel':channel,
                       'booking_source':source,'payment_method':payment}
            le = encoders[col_name]
            val = val_map[col_name]
            if val in le.classes_:
                row[col_name+'_enc'] = le.transform([val])[0]
            else:
                row[col_name+'_enc'] = 0

        row.update({
            'lead_time_days': lead, 'pax_count': pax,
            'fare_per_pax': fare_per_pax, 'tax_ratio': tax_ratio,
            'is_repeat': int(repeat), 'has_ancillary': int(ancillary),
            'is_group_booking': is_group, 'is_weekend': 0,
            'fare_to_route_avg': fare_to_route, 'booking_month': month
        })

        X_pred = pd.DataFrame([row])[features]
        prob = model.predict_proba(X_pred)[0][1]

        st.markdown("---")
        if prob >= 0.40:
            st.markdown(f"""<div class="pred-high">
                <h2>⚠️ HIGH CANCELLATION RISK</h2>
                <h1 style="color:#ef4444;font-size:3rem">{prob*100:.1f}%</h1>
                <p>This booking has a high probability of being cancelled or refunded.<br>
                <b>Recommended action:</b> Flag for proactive outreach, consider requiring a deposit.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="pred-low">
                <h2>✅ LOW CANCELLATION RISK</h2>
                <h1 style="color:#10b981;font-size:3rem">{prob*100:.1f}%</h1>
                <p>This booking has a low probability of cancellation.<br>
                <b>Standard processing applies.</b></p>
            </div>""", unsafe_allow_html=True)

        # Risk factors
        st.markdown("#### 🔍 Key Risk Factors for This Booking")
        factors = []
        if lead > 60:   factors.append(f"⚠️ Long lead time ({lead} days) — early bookings cancel more")
        if pax >= 3:    factors.append(f"⚠️ Group booking ({pax} passengers) — higher cancellation tendency")
        if not repeat:  factors.append("⚠️ New customer — higher cancel rate than repeat customers")
        if cabin in ['Premium Economy']: factors.append("⚠️ Premium Economy — highest cancellation rate by cabin")
        if channel == 'B2C_Website':    factors.append("⚠️ B2C Website — highest cancellation channel")
        if not factors:
            factors.append("✅ No major risk factors detected for this booking")
        for f in factors:
            st.markdown(f"- {f}")

# ══════════════════════════════════════════════════════════════
# PAGE 7 — COMPLAINT SUMMARISER
# ══════════════════════════════════════════════════════════════
elif page == "💬  Complaint Summariser":
    st.title("💬 AI-Powered Complaint Summariser")
    st.markdown("""
    This tool uses **Groq's free LLM API (Llama 3.3 70B)** to generate structured summaries
    of customer complaints — helping support agents triage and respond faster.
    """)

    CATEGORY_MAP = {
        'Refund not processed after 30 days'                    : 'Refund Issues',
        'Duplicate charge on credit card for same booking'      : 'Refund Issues',
        'Fare difference charged incorrectly during rebooking'  : 'Refund Issues',
        'Flight delayed by 3 hours, requesting compensation'    : 'Schedule Change',
        'Schedule change not communicated'                      : 'Schedule Change',
        'Connection time too short, missed connecting flight'   : 'Schedule Change',
        'Baggage not received at destination'                   : 'Baggage',
        'Unable to add extra baggage through portal'            : 'Baggage',
        'PNR not found in airline system after ticketing'       : 'Ticketing Issues',
        'Name spelling error on ticket needs correction'        : 'Ticketing Issues',
        'EMD for ancillary service not reflecting'              : 'Ticketing Issues',
        'Coupon status shows used but passenger never flew'     : 'Ticketing Issues',
        'Meal preference not available onboard'                 : 'Onboard Service',
        'Seat allocation issue, paid for window got middle'     : 'Onboard Service',
        'Infant ticket pricing seems incorrect'                 : 'Pricing Error',
        'Visa rejected but non-refundable ticket purchased'     : 'Policy / Visa',
    }

    DEMO_SUMMARIES = {
        'Refund Issues':    {'summary':'Customers experiencing delays in refund processing — refunds not processed after 30 days and duplicate charges on rebookings, indicating a systemic payment reconciliation issue.','urgency':'HIGH','root_cause':'Payment pipeline delays and billing system errors during rebooking workflows.','recommended_action':'Escalate to finance immediately. Implement refund status notifications at 7, 14, 21 day marks.','agent_script':'I can see your refund concern and am escalating it to our finance team as a priority. You will receive a resolution within 48 hours.'},
        'Schedule Change':  {'summary':'Flight schedule changes not being communicated proactively, resulting in missed connections and compensation requests.','urgency':'HIGH','root_cause':'Schedule change alerts not triggering real-time passenger notifications.','recommended_action':'Enable real-time schedule change push notifications via SMS and email. Review MCT breach logic.','agent_script':'I sincerely apologise for the disruption. I am processing your compensation claim now and will review your connection booking.'},
        'Baggage':          {'summary':'Two distinct issues: passengers not receiving checked baggage at destination (World Tracer cases) and portal failures for extra baggage add-ons.','urgency':'MEDIUM','root_cause':'Baggage tracing required for delayed bags. Portal ancillary service integration failure.','recommended_action':'File World Tracer reports. Raise urgent ticket to tech team for portal baggage add-on bug.','agent_script':'I have filed a priority World Tracer report for your baggage and you will receive updates every 6 hours.'},
        'Ticketing Issues': {'summary':'Multiple ticketing failures: PNR sync errors, name corrections, EMD discrepancies, and coupon status errors — indicating integration reliability issues.','urgency':'HIGH','root_cause':'PNR synchronisation failures between Clarity platform and airline systems. EMD issuance pipeline gaps.','recommended_action':'Implement PNR validation checks post-ticketing. Create automated name correction workflow.','agent_script':'I can see a ticketing discrepancy on your booking. I am contacting the airline directly and will send a corrected itinerary within 2 hours.'},
        'Onboard Service':  {'summary':'Meal preferences and seat allocations not matching paid selections — failure in post-booking SSR synchronisation with airline systems.','urgency':'LOW','root_cause':'Special Service Requests for meals and seats not reliably sent to airlines upon ticket issuance.','recommended_action':'Verify SSR transmission in ticketing API. Implement confirmation loop to validate SSRs within 24 hours.','agent_script':'I apologise for the inconvenience. I am re-sending your meal request and will confirm your window seat allocation now.'},
        'Pricing Error':    {'summary':'Infant fare pricing calculating incorrectly — suggests fare rules configuration issue for infant passenger types.','urgency':'MEDIUM','root_cause':'Infant fare calculation logic not applying correct percentage or missing surcharge rules for specific airlines.','recommended_action':'Audit infant fare rules across all airlines. Run regression test on infant + adult fare combinations.','agent_script':'I apologise for the confusion. I am reviewing the correct fare and will send you an updated quote within 1 hour.'},
        'Policy / Visa':    {'summary':'Customers purchasing non-refundable tickets without adequate visa requirement warnings, leading to financial loss on visa rejection.','urgency':'MEDIUM','root_cause':'No visa requirement warning or fare flexibility recommendation shown at checkout for high-rejection-rate destinations.','recommended_action':'Add visa advisory at checkout. Recommend refundable fares or travel insurance for these routes.','agent_script':'I completely understand your frustration. I am escalating to our exceptions team to review whether a goodwill waiver can be applied.'},
    }

    # API config
    st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        api_key = st.text_input(
            "Groq API Key (optional — leave blank for demo mode)",
            type="password",
            placeholder="gsk_... — get free at console.groq.com"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_demo = (api_key.strip() == "")
        if use_demo:
            st.info("ℹ️ Running in **Demo Mode**")
        else:
            st.success("✅ Live API mode active")

    # Complaint selector
    st.markdown('<div class="section-header">📝 Select Complaints to Summarise</div>', unsafe_allow_html=True)
    df_c = df[df['customer_complaint'].notna()].copy()
    df_c['complaint_category'] = df_c['customer_complaint'].map(CATEGORY_MAP)

    all_cats = sorted(df_c['complaint_category'].dropna().unique())
    selected_cats = st.multiselect(
        "Select complaint categories to analyse:",
        options=all_cats,
        default=all_cats,
        help="Hold Ctrl/Cmd to select multiple"
    )

    if not selected_cats:
        st.warning("Please select at least one complaint category.")
        st.stop()

    df_sel = df_c[df_c['complaint_category'].isin(selected_cats)]
    st.markdown(f"**{len(df_sel)} complaints** across **{len(selected_cats)} categories** selected.")

    if st.button("🚀 Generate AI Summaries", type="primary", use_container_width=True):
        st.markdown("---")
        st.markdown("### 📄 Complaint Summaries")

        def call_groq(category, count, samples, key):
            sample_text = "\n".join([
                f"  - [{r['booking_id']}] {r['airline']} | {r['cabin_class']} | "
                f"{r['booking_channel']} | Status: {r['booking_status']} | "
                f"Complaint: {r['customer_complaint']}"
                for _, r in samples.iterrows()
            ])
            prompt = f"""You are an expert travel support analyst at Clarity Travel Technology.
Analyse these {count} customer complaints in category: {category}

Complaints:
{sample_text}

Respond ONLY with a valid JSON object:
{{
  "summary": "2-3 sentence summary",
  "urgency": "HIGH or MEDIUM or LOW",
  "root_cause": "one sentence",
  "recommended_action": "specific actionable steps",
  "agent_script": "short agent response script"
}}"""
            payload = json.dumps({
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role":"user","content":prompt}],
                "max_tokens": 400, "temperature": 0.3
            }).encode('utf-8')
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=payload,
                headers={"Content-Type":"application/json","Authorization":f"Bearer {key}"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                content = data['choices'][0]['message']['content']
                content = content.strip().replace('```json','').replace('```','').strip()
                return json.loads(content)

        badge_map = {'HIGH':'badge-high','MEDIUM':'badge-medium','LOW':'badge-low'}
        card_map  = {'HIGH':'high','MEDIUM':'medium','LOW':'low'}
        icon_map  = {'HIGH':'🔴','MEDIUM':'🟡','LOW':'🟢'}

        for cat in selected_cats:
            cat_df  = df_sel[df_sel['complaint_category']==cat]
            count   = len(cat_df)
            samples = cat_df.sample(min(5, count), random_state=42)

            with st.spinner(f"Summarising {cat}..."):
                if use_demo:
                    result = DEMO_SUMMARIES.get(cat, {
                        'summary':f'Multiple {cat} complaints logged.',
                        'urgency':'MEDIUM',
                        'root_cause':'Requires investigation.',
                        'recommended_action':'Review and escalate as needed.',
                        'agent_script':'Thank you for contacting us. We are reviewing your case.'
                    })
                else:
                    try:
                        result = call_groq(cat, count, samples, api_key)
                    except Exception as e:
                        result = DEMO_SUMMARIES.get(cat, {
                            'summary': f'API error: {e}',
                            'urgency': 'MEDIUM',
                            'root_cause': 'API unavailable.',
                            'recommended_action': 'Retry or use demo mode.',
                            'agent_script': 'Thank you for contacting us.'
                        })

            urgency = result.get('urgency','MEDIUM')
            card_class = card_map.get(urgency,'')
            badge_class = badge_map.get(urgency,'')
            icon = icon_map.get(urgency,'⚪')

            st.markdown(f"""
<div class="summary-card {card_class}">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
    <span style="font-size:1.3rem;font-weight:700">{icon} {cat}</span>
    <span class="{badge_class}">{urgency}</span>
    <span style="color:#64748b;font-size:0.9rem">{count} complaints</span>
  </div>
  <p style="margin:6px 0"><b>📋 Summary:</b> {result.get('summary','')}</p>
  <p style="margin:6px 0"><b>🔍 Root Cause:</b> {result.get('root_cause','')}</p>
  <p style="margin:6px 0"><b>✅ Action:</b> {result.get('recommended_action','')}</p>
  <p style="margin:6px 0;background:#f1f5f9;padding:8px;border-radius:6px;border-left:3px solid #60a5fa">
    <b>💬 Agent Script:</b> <i>"{result.get('agent_script','')}"</i></p>
</div>
""", unsafe_allow_html=True)

        st.success(f"✅ Summarised {len(selected_cats)} complaint categories covering {len(df_sel)} complaints.")

# ══════════════════════════════════════════════════════════════
# PAGE 8 — PRIORITY ACTION BOARD
# ══════════════════════════════════════════════════════════════
elif page == "📋  Priority Action Board":
    st.title("📋 Priority Action Board")
    st.markdown("Full urgency-ranked complaint dashboard for the **Support Team Lead**.")

    CATEGORY_MAP = {
        'Refund not processed after 30 days'                    : 'Refund Issues',
        'Duplicate charge on credit card for same booking'      : 'Refund Issues',
        'Fare difference charged incorrectly during rebooking'  : 'Refund Issues',
        'Flight delayed by 3 hours, requesting compensation'    : 'Schedule Change',
        'Schedule change not communicated'                      : 'Schedule Change',
        'Connection time too short, missed connecting flight'   : 'Schedule Change',
        'Baggage not received at destination'                   : 'Baggage',
        'Unable to add extra baggage through portal'            : 'Baggage',
        'PNR not found in airline system after ticketing'       : 'Ticketing Issues',
        'Name spelling error on ticket needs correction'        : 'Ticketing Issues',
        'EMD for ancillary service not reflecting'              : 'Ticketing Issues',
        'Coupon status shows used but passenger never flew'     : 'Ticketing Issues',
        'Meal preference not available onboard'                 : 'Onboard Service',
        'Seat allocation issue, paid for window got middle'     : 'Onboard Service',
        'Infant ticket pricing seems incorrect'                 : 'Pricing Error',
        'Visa rejected but non-refundable ticket purchased'     : 'Policy / Visa',
    }

    URGENCY_MAP = {
        'Ticketing Issues': 'HIGH',
        'Refund Issues':    'HIGH',
        'Schedule Change':  'HIGH',
        'Baggage':          'MEDIUM',
        'Policy / Visa':    'MEDIUM',
        'Pricing Error':    'MEDIUM',
        'Onboard Service':  'LOW',
    }

    df_c = df[df['customer_complaint'].notna()].copy()
    df_c['complaint_category'] = df_c['customer_complaint'].map(CATEGORY_MAP)
    df_c['urgency']            = df_c['complaint_category'].map(URGENCY_MAP)

    # Top KPIs
    total_complaints = len(df_c)
    high_count   = len(df_c[df_c['urgency']=='HIGH'])
    medium_count = len(df_c[df_c['urgency']=='MEDIUM'])
    low_count    = len(df_c[df_c['urgency']=='LOW'])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{total_complaints}</div>
            <div class="lbl">Total Complaints</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card" style="border-color:#ef4444">
            <div class="val" style="color:#ef4444">{high_count}</div>
            <div class="lbl">🔴 HIGH Urgency</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card" style="border-color:#f59e0b">
            <div class="val" style="color:#f59e0b">{medium_count}</div>
            <div class="lbl">🟡 MEDIUM Urgency</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card" style="border-color:#10b981">
            <div class="val" style="color:#10b981">{low_count}</div>
            <div class="lbl">🟢 LOW Urgency</div></div>""", unsafe_allow_html=True)

    # Charts
    st.markdown('<div class="section-header">📊 Urgency Distribution</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        urgency_counts = pd.Series({'HIGH':high_count,'MEDIUM':medium_count,'LOW':low_count})
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = ['#ef4444','#f59e0b','#10b981']
        bars = ax.bar(urgency_counts.index, urgency_counts.values, color=colors, edgecolor='white', width=0.5)
        ax.set_title('Total Complaints by Urgency Level', fontweight='bold')
        ax.set_ylabel('Number of Complaints')
        ax.set_ylim(0, max(urgency_counts.values) * 1.2)
        for bar, val in zip(bars, urgency_counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    str(int(val)), ha='center', fontweight='bold', fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        cat_summary = df_c.groupby(['complaint_category','urgency']).size().reset_index(name='count')
        cat_summary = cat_summary.sort_values('count', ascending=True)
        color_map = {'HIGH':'#ef4444','MEDIUM':'#f59e0b','LOW':'#10b981'}
        bar_colors = [color_map.get(u,'#94a3b8') for u in cat_summary['urgency']]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.barh(cat_summary['complaint_category'], cat_summary['count'],
                color=bar_colors, edgecolor='white')
        ax.set_title('Volume by Category  (Red=HIGH | Yellow=MEDIUM | Green=LOW)', fontweight='bold')
        ax.set_xlabel('Number of Complaints')
        from matplotlib.patches import Patch
        legend_els = [Patch(facecolor='#ef4444',label='HIGH'),
                      Patch(facecolor='#f59e0b',label='MEDIUM'),
                      Patch(facecolor='#10b981',label='LOW')]
        ax.legend(handles=legend_els, loc='lower right', fontsize=9)
        for i, val in enumerate(cat_summary['count']):
            ax.text(val+0.2, i, str(val), va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Priority table
    st.markdown('<div class="section-header">🎯 Ranked Action List</div>', unsafe_allow_html=True)
    priority_order = {'HIGH':0,'MEDIUM':1,'LOW':2}
    icon_map = {'HIGH':'🔴','MEDIUM':'🟡','LOW':'🟢'}

    action_map = {
        'Ticketing Issues': 'Implement PNR validation checks. Create automated name correction workflow. Fix EMD logic.',
        'Refund Issues':    'Escalate to finance immediately. Send refund status updates at 7, 14, 21 days.',
        'Schedule Change':  'Enable real-time SMS/email alerts on schedule changes. Review MCT breach rebooking.',
        'Baggage':          'File World Tracer reports. Fix portal baggage add-on bug (ETA 24 hrs).',
        'Policy / Visa':    'Add visa advisory at checkout. Recommend refundable fares for high-risk destinations.',
        'Pricing Error':    'Audit infant fare rules across all airlines. Run regression tests.',
        'Onboard Service':  'Verify SSR transmission post-ticketing. Implement SSR confirmation loop.',
    }

    summary_df = df_c.groupby(['complaint_category','urgency']).agg(
        count=('booking_id','count'),
        avg_fare=('total_fare_inr','mean'),
        avg_satisfaction=('satisfaction_score','mean'),
        cancel_rate=('is_cancelled','mean')
    ).reset_index()
    summary_df['priority_rank'] = summary_df['urgency'].map(priority_order)
    summary_df = summary_df.sort_values(['priority_rank','count'], ascending=[True,False])

    for _, row in summary_df.iterrows():
        cat     = row['complaint_category']
        urgency = row['urgency']
        icon    = icon_map.get(urgency,'⚪')
        count   = int(row['count'])
        action  = action_map.get(cat, 'Review and escalate as appropriate.')
        avg_sat = f"{row['avg_satisfaction']:.2f}" if not pd.isna(row['avg_satisfaction']) else 'N/A'
        cancel  = f"{row['cancel_rate']*100:.0f}%"

        badge_cls = {'HIGH':'badge-high','MEDIUM':'badge-medium','LOW':'badge-low'}.get(urgency,'')
        card_cls  = {'HIGH':'high','MEDIUM':'medium','LOW':'low'}.get(urgency,'')

        st.markdown(f"""
<div class="summary-card {card_cls}">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:8px">
    <span style="font-size:1.1rem;font-weight:700">{icon} {cat}</span>
    <span class="{badge_cls}">{urgency}</span>
    <span style="color:#475569;font-size:0.9rem">{count} complaints</span>
    <span style="color:#475569;font-size:0.9rem">| Avg satisfaction: {avg_sat}/5</span>
    <span style="color:#475569;font-size:0.9rem">| Cancel rate: {cancel}</span>
  </div>
  <p style="margin:4px 0;font-size:0.95rem"><b>✅ Action:</b> {action}</p>
</div>
""", unsafe_allow_html=True)

    # Filter & export table
    st.markdown('<div class="section-header">🔎 Filter & Explore Raw Complaints</div>', unsafe_allow_html=True)
    urgency_filter = st.multiselect("Filter by Urgency", ['HIGH','MEDIUM','LOW'],
                                     default=['HIGH','MEDIUM','LOW'])
    filtered = df_c[df_c['urgency'].isin(urgency_filter)][
        ['booking_id','airline','cabin_class','booking_channel',
         'booking_status','complaint_category','urgency','customer_complaint','satisfaction_score']
    ].sort_values('urgency')
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.markdown(f"Showing **{len(filtered)}** complaints")

    csv = filtered.to_csv(index=False)
    st.download_button("⬇️ Download Filtered Complaints CSV",
                       data=csv, file_name="clarity_complaints_filtered.csv",
                       mime="text/csv", use_container_width=True)
