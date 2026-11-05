import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="Who Survived the Titanic?",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Bright UI & Green Theme Styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #475569;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 12px;
        font-weight: 600;
        color: #047857;
        background-color: #d1fae5;
        padding: 2px 8px;
        border-radius: 999px;
        display: inline-block;
        margin-top: 6px;
    }
    .story-card {
        background-color: #f0fdf4;
        border-left: 4px solid #10b981;
        border-radius: 0 10px 10px 0;
        padding: 16px 20px;
    }
    .story-title {
        font-size: 16px;
        font-weight: 700;
        color: #065f46;
        margin-bottom: 8px;
    }
    .story-text {
        font-size: 14px;
        line-height: 1.6;
        color: #1e293b;
    }
    .footnote {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Header & Headline
st.title("Who Survived the Titanic?")
st.markdown("#### **Your odds depended on your ticket and your gender**")

# 4. File Uploader
uploaded_file = st.file_uploader("Upload train.csv (Kaggle Titanic)", type=["csv"])

@st.cache_data
def load_default_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_default_data()

# Data preparation
df['Pclass_Label'] = df['Pclass'].map({1: '1st class', 2: '2nd class', 3: '3rd class'})
df['Port_Name'] = df['Embarked'].map({'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'})

# 5. Ticket Filter Radio Buttons
ticket_option = st.radio(
    "Pick your ticket",
    options=["All passengers", "1st class", "2nd class", "3rd class"],
    horizontal=True
)

if ticket_option == "All passengers":
    filtered_df = df
else:
    filtered_df = df[df['Pclass_Label'] == ticket_option]

# 6. KPI Calculations
total_count = len(filtered_df)
survived_count = int(filtered_df['Survived'].sum())
overall_surv_rate = (survived_count / total_count * 100) if total_count > 0 else 0

female_df = filtered_df[filtered_df['Sex'] == 'female']
male_df = filtered_df[filtered_df['Sex'] == 'male']
f_rate = (female_df['Survived'].mean() * 100) if len(female_df) > 0 else 0
m_rate = (male_df['Survived'].mean() * 100) if len(male_df) > 0 else 0

kids_df = filtered_df[filtered_df['Age'] < 13]
kids_rate = (kids_df['Survived'].mean() * 100) if len(kids_df) > 0 else 0

p1_rate = df[df['Pclass'] == 1]['Survived'].mean() * 100
p3_rate = df[df['Pclass'] == 3]['Survived'].mean() * 100

st.write("")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Overall survival</div>
        <div class="metric-value">{overall_surv_rate:.0f}%</div>
        <span class="metric-sub">↑ {survived_count} of {total_count} passengers</span>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Women vs men</div>
        <div class="metric-value">{f_rate:.0f}% vs {m_rate:.0f}%</div>
        <span class="metric-sub">Protocol bias</span>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Children under 13</div>
        <div class="metric-value">{kids_rate:.0f}%</div>
        <span class="metric-sub">Age protection</span>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">1st vs 3rd class (all)</div>
        <div class="metric-value">{p1_rate:.0f}% vs {p3_rate:.0f}%</div>
        <span class="metric-sub">Socioeconomic gap</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.divider()

# 7. Middle Row: Side-by-Side Charts
col_mid1, col_mid2 = st.columns(2)

# Common clean layout helper for Plotly
clean_plot_layout = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#334155', size=12),
    margin=dict(t=30, b=20, l=40, r=20)
)

with col_mid1:
    st.subheader("Survival rate by sex and class")
    
    rate_df = (
        filtered_df.groupby(['Sex', 'Pclass_Label'])['Survived']
        .mean()
        .mul(100)
        .reset_index()
    )
    rate_df['Sex'] = rate_df['Sex'].map({'female': 'Female', 'male': 'Male'})

    # Green Palette: Vibrant Emerald for 1st, Light Mint for 2nd, Muted Slate for 3rd
    fig_sex_class = px.bar(
        rate_df,
        x='Sex',
        y='Survived',
        color='Pclass_Label',
        barmode='group',
        text=rate_df['Survived'].apply(lambda x: f"{x:.0f}%"),
        color_discrete_map={
            '1st class': '#10b981',  # Bright emerald green
            '2nd class': '#6ee7b7',  # Light mint green
            '3rd class': '#cbd5e1'   # Neutral slate
        },
        category_orders={"Pclass_Label": ["1st class", "2nd class", "3rd class"]}
    )
    fig_sex_class.update_layout(
        **clean_plot_layout,
        yaxis=dict(
            title="Survival rate",
            range=[0, 110],
            ticksuffix="%",
            showgrid=True,
            gridcolor="#f1f5f9"
        ),
        xaxis=dict(title="", showgrid=False),
        showlegend=False,
        height=350
    )
    fig_sex_class.update_traces(textposition='outside')
    st.plotly_chart(fig_sex_class, use_container_width=True)

with col_mid2:
    st.subheader("Age of passengers: who survived?")
    
    age_df = filtered_df.dropna(subset=['Age']).copy()
    missing_age_count = filtered_df['Age'].isna().sum()

    bin_edges = list(range(0, 85, 5))
    age_df['Age_Bin'] = pd.cut(age_df['Age'], bins=bin_edges, right=False)
    
    hist_data = age_df.groupby(['Age_Bin', 'Survived'], observed=False).size().unstack(fill_value=0)
    bin_labels = [b.left for b in hist_data.index]

    fig_age = go.Figure()
    # Bottom: Survived (Bright Green)
    fig_age.add_trace(go.Bar(
        x=bin_labels,
        y=hist_data[1] if 1 in hist_data else [0]*len(bin_labels),
        name='Survived',
        marker_color='#10b981'
    ))
    # Top: Died (Soft slate grey for clean contrast)
    fig_age.add_trace(go.Bar(
        x=bin_labels,
        y=hist_data[0] if 0 in hist_data else [0]*len(bin_labels),
        name='Died',
        marker_color='#cbd5e1'
    ))

    fig_age.update_layout(
        **clean_plot_layout,
        barmode='stack',
        xaxis=dict(
            title="Age (years)",
            tickmode='linear',
            tick0=0,
            dtick=5,
            showgrid=False
        ),
        yaxis=dict(
            title="Passengers",
            showgrid=True,
            gridcolor="#f1f5f9"
        ),
        showlegend=False,
        height=350,
        bargap=0.15
    )
    st.plotly_chart(fig_age, use_container_width=True)
    st.markdown(f'<div class="footnote">{missing_age_count} passengers have no recorded age and are not shown.</div>', unsafe_allow_html=True)

st.write("")
st.divider()

# 8. Bottom Row: Embarkation Port & Narrative Insight
col_bot1, col_bot2 = st.columns(2)

with col_bot1:
    st.subheader("Survival rate by port of embarkation")
    
    port_df = filtered_df.dropna(subset=['Port_Name'])
    port_rate = (
        port_df.groupby('Port_Name')['Survived']
        .mean()
        .mul(100)
        .reset_index()
    )
    
    order_dict = {'Cherbourg': 3, 'Queenstown': 2, 'Southampton': 1}
    port_rate['rank'] = port_rate['Port_Name'].map(order_dict)
    port_rate = port_rate.sort_values('rank', ascending=True)

    fig_port = go.Figure(go.Bar(
        x=port_rate['Survived'],
        y=port_rate['Port_Name'],
        orientation='h',
        text=port_rate['Survived'].apply(lambda x: f"{x:.0f}%"),
        textposition='outside',
        marker_color=['#cbd5e1', '#a7f3d0', '#10b981']  # Highlights Cherbourg in bright green
    ))

    fig_port.update_layout(
        **clean_plot_layout,
        xaxis=dict(
            title="Survival rate",
            ticksuffix="%",
            range=[0, 75],
            showgrid=True,
            gridcolor="#f1f5f9"
        ),
        yaxis=dict(title="", showgrid=False),
        height=260
    )
    st.plotly_chart(fig_port, use_container_width=True)
    st.markdown('<div class="footnote">Passengers with an unknown port (?) are excluded.</div>', unsafe_allow_html=True)

with col_bot2:
    st.markdown("""
    <div class="story-card">
        <div class="story-title">What this tells us</div>
        <div class="story-text">
            When disaster struck, survival was far from random. Women were far more likely to survive than men (<strong>74% vs 19%</strong>), first-class passengers fared much better than third-class (<strong>63% vs 24%</strong>), and children did better than adults (<strong>58% vs 36%</strong>). Passengers from Cherbourg survived most often (<strong>55%</strong> vs 39% for Queenstown and 34% for Southampton), likely because more of them held first-class tickets.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="footnote" style="margin-top: 14px;">Source: Kaggle Titanic dataset (train.csv), 891 passengers.</div>', unsafe_allow_html=True)