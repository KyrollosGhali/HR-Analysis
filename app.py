import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import base64

st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

df = load_data()


st.sidebar.title("Attributes Filter")

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

job_filter = st.sidebar.multiselect(
    "Job Role",
    options=df["Job Role"].unique(),
    default=df["Job Role"].unique()
)

job_level_filter = st.sidebar.multiselect(
    "Job Level",
    options=df["Job Level"].unique(),
    default=df["Job Level"].unique()
)

marital_filter = st.sidebar.multiselect(
    "Marital Status",
    options=df["Marital Status"].unique(),
    default=df["Marital Status"].unique()
)

remote_filter = st.sidebar.multiselect(
    "Remote Work",
    options=df["Remote Work"].unique(),
    default=df["Remote Work"].unique()
)

# Range sliders for Age and Monthly Income
age_min_default = int(df["Age"].min()) if not pd.isna(df["Age"].min()) else 18
age_max_default = int(df["Age"].max()) if not pd.isna(df["Age"].max()) else 65
age_slider = st.sidebar.slider(
    "Age Range",
    min_value=age_min_default,
    max_value=age_max_default,
    value=(age_min_default, age_max_default),
)

income_min_default = int(df["Monthly Income"].min()) if not pd.isna(df["Monthly Income"].min()) else 0
income_max_default = int(df["Monthly Income"].max()) if not pd.isna(df["Monthly Income"].max()) else 100000
income_slider = st.sidebar.slider(
    "Monthly Income Range",
    min_value=income_min_default,
    max_value=income_max_default,
    value=(income_min_default, income_max_default),
    step=100,
)

# attrition_filter = st.sidebar.multiselect(
#     "Attrition",
#     options=df["Attrition"].unique(),
#     default=df["Attrition"].unique()
# )


filtered_df = df[
    (df["Gender"].isin(gender_filter))
    & (df["Job Role"].isin(job_filter))
    & (df["Job Level"].isin(job_level_filter))
    & (df["Age"].between(age_slider[0], age_slider[1]))
    & (df["Monthly Income"].between(income_slider[0], income_slider[1]))
    & (df["Marital Status"].isin(marital_filter))
    & (df["Remote Work"].isin(remote_filter))
]
filtered_df = filtered_df.copy()
filtered_df["AttritionNumeric"] = (
    filtered_df["Attrition"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(
        {
            "yes": 1,
            "no": 0,
            "left": 1,
            "stayed": 0,
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
        }
    )
)


logo_path = Path("images/logo.png")

if logo_path.exists():
    logo_base64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .fixed-logo {{
            position: fixed;
            top: 3.8rem;
            right: 1rem;
            z-index: 999;
            background: rgba(255, 255, 255, 0);
            border-radius: 12px;
            padding: 8px 10px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
        }}
        .fixed-logo img {{
            width: 150px;
            height: auto;
            display: block;
        }}
        @media (max-width: 768px) {{
            .fixed-logo {{
                top: 4.2rem;
                right: 0.5rem;
                padding: 6px 8px;
            }}
            .fixed-logo img {{
                width: 110px;
            }}
        }}
        </style>
        <div class="fixed-logo">
            <img src="data:image/png;base64,{logo_base64}" alt="Logo" />
        </div>
        """,
        unsafe_allow_html=True,
    )

st.title("Employee Attrition Dashboard")

total_employees = len(filtered_df)

attrition_rate = (
    filtered_df["AttritionNumeric"].mean() * 100
)

if pd.isna(attrition_rate):
    attrition_rate = 0.0

avg_age = filtered_df["Age"].mean()

avg_income = filtered_df["Monthly Income"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Employees",
    f"{total_employees:,}"
)

col2.metric(
    "Attrition Rate",
    f"{attrition_rate:.2f}%"
)

col3.metric(
    "Average Age",
    f"{avg_age:.1f}"
)

col4.metric(
    "Avg Monthly Income",
    f"${avg_income:,.0f}"
)

st.divider()


col1, col2 = st.columns(2)

with col1:

    fig = px.pie(
        filtered_df,
        names="Attrition",
        title="Attrition Distribution"
    )
    fig.update_traces(textinfo='percent+label', hovertemplate='%{label}: %{percent} (%{value})')

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    attr_gender = (
        filtered_df.groupby("Gender")["AttritionNumeric"]
        .mean()
        .reset_index()
    )

    attr_gender["AttritionNumeric"] *= 100

    fig = px.bar(
        attr_gender,
        x="Gender",
        y="AttritionNumeric",
        color="Gender",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        title="Attrition Rate by Gender"
    )
    fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    fig.update_layout(yaxis_title='Attrition Rate (%)', uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)

    st.plotly_chart(
        fig,
        use_container_width=True
    )


col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        filtered_df,
        x="Age",
        nbins=25,
        title="Age Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.box(
        filtered_df,
        x="Attrition",
        y="Age",
        title="Age vs Attrition"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        filtered_df,
        x="Monthly Income",
        title="Income Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.box(
        filtered_df,
        x="Attrition",
        y="Monthly Income",
        title="Income vs Attrition"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


role_attr = (
    filtered_df.groupby("Job Role")["AttritionNumeric"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

role_attr["AttritionNumeric"] *= 100

fig = px.bar(
    role_attr,
    x="Job Role",
    y="AttritionNumeric",
    color="Job Role",
    color_discrete_sequence=px.colors.qualitative.Plotly,
    title="Attrition Rate by Job Role"
)
fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
fig.update_layout(yaxis_title='Attrition Rate (%)', uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)

st.plotly_chart(
    fig,
    use_container_width=True
)

wlb = (
    filtered_df.groupby(
        "Work-Life Balance"
    )["AttritionNumeric"]
    .mean()
    .reset_index()
)

wlb["AttritionNumeric"] *= 100

fig = px.bar(
    wlb,
    x="Work-Life Balance",
    y="AttritionNumeric",
    color="Work-Life Balance",
    color_discrete_sequence=px.colors.qualitative.Plotly,
    title="Work-Life Balance Impact"
)
fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
fig.update_layout(yaxis_title='Attrition Rate (%)', uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)

st.plotly_chart(
    fig,
    use_container_width=True
)


satisfaction = (
    filtered_df.groupby(
        "Job Satisfaction"
    )["AttritionNumeric"]
    .mean()
    .reset_index()
)

satisfaction["AttritionNumeric"] *= 100

fig = px.bar(
    satisfaction,
    x="Job Satisfaction",
    y="AttritionNumeric",
    color="Job Satisfaction",
    color_discrete_sequence=px.colors.qualitative.Plotly,
    title="Job Satisfaction Impact"
)
fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
fig.update_layout(yaxis_title='Attrition Rate (%)', uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)

st.plotly_chart(
    fig,
    use_container_width=True
)


remote = (
    filtered_df.groupby(
        "Remote Work"
    )["AttritionNumeric"]
    .mean()
    .reset_index()
)

remote["AttritionNumeric"] *= 100

fig = px.bar(
    remote,
    x="Remote Work",
    y="AttritionNumeric",
    color="Remote Work",
    color_discrete_sequence=px.colors.qualitative.Plotly,
    title="Remote Work Impact"
)
fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
fig.update_layout(yaxis_title='Attrition Rate (%)', uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)

st.plotly_chart(
    fig,
    use_container_width=True
)

# # ==========================
# # SCATTER PLOT
# # ==========================

# fig = px.scatter(
#     filtered_df,
#     x="Monthly Income",
#     y="Number of Promotions",
#     color="Attrition",
#     title="Income vs Promotions"
# )

# st.plotly_chart(
#     fig,
#     use_container_width=True
# )


st.subheader("Promotions: Stayed vs Left")

prom_pivot = (
    filtered_df.groupby(["Number of Promotions", "Attrition"]).size().reset_index(name="Count")
)

prom_pivot["Number of Promotions"] = pd.to_numeric(prom_pivot["Number of Promotions"], errors="coerce").fillna(0).astype(int)
prom_pivot = prom_pivot.sort_values("Number of Promotions")

fig = px.line(
    prom_pivot,
    x="Number of Promotions",
    y="Count",
    color="Attrition",
    markers=True,
    title="Employees Stayed vs Left by Number of Promotions",
)
fig.update_layout(xaxis=dict(dtick=1), yaxis_title="Count", legend_title="Attrition")

st.plotly_chart(
    fig,
    use_container_width=True
)


st.subheader("Job Level Impact")

job_level = (
    filtered_df.groupby("Job Level")["AttritionNumeric"]
    .mean()
    .reset_index()
)

job_level["AttritionNumeric"] *= 100

fig = px.bar(
    job_level,
    x="Job Level",
    y="AttritionNumeric",
    color="Job Level",
    color_discrete_sequence=px.colors.qualitative.Plotly,
    title="Attrition Rate by Job Level"
)
fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
fig.update_layout(yaxis_title='Attrition Rate (%)', uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)

st.plotly_chart(
    fig,
    use_container_width=True
)


# Business Recommendations
st.subheader("Business Recommendations")
st.markdown(
    """
Based on the findings, the organization should:

1. Expand remote and hybrid work options.
2. Improve work-life balance programs and policies.
3. Establish clearer promotion and career development pathways.
4. Support employees with long commutes through flexible work arrangements.
5. Strengthen employer branding and company reputation.
6. Focus retention strategies on entry-level employees and single employees, who represent the highest-risk groups.
"""
)

# st.subheader("Employee Explorer")

# st.dataframe(
#     filtered_df,
#     use_container_width=True
# )

csv = filtered_df.to_csv(index=False)

st.download_button(
    "Download Filtered Data",
    csv,
    "filtered_hr_data.csv",
    "text/csv"
)

