from pathlib import Path
import base64
import itertools

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide",
)

DATA_FILES = [Path("train.csv"), Path("test.csv")]
ATTRITION_MAP = {
    "yes": 1,
    "no": 0,
    "left": 1,
    "stayed": 0,
    "1": 1,
    "0": 0,
    "true": 1,
    "false": 0,
}

JOB_SATISFACTION_ORDER = ["Low", "Medium", "High", "Very High"]
WORK_LIFE_ORDER = ["Poor", "Fair", "Good", "Excellent"]
AGE_BINS = [17, 24, 34, 44, 54, 120]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55+"]
TENURE_BINS = [0, 5, 10, 15, 20, 30, 60]
TENURE_LABELS = ["0-5", "6-10", "11-15", "16-20", "21-30", "31+"]
ACTIONABLE_COLUMNS = [
    "Work-Life Balance",
    "Overtime",
    "Remote Work",
    "Leadership Opportunities",
    "Innovation Opportunities",
    "Employee Recognition",
    "Company Reputation",
    "Job Satisfaction",
]
RISK_PROFILE_COLUMNS = [
    "Work-Life Balance",
    "Job Level",
    "Marital Status",
    "Overtime",
    "Remote Work",
    "Job Satisfaction",
    "Leadership Opportunities",
    "Innovation Opportunities",
    "Employee Recognition",
    "Company Reputation",
]


@st.cache_data
def load_data() -> pd.DataFrame:
    frames = []
    for file_path in DATA_FILES:
        if file_path.exists():
            frame = pd.read_csv(file_path)
            frame["Dataset Source"] = file_path.stem
            frames.append(frame)

    if not frames:
        raise FileNotFoundError("Could not find train.csv or test.csv.")

    data = pd.concat(frames, ignore_index=True)
    data["AttritionFlag"] = (
        data["Attrition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(ATTRITION_MAP)
    )

    data["Age Group"] = pd.cut(
        data["Age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        include_lowest=True,
    )
    data["Tenure Band"] = pd.cut(
        data["Years at Company"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        include_lowest=True,
    )
    return data


def pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def fmt_points(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:+.1f} pts"


def attrition_rate(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    return float(frame["AttritionFlag"].mean())


def summarize_factor(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    summary = (
        frame.groupby(column, observed=True)
        .agg(
            Employees=("AttritionFlag", "size"),
            Leavers=("AttritionFlag", "sum"),
            AttritionRate=("AttritionFlag", "mean"),
        )
        .sort_values("AttritionRate", ascending=False)
        .reset_index()
    )
    summary["AttritionRatePct"] = summary["AttritionRate"] * 100
    return summary


def make_bar_chart(summary: pd.DataFrame, x_col: str, y_col: str, title: str, color_col: str | None = None) -> go.Figure:
    fig = px.bar(
        summary,
        x=x_col,
        y=y_col,
        color=color_col,
        text_auto=False,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    fig.update_layout(
        yaxis_title="Attrition rate (%)",
        xaxis_title="",
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        showlegend=False if color_col is not None else True,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def make_count_chart(summary: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    fig = px.bar(
        summary,
        x=x_col,
        y=y_col,
        text=y_col,
        title=title,
        color_discrete_sequence=["#2563eb"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis_title="Employees leaving",
        xaxis_title="",
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def build_income_fairness(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for job_level, group in frame.groupby("Job Level", observed=True):
        working = group[["Job Level", "Monthly Income", "AttritionFlag"]].copy()
        unique_values = working["Monthly Income"].nunique()
        if unique_values < 2:
            working["Income Band"] = "Single band"
        else:
            band_count = min(4, unique_values)
            band_labels = [f"Q{i}" for i in range(1, band_count + 1)]
            working["Income Band"] = pd.qcut(
                working["Monthly Income"].rank(method="first"),
                q=band_count,
                labels=band_labels,
            )
        rows.append(working)

    income_frame = pd.concat(rows, ignore_index=True)
    return (
        income_frame.groupby(["Job Level", "Income Band"], observed=True)["AttritionFlag"]
        .mean()
        .reset_index()
    )


def build_tenure_summary(frame: pd.DataFrame) -> pd.DataFrame:
    tenure = (
        frame.groupby("Tenure Band", observed=True)
        .agg(Employees=("AttritionFlag", "size"), AttritionRate=("AttritionFlag", "mean"))
        .reset_index()
    )
    tenure["AttritionRatePct"] = tenure["AttritionRate"] * 100
    return tenure


def best_actionable_driver(frame: pd.DataFrame, column: str) -> dict | None:
    summary = frame.groupby(column, observed=True).agg(
        Employees=("AttritionFlag", "size"),
        AttritionRate=("AttritionFlag", "mean"),
    )
    summary = summary[summary["Employees"] >= 100]
    if summary.empty:
        return None

    ranked = summary.sort_values(["AttritionRate", "Employees"], ascending=[False, False])
    best_row = ranked.iloc[0]
    best_value = ranked.index[0]
    return {
        "driver": column,
        "value": best_value,
        "employees": int(best_row["Employees"]),
        "rate": float(best_row["AttritionRate"]),
        "lift": float(best_row["AttritionRate"] - attrition_rate(frame)),
    }


def find_high_risk_profile(frame: pd.DataFrame, candidate_columns: list[str], min_count: int = 500) -> tuple[dict | None, pd.DataFrame]:
    baseline = attrition_rate(frame)
    all_profiles: list[dict] = []

    for combo_size in (3, 4):
        for combo in itertools.combinations(candidate_columns, combo_size):
            working = frame.copy()
            parts: list[str] = []
            valid = True

            for column in combo:
                summary = working.groupby(column, observed=True).agg(
                    Employees=("AttritionFlag", "size"),
                    AttritionRate=("AttritionFlag", "mean"),
                )
                summary = summary[summary["Employees"] >= 100]
                if summary.empty:
                    valid = False
                    break

                ranked = summary.sort_values(["AttritionRate", "Employees"], ascending=[False, False])
                chosen_value = ranked.index[0]
                working = working[working[column] == chosen_value]
                parts.append(f"{column} = {chosen_value}")

            if not valid or len(working) < min_count:
                continue

            profile_rate = attrition_rate(working)
            all_profiles.append(
                {
                    "Profile": " | ".join(parts),
                    "Employees": len(working),
                    "AttritionRate": profile_rate,
                    "Lift": profile_rate - baseline,
                }
            )

    if not all_profiles:
        return None, pd.DataFrame(columns=["Profile", "Employees", "AttritionRate", "Lift"])

    profile_table = pd.DataFrame(all_profiles).sort_values(["AttritionRate", "Employees"], ascending=[False, False]).reset_index(drop=True)
    best_profile = profile_table.iloc[0].to_dict()
    return best_profile, profile_table.head(10)


def make_heatmap(matrix: pd.DataFrame, title: str, x_title: str, y_title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Attrition rate"),
            hovertemplate=f"{x_title}: %{{x}}<br>{y_title}: %{{y}}<br>Attrition: %{{z:.1%}}<extra></extra>",
        )
    )
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, margin=dict(l=10, r=10, t=60, b=10))
    return fig


data = load_data()
baseline_rate = attrition_rate(data)
remote_share = data["Remote Work"].value_counts(normalize=True).get("Yes", 0.0)
total_leavers = int(data["AttritionFlag"].sum())

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
            width: 300px;
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
st.caption(
    f"Week 1 - Kyfa Internship"
)

# st.sidebar.title("Analysis Scope")
# st.sidebar.info(
#     "This dashboard uses the combined train + test dataset for the headline questions so the numbers stay aligned with the brief."
# )

col1, col2, col3, col4 = st.columns(4)
col1.metric("Employees", f"{len(data):,}")
col2.metric("Attrition rate", pct(baseline_rate))
col3.metric("Employees who left", f"{total_leavers:,}")
col4.metric("Remote workforce", pct(remote_share))

st.divider()

overview_col1, overview_col2 = st.columns(2)
role_summary = summarize_factor(data, "Job Role")

with overview_col1:
    top_roles_by_volume = role_summary.sort_values("Leavers", ascending=False)
    fig = make_count_chart(top_roles_by_volume, "Job Role", "Leavers", "Where the exits are concentrated by job role")
    st.plotly_chart(fig, use_container_width=True)

with overview_col2:
    fig = make_bar_chart(role_summary.sort_values("AttritionRate", ascending=False), "Job Role", "AttritionRatePct", "Attrition rate by job role")
    st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"""
    **Headline take-away:** Technology has the biggest loss volume, while Education has the highest rate of attrition. That means leadership should look first at Technology for immediate headcount impact, and at Education for the riskier retention pattern.
    """
)

tabs = st.tabs([
    "Q1 Headline",
    "Q2 Overtime",
    "Q3 Remote work",
    "Q4 Pay fairness",
    "Q5 Retention timeline",
    "Q6 Engagement warning signs",
    "Q7 Life stage",
    "Q8 Career stagnation",
    "Q9 Highest-risk profile",
    "Q10 What moves the needle",
])

with tabs[0]:
    st.subheader("Q1 · The headline")
    q1_col1, q1_col2 = st.columns(2)
    with q1_col1:
        fig = make_count_chart(
            top_roles_by_volume,
            "Job Role",
            "Leavers",
            "Job roles with the most people leaving"
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="q1_chart_leavers"
        )

    with q1_col2:
        fig = make_bar_chart(
            role_summary.sort_values("AttritionRate", ascending=False),
            "Job Role",
            "AttritionRatePct",
            "Attrition rate by job role"
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="q1_chart_attrition_rate"
        )
    top_role = top_roles_by_volume.iloc[0]
    highest_rate_role = role_summary.sort_values("AttritionRate", ascending=False).iloc[0]
    st.markdown(
        f"""
        **Answer:** {pct(baseline_rate)} of employees left overall. Technology is the first place to look because it has the highest number of leavers ({int(top_role['Leavers']):,}), even though Education has the highest rate ({pct(float(highest_rate_role['AttritionRate']))}).

        **Why this matters:** Technology is the largest absolute drain on headcount, so fixing it moves the most people. Education is the riskiest role by rate, so it should be the second priority for root-cause review.
        """
    )
    st.info("Recommendation: start with Technology for volume, then investigate Education for role design, workload, pay bands, and manager support.")

with tabs[1]:
    st.subheader("Q2 · Overtime")
    overtime_summary = summarize_factor(data, "Overtime")
    overtime_no = overtime_summary[overtime_summary["Overtime"] == "No"].iloc[0]
    overtime_yes = overtime_summary[overtime_summary["Overtime"] == "Yes"].iloc[0]
    lift_pp = float(overtime_yes["AttritionRate"] - overtime_no["AttritionRate"])
    relative_lift = float(overtime_yes["AttritionRate"] / overtime_no["AttritionRate"] - 1)

    q2_col1, q2_col2 = st.columns([1, 1])
    with q2_col1:
        fig = make_bar_chart(overtime_summary.sort_values("AttritionRate", ascending=False), "Overtime", "AttritionRatePct", "Attrition rate for overtime vs non-overtime employees")
        st.plotly_chart(fig, use_container_width=True)
    with q2_col2:
        st.metric("Overtime attrition", pct(float(overtime_yes["AttritionRate"])))
        st.metric("No-overtime attrition", pct(float(overtime_no["AttritionRate"])))
        st.metric("Difference", fmt_points(lift_pp))
        st.metric("Relative lift", f"{relative_lift * 100:.1f}% more likely")

    st.markdown(
        f"""
        **Answer:** Yes. Employees who work overtime leave at {pct(float(overtime_yes['AttritionRate']))} versus {pct(float(overtime_no['AttritionRate']))} for employees who do not, a gap of {fmt_points(lift_pp)}.

        **Business read:** overtime is a workload signal, not just a scheduling detail. If the company wants attrition down, HR and managers need to cut recurring overtime, rebalance teams, and make managers accountable for workload planning.
        """
    )
    st.warning("Recommendation: treat overtime as an early warning indicator and reduce chronic overload before it turns into resignations.")

with tabs[2]:
    st.subheader("Q3 · Remote work")
    remote_summary = summarize_factor(data, "Remote Work")
    remote_no = remote_summary[remote_summary["Remote Work"] == "No"].iloc[0]
    remote_yes = remote_summary[remote_summary["Remote Work"] == "Yes"].iloc[0]
    remote_no_rate = float(remote_no["AttritionRate"])
    remote_yes_rate = float(remote_yes["AttritionRate"])
    remote_drop = remote_no_rate - remote_yes_rate

    q3_col1, q3_col2 = st.columns([1, 1])
    with q3_col1:
        fig = make_bar_chart(remote_summary.sort_values("AttritionRate", ascending=False), "Remote Work", "AttritionRatePct", "Attrition rate by remote-work access")
        st.plotly_chart(fig, use_container_width=True)
    with q3_col2:
        st.metric("Remote-work share", pct(remote_share))
        st.metric("No remote attrition", pct(remote_no_rate))
        st.metric("Remote attrition", pct(remote_yes_rate))
        st.metric("Effect size", fmt_points(remote_drop))

    st.markdown(
        f"""
        **Answer:** Remote work appears to keep people. Attrition is {pct(remote_no_rate)} for employees without remote work, but only {pct(remote_yes_rate)} for employees with remote work, a difference of {fmt_points(remote_drop)}.

        **Caution:** only {pct(remote_share)} of staff work remotely, so this is a strong association but not proof of pure cause. It still suggests remote or hybrid access is a valuable retention lever.
        """
    )
    st.info("Recommendation: expand remote/hybrid access where the role allows it, but do not overclaim that remote work alone explains retention.")

with tabs[3]:
    st.subheader("Q4 · Pay fairness")
    income_frame = build_income_fairness(data)
    income_pivot = (
        income_frame.pivot(index="Job Level", columns="Income Band", values="AttritionFlag")
        .reindex(index=[lvl for lvl in ["Entry", "Mid", "Senior"] if lvl in income_frame["Job Level"].unique()])
    )

    q4_col1, q4_col2 = st.columns([1.2, 0.8])
    with q4_col1:
        fig = make_heatmap(income_pivot, "Attrition inside job levels by pay band", "Income band", "Job level")
        st.plotly_chart(fig, use_container_width=True)
    with q4_col2:
        level_summary = []
        for level, group in data.groupby("Job Level", observed=True):
            banded = build_income_fairness(group)
            low_band = banded.sort_values("Income Band").iloc[0]
            high_band = banded.sort_values("Income Band").iloc[-1]
            level_summary.append(
                {
                    "Job Level": level,
                    "Low band": float(low_band["AttritionFlag"]),
                    "High band": float(high_band["AttritionFlag"]),
                    "Gap": float(low_band["AttritionFlag"] - high_band["AttritionFlag"]),
                }
            )
        # level_table = pd.DataFrame(level_summary).sort_values("Gap", ascending=False)
        # st.dataframe(
        #     level_table.assign(
        #         **{
        #             "Low band": level_table["Low band"].map(pct),
        #             "High band": level_table["High band"].map(pct),
        #             "Gap": level_table["Gap"].map(fmt_points),
        #         }
        #     ),
        #     use_container_width=True,
        #     hide_index=True,
        # )

    st.markdown(
        """
        **Answer:** Yes, lower-paid employees generally leave more often within the same job level, but the effect flattens quickly.

        In Entry roles, attrition stays high across all pay bands, so simply paying a little more does not solve the problem. In Mid and Senior roles, the drop from the lowest band to the highest band is much smaller, which suggests pay matters most at the bottom of the band structure.

        **Recommendation:** tighten pay floors and compression in Entry and early Mid roles, then use growth, recognition, and workload fixes for higher levels where extra pay alone has diminishing returns.
        """
    )

with tabs[4]:
    st.subheader("Q5 · The retention timeline")
    tenure_summary = build_tenure_summary(data)
    q5_col1, q5_col2 = st.columns([1, 1])
    with q5_col1:
        fig = px.bar(
            tenure_summary,
            x="Tenure Band",
            y="AttritionRatePct",
            title="Attrition by time at company",
            color_discrete_sequence=["#0f766e"],
        )
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        fig.update_layout(yaxis_title="Attrition rate (%)", xaxis_title="Years at company", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with q5_col2:
        earliest_band = tenure_summary.sort_values("AttritionRate", ascending=False).iloc[0]
        st.metric("Highest-risk stage", str(earliest_band["Tenure Band"]))
        st.metric("Attrition at that stage", pct(float(earliest_band["AttritionRate"])))
        st.metric("Company-average attrition", pct(baseline_rate))

    st.markdown(
        """
        **Answer:** Attrition is highest at the beginning of the employee lifecycle and then declines as time at the company increases.

        **Meaning for HR:** retention effort should be concentrated on onboarding, the first-year manager experience, and the early-career path, because that is where the company is losing the most people.
        """
    )
    st.info("Recommendation: invest first in onboarding, 90-day check-ins, and first-year manager coaching rather than waiting until tenure is already long.")

with tabs[5]:
    st.subheader("Q6 · Engagement warning signs")
    combo_summary = (
        data.groupby(["Job Satisfaction", "Work-Life Balance"], observed=True)["AttritionFlag"]
        .mean()
        .reset_index()
    )
    combo_matrix = combo_summary.pivot(index="Job Satisfaction", columns="Work-Life Balance", values="AttritionFlag")
    combo_matrix = combo_matrix.reindex(index=JOB_SATISFACTION_ORDER, columns=WORK_LIFE_ORDER)

    q6_col1, q6_col2 = st.columns([1.2, 0.8])
    with q6_col1:
        fig = make_heatmap(combo_matrix, "Attrition by job satisfaction and work-life balance", "Work-life balance", "Job satisfaction")
        st.plotly_chart(fig, use_container_width=True)
    with q6_col2:
        worst_combo = combo_summary.sort_values("AttritionFlag", ascending=False).iloc[0]
        st.metric("Worst combination", f"{worst_combo['Job Satisfaction']} / {worst_combo['Work-Life Balance']}")
        st.metric("Attrition at worst combo", pct(float(worst_combo["AttritionFlag"])))
        st.metric("Company average", pct(baseline_rate))

    st.markdown(
        """
        **Answer:** The strongest early-warning sign is poor work-life balance combined with low satisfaction. That combination has the highest attrition risk and should be treated as a manager alert, not just an engagement score.

        **Recommendation:** managers should watch for people who report poor balance and weak satisfaction together, then intervene with workload changes, coaching, and a fast check on role fit.
        """
    )

with tabs[6]:
    st.subheader("Q7 · Life stage")
    age_summary = summarize_factor(data, "Age Group")
    marital_summary = summarize_factor(data, "Marital Status")
    dependents_summary = (
        data.groupby(["Marital Status", "Number of Dependents"], observed=True)["AttritionFlag"]
        .mean()
        .reset_index()
    )
    dependents_pivot = dependents_summary.pivot(index="Marital Status", columns="Number of Dependents", values="AttritionFlag")

    q7_col1, q7_col2 = st.columns(2)
    with q7_col1:
        fig = make_bar_chart(age_summary.sort_values("AttritionRate", ascending=False), "Age Group", "AttritionRatePct", "Attrition by age group")
        st.plotly_chart(fig, use_container_width=True)
    with q7_col2:
        fig = make_bar_chart(marital_summary.sort_values("AttritionRate", ascending=False), "Marital Status", "AttritionRatePct", "Attrition by marital status")
        st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(
        make_heatmap(dependents_pivot, "Attrition by marital status and dependents", "Number of dependents", "Marital status"),
        use_container_width=True,
    )

    high_risk_life_stage = (
        data.groupby(["Marital Status", "Age Group"], observed=True)["AttritionFlag"]
        .mean()
        .reset_index()
        .sort_values("AttritionFlag", ascending=False)
        .iloc[0]
    )

    st.markdown(
        """
        **Answer:** Age, marital status, and dependents all matter, but the clearest risk group is Single employees, especially in the younger age bands.

        **Best read of the data:** younger employees have higher attrition, and single employees leave far more often than married employees. Dependents soften the pattern a little, but they do not eliminate the risk.

        **Recommendation:** retain this group with faster career visibility, tighter manager support, and flexibility that helps them feel the company is investing in them early.
        """
    )
    st.info(
        f"Highest-risk age/marital slice: {high_risk_life_stage['Marital Status']} employees in the {high_risk_life_stage['Age Group']} band."
    )

with tabs[7]:
    st.subheader("Q8 · Career stagnation")
    promotions_summary = (
        data.groupby("Number of Promotions", observed=True)["AttritionFlag"]
        .mean()
        .reset_index()
        .sort_values("Number of Promotions")
    )
    job_level_summary = summarize_factor(data, "Job Level")
    growth_summary = (
        data.groupby(["Leadership Opportunities", "Innovation Opportunities"], observed=True)["AttritionFlag"]
        .mean()
        .reset_index()
    )

    q8_col1, q8_col2 = st.columns(2)
    with q8_col1:
        fig = px.line(
            promotions_summary,
            x="Number of Promotions",
            y="AttritionFlag",
            markers=True,
            title="Attrition by number of promotions",
            color_discrete_sequence=["#dc2626"],
        )
        fig.update_layout(yaxis_title="Attrition rate", xaxis_title="Number of promotions")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with q8_col2:
        fig = make_bar_chart(job_level_summary.sort_values("AttritionRate", ascending=False), "Job Level", "AttritionRatePct", "Attrition by job level")
        st.plotly_chart(fig, use_container_width=True)

    opp_fig = go.Figure()
    opp_fig.add_trace(
        go.Bar(
            name="Leadership opportunities",
            x=growth_summary["Leadership Opportunities"],
            y=growth_summary["AttritionFlag"],
            marker_color="#0f766e",
        )
    )
    opp_fig.add_trace(
        go.Bar(
            name="Innovation opportunities",
            x=growth_summary["Innovation Opportunities"],
            y=growth_summary["AttritionFlag"],
            marker_color="#2563eb",
        )
    )
    opp_fig.update_layout(
        barmode="group",
        title="Attrition impact of growth and opportunity signals",
        yaxis_title="Attrition rate",
        xaxis_title="Opportunity access",
        yaxis_tickformat=".0%",
    )
    st.plotly_chart(opp_fig, use_container_width=True)

    st.markdown(
        """
        **Answer:** The data supports a growth-stagnation story. Attrition is much higher when promotions are rare, job level is low, and leadership or innovation opportunities are missing.

        **Recommendation:** build clear promotion paths, create lateral mobility, and add visible project ownership so employees do not feel stuck.
        """
    )

with tabs[8]:
    st.subheader("Q9 · The highest-risk profile")
    best_profile, profile_table = find_high_risk_profile(data, RISK_PROFILE_COLUMNS, min_count=500)

    if best_profile is None:
        st.error("Could not build a high-risk profile from the available data.")
    else:
        profile_rate = float(best_profile["AttritionRate"])
        profile_lift = float(best_profile["Lift"])
        profile_count = int(best_profile["Employees"])

        q9_col1, q9_col2 = st.columns([1, 1])
        with q9_col1:
            comparison = pd.DataFrame(
                {
                    "Group": ["Company average", "Highest-risk profile"],
                    "Attrition rate": [baseline_rate, profile_rate],
                }
            )
            fig = px.bar(comparison, x="Group", y="Attrition rate", text="Attrition rate", title="Company average vs highest-risk profile")
            fig.update_traces(texttemplate="%{y:.1%}", textposition="outside", marker_color=["#94a3b8", "#dc2626"])
            fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with q9_col2:
            st.metric("Profile attrition", pct(profile_rate))
            st.metric("Above company average", fmt_points(profile_lift))
            st.metric("Matching employees", f"{profile_count:,}")

        st.markdown(
            f"""
            **Answer:** The highest-risk profile I found with a meaningful sample is: **{best_profile['Profile']}**.

            That group has {pct(profile_rate)} attrition, which is {fmt_points(profile_lift)} above the company average, and it includes {profile_count:,} employees. That is large enough to matter operationally, not just statistically.
            """
        )
        # st.dataframe(
        #     profile_table.assign(
        #         AttritionRate=profile_table["AttritionRate"].map(pct),
        #         Lift=profile_table["Lift"].map(fmt_points),
        #     ),
        #     use_container_width=True,
        #     hide_index=True,
        # )
        st.info("Recommendation: if leadership wants a single targetable segment, start with entry-level single employees reporting poor work-life balance, then add overtime and remote-access fixes.")

with tabs[9]:
    st.subheader("Q10 · What moves the needle")
    driver_rows = []
    for column in ACTIONABLE_COLUMNS:
        driver = best_actionable_driver(data, column)
        if driver is not None:
            driver_rows.append(driver)

    drivers = pd.DataFrame(driver_rows).sort_values("lift", ascending=False).reset_index(drop=True)
    top_three = drivers.head(3).copy()
    top_three["lift_pct"] = top_three["lift"] * 100

    q10_col1, q10_col2 = st.columns([1, 1])
    with q10_col1:
        fig = px.bar(
            top_three,
            x="driver",
            y="lift_pct",
            color="driver",
            text="lift_pct",
            title="Top 3 actionable drivers above baseline",
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        fig.update_traces(texttemplate="%{y:.1f} pts", textposition="outside")
        fig.update_layout(yaxis_title="Lift above company average (pts)", xaxis_title="", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    # with q10_col2:
    #     st.dataframe(
    #         top_three.assign(
    #             rate=top_three["rate"].map(pct),
    #             lift=top_three["lift"].map(fmt_points),
    #         )[["driver", "value", "rate", "lift", "employees"]],
    #         use_container_width=True,
    #         hide_index=True,
    #     )

    if not top_three.empty:
        top_driver = top_three.iloc[0]
        impact_group = data[data[top_driver["driver"]] == top_driver["value"]]
        estimated_prevented_exits = int(round((float(top_driver["rate"]) - baseline_rate) * len(impact_group)))

        st.markdown(
            f"""
            **Answer:** Among actionable levers, the biggest mover is **{top_driver['driver']} = {top_driver['value']}**, followed by **{top_three.iloc[1]['driver']} = {top_three.iloc[1]['value']}** and **{top_three.iloc[2]['driver']} = {top_three.iloc[2]['value']}**.

            **Why #1:** it has the largest attrition lift above the baseline and the clearest operational meaning. If HR could reduce this segment back to company-average attrition, the rough upside is about **{estimated_prevented_exits:,} fewer exits** in this dataset.

            **Recommendation:** fix workload and work-life balance first. That is the most direct lever with the highest estimated payoff, and it also helps the overtime and remote-work story at the same time.
            """
        )
    else:
        st.warning("Not enough data to rank actionable drivers.")

st.divider()
st.subheader("Business recommendations")
st.markdown(
    """
1. Start with Technology because it has the largest absolute number of leavers; use Education as the secondary risk review because it has the highest rate.
2. Cut chronic overtime by rebalancing workload, staffing, and manager accountability.
3. Expand remote or hybrid access where the role allows it, because remote employees leave much less often, but keep the conclusion honest about the small remote share.
4. Tighten pay floors in Entry and early Mid roles; pay matters there, but it flattens at higher levels, so use growth and recognition for retention beyond pay.
5. Prioritize onboarding and the first years at the company, because attrition is highest early in tenure.
6. Train managers to watch for poor work-life balance plus low satisfaction as an early-warning sign.
7. Build retention programs for single, early-career employees with faster career visibility and better support.
8. Create more promotions, lateral mobility, leadership chances, and innovation projects so employees do not feel stuck.
9. Target the highest-risk profile first: entry-level, single employees with poor work-life balance, then add overtime and remote-access fixes.
10. If only one lever can move next quarter, make work-life balance the priority because it has the largest actionable lift.
"""
)

st.download_button(
    "Download combined data",
    data.to_csv(index=False),
    "combined_hr_data.csv",
    "text/csv",
)