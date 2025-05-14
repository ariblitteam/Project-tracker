import openai
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

openai.api_key = st.secrets["openai"]["api_key"]


@st.cache_data(show_spinner="Loading data...")
def fetch_project_data():
    api_url = "https://projecttracker.anandrathiinsurance.com/getallprojects1/"
    response = requests.get(api_url)
    data = response.json()

    if isinstance(data, dict) and "data" in data:
        df = pd.DataFrame(data["data"])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        st.error("Unexpected data format received from API.")
        st.stop()

    date_columns = [
        "requirementReceivedDate", "requirementGatheringActualEndDate",
        "storyWrittingStorySignOffDate", "estimaionActualEndDate",
        "developmentPlanStartDate", "developmentActualEndDate",
        "sitActualEndDate", "uatActualEndDate"
    ]

    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace("-", "/", regex=False)
            df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
    return df

def get_tat_threshold(size, stage):
    thresholds = {
        "Requirement Gathering": {"Small": 4, "Medium": 8, "Large": 10},
        "User Story Writing": {"Small": 5, "Medium": 9, "Large": 12},
        "Estimation": {"Small": 4, "Medium": 8, "Large": 10},
        "Development": {"Small": 7, "Medium": 15, "Large": 65},
        "SIT": {"Small": 3, "Medium": 7, "Large": 20},
        "UAT": {"Small": 3, "Medium": 6, "Large": 12},
    }
    return thresholds.get(stage, {}).get(size, 999)

def calc_days(start, end, mode="Working Days"):
    if pd.isna(start) or pd.isna(end):
        return "Pending"
    elif end < start:
        return "Invalid Date Range"
    if mode == "Working Days":
        return np.busday_count(start.date(), end.date() + pd.Timedelta(days=1))
    else:
        return (end - start).days

def calculate_tat(df, mode="Working Days"):
    df['requirement_tat'] = df.apply(lambda x: calc_days(x['requirementReceivedDate'], x['requirementGatheringActualEndDate'], mode), axis=1)
    df['user_story_tat'] = df.apply(lambda x: calc_days(x['requirementGatheringActualEndDate'], x['storyWrittingStorySignOffDate'], mode), axis=1)
    df['estimation_tat'] = df.apply(lambda x: calc_days(x['storyWrittingStorySignOffDate'], x['estimaionActualEndDate'], mode), axis=1)
    df['development_tat'] = df.apply(lambda x: calc_days(x['developmentPlanStartDate'], x['developmentActualEndDate'], mode), axis=1)
    df['sit_tat'] = df.apply(lambda x: calc_days(x['developmentActualEndDate'], x['sitActualEndDate'], mode), axis=1)
    df['uat_tat'] = df.apply(lambda x: calc_days(x['sitActualEndDate'], x['uatActualEndDate'], mode), axis=1)
    return df

def add_expected_tat(df):
    stages = {
        'requirement_expected_tat': 'Requirement Gathering',
        'user_story_expected_tat': 'User Story Writing',
        'estimation_expected_tat': 'Estimation',
        'development_expected_tat': 'Development',
        'sit_expected_tat': 'SIT',
        'uat_expected_tat': 'UAT'
    }
    for col, stage in stages.items():
        df[col] = df['projectSize'].apply(lambda size: get_tat_threshold(size, stage))
    return df

def add_tat_status(df):
    for index, row in df.iterrows():
        size = row['projectSize']
        for stage_key, stage_col in {
            "Requirement Gathering": 'requirement_tat',
            "User Story Writing": 'user_story_tat',
            "Estimation": 'estimation_tat',
            "Development": 'development_tat',
            "SIT": 'sit_tat',
            "UAT": 'uat_tat'
        }.items():
            tat_value = row[stage_col]
            if tat_value == "Pending":
                df.at[index, f'{stage_col}_status'] = "Pending"
            elif tat_value == "Invalid Date Range":
                df.at[index, f'{stage_col}_status'] = "Invalid Date Range"
            else:
                df.at[index, f'{stage_col}_status'] = "In TAT" if tat_value <= get_tat_threshold(size, stage_key) else "Out TAT"
    return df

def highlight_tat(val):
    if val == 'In TAT':
        return 'background-color: lightgreen'
    elif val == 'Out TAT':
        return 'background-color: salmon'
    elif val == 'Pending':
        return 'background-color: darkgrey'
    elif val == 'Invalid Date Range':
        return 'background-color: orange'
    return ''

def calculate_deviation(df):
    deviations = []
    stage_mapping = {
        'Requirement Gathering': ('requirement_tat', 'requirement_expected_tat'),
        'User Story Writing': ('user_story_tat', 'user_story_expected_tat'),
        'Estimation': ('estimation_tat', 'estimation_expected_tat'),
        'Development': ('development_tat', 'development_expected_tat'),
        'SIT': ('sit_tat', 'sit_expected_tat'),
        'UAT': ('uat_tat', 'uat_expected_tat')
    }

    if 'baName' not in df.columns:
        raise ValueError("The input DataFrame must include a 'BA Name' column.")

    for ba_name, ba_group in df.groupby('baName'):
        for stage_name, (tat_col, expected_col) in stage_mapping.items():
            # Coerce actual and expected TAT values to numeric
            actual_tat = pd.to_numeric(ba_group[tat_col], errors='coerce')
            expected_tat = pd.to_numeric(ba_group[expected_col], errors='coerce')

            diffs = actual_tat - expected_tat
            valid_diffs = diffs.dropna()

            max_dev = int(valid_diffs.max()) if not valid_diffs.empty else 0
            avg_dev = int(round(valid_diffs.mean())) if not valid_diffs.empty else 0
            has_invalid = 'Yes' if (ba_group[tat_col] == 'Invalid Date Range').any() else 'No'

            deviations.append({
                'BA Name': ba_name,
                'Stage': stage_name,
                'Max Deviation': max_dev,
                'Average Deviation': avg_dev,
                'Has Invalid Dates': has_invalid
            })

    return pd.DataFrame(deviations)




def highlight_invalid_rows(row):
    return ['background-color: lightgrey' if row['Has Invalid Dates'] == 'Yes' else '' for _ in row]

def main():
    st.set_page_config(page_title='Project SOP TAT Dashboard', layout="wide")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_email = ""

    st.markdown("<h3 style='text-align: left; font-weight: bold;'>Project SOP TAT Dashboard</h3>", unsafe_allow_html=True)

    if not st.session_state.logged_in:
        email = st.text_input("Enter Email ID to Login")
        if st.button("Login"):
            if email.strip().lower().endswith("@rathi.com"):
                st.session_state.logged_in = True
                st.session_state.user_email = email.strip().lower()
                st.rerun()
            else:
                st.error("Please use your official '@rathi.com' email address.")
        return

    with st.sidebar:
        if 'user_email' in st.session_state:
            st.success(f"Logged in as: {st.session_state.user_email}")
        if st.button("Logout", help="Click here to logout"):
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.rerun()

        tat_mode = st.radio("TAT Calculation Mode", options=["Working Days", "Calendar Days"], index=0)

    df = fetch_project_data()
    df = calculate_tat(df, mode=tat_mode)
    df = add_expected_tat(df)  # FIX: moved before add_tat_status
    df = add_tat_status(df)

    df = df[df['requirementReceivedDate'] >= pd.to_datetime('2024-01-01')]
    df['Year'] = df['requirementReceivedDate'].dt.year
    df['Month'] = df['requirementReceivedDate'].dt.strftime('%B')

    date_cols = [col for col in df.columns if "date" in col.lower()]
    for col in date_cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime('%d/%m/%Y')

    #SubStatus_names = df['projectSubStatus'].dropna().unique().tolist()

# Multiselect without default selection
    selected_bas = st.sidebar.multiselect("Select Project SubStatus", options=df['projectSubStatus'].dropna().unique().tolist(), default=[])

    selected_size = st.sidebar.selectbox("Select Project Size", options=["All", "Small", "Medium", "Large"])
    #selected_ba = st.sidebar.selectbox("Select BA Name", options=["All"] + sorted(df['baName'].dropna().unique()))
    selected_status = st.sidebar.selectbox("Select BA Name", options=["All"] + sorted(df['baName'].dropna().unique()))
    selected_year = st.sidebar.selectbox("Select Year", options=["All"] + sorted(df['Year'].dropna().unique(), reverse=True))
    selected_month = st.sidebar.selectbox("Select Month", options=["All"] + list(pd.date_range(start='2024-01-01', periods=12, freq='MS').strftime('%B')))

    
    filtered_df = df.copy()
    if selected_size != "All":
        filtered_df = filtered_df[filtered_df['projectSize'] == selected_size]
    if selected_bas:
        filtered_df = df[df['projectSubStatus'].isin(selected_bas)]        
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df['baName'] == selected_status]
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df['Year'] == selected_year]
    if selected_month != "All":
        filtered_df = filtered_df[filtered_df['Month'] == selected_month]

    #tab1, tab2 = st.tabs(["Detailed View", "Charts View"])
    tab1, tab2, tab3 = st.tabs(["Detailed View", "Charts View", "AI Assistant"])

    total_projects = filtered_df['projectId'].nunique()
    stage_cols = ['requirement_tat_status', 'user_story_tat_status', 'estimation_tat_status',
                  'development_tat_status', 'sit_tat_status', 'uat_tat_status']
    intat_count = (filtered_df[stage_cols] == 'In TAT').sum().sum()
    outtat_count = (filtered_df[stage_cols] == 'Out TAT').sum().sum()
    total_stages = intat_count + outtat_count
    pending_count = (filtered_df[stage_cols] == 'Pending').sum().sum()
    overall_tat_percentage = int(round((intat_count / total_stages) * 100)) if total_stages > 0 else 0

    with tab1:
        st.markdown(f"<div style='text-align: left; color: gray; font-size: 14px;'>TAT Mode: <strong>{tat_mode}</strong></div>", unsafe_allow_html=True)
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total Projects", total_projects)
        col2.metric("Total Stages", total_stages)
        col3.metric("Total In-TAT", intat_count)
        col4.metric("Total Out-TAT", outtat_count)
        col5.metric("Overall In-TAT %", f"{overall_tat_percentage}%")
        col6.metric("Total Pending", pending_count)

        st.markdown("---")
        st.markdown("<h6 style='text-align: center; font-weight: bold;'>BA-wise Summary Report</h6>", unsafe_allow_html=True)

        def ba_summary(df):
            stage_cols = {
                'requirement_tat_status': 'Requirement',
                'user_story_tat_status': 'User Story',
                'estimation_tat_status': 'Estimation',
                'development_tat_status': 'Development',
                'sit_tat_status': 'SIT',
                'uat_tat_status': 'UAT'
            }

            summary_list = []
            for ba, group in df.groupby('baName'):
                ba_summary = {"BA Name": ba, "Total Projects": group['projectId'].nunique()}
                for col, stage in stage_cols.items():
                    ba_summary[f'{stage} In TAT'] = (group[col] == 'In TAT').sum()
                    ba_summary[f'{stage} Out TAT'] = (group[col] == 'Out TAT').sum()
                    total_stage = ba_summary[f'{stage} In TAT'] + ba_summary[f'{stage} Out TAT']
                    ba_summary[f'{stage} In-TAT %'] = round((ba_summary[f'{stage} In TAT'] / total_stage * 100), 1) if total_stage > 0 else 0
                summary_list.append(ba_summary)

            return pd.DataFrame(summary_list)

        ba_kpi_df = ba_summary(filtered_df)
        st.dataframe(ba_kpi_df, use_container_width=True)

        st.markdown("---")
        st.markdown("<h6 style='text-align: center; font-weight: bold;'>SOP TAT Detailed Report</h6>", unsafe_allow_html=True)

        display_cols = [
            'projectId','projectName', 'baName', 'projectSubStatus', 'projectSize',
            'requirementReceivedDate', 'requirementGatheringActualEndDate',
            'requirement_expected_tat', 'requirement_tat', 'requirement_tat_status',
            'storyWrittingStorySignOffDate', 'user_story_expected_tat', 'user_story_tat', 'user_story_tat_status',
            'estimaionActualEndDate', 'estimation_expected_tat', 'estimation_tat', 'estimation_tat_status',
            'developmentPlanStartDate', 'developmentActualEndDate',
            'development_expected_tat', 'development_tat', 'development_tat_status',
            'sitActualEndDate', 'sit_expected_tat', 'sit_tat', 'sit_tat_status',
            'uatActualEndDate', 'uat_expected_tat', 'uat_tat', 'uat_tat_status'
        ]
        styled_df = filtered_df[display_cols].style.applymap(highlight_tat, subset=stage_cols)
        st.dataframe(styled_df, use_container_width=True)

        st.markdown("<h6 style='text-align: center; font-weight: bold;'>BA-wise Stages Deviation Metrics</h6>", unsafe_allow_html=True)
        deviation_df = calculate_deviation(filtered_df)
        st.dataframe(deviation_df.style.apply(highlight_invalid_rows, axis=1), use_container_width=True)

        st.markdown("<h6 style='text-align: center; font-weight: bold;'>Projects with Invalid Date Ranges</h6>", unsafe_allow_html=True)
        invalid_mask = (filtered_df[stage_cols] == 'Invalid Date Range').any(axis=1)
        invalid_records = filtered_df[invalid_mask]
        if not invalid_records.empty:
            st.dataframe(invalid_records[[
                'projectId', 'baName', 'projectSubStatus', 'projectSize',
                'requirementReceivedDate', 'requirementGatheringActualEndDate', 'requirement_tat',
                'storyWrittingStorySignOffDate', 'user_story_tat',
                'estimaionActualEndDate', 'estimation_tat',
                'developmentPlanStartDate', 'developmentActualEndDate', 'development_tat',
                'sitActualEndDate', 'sit_tat',
                'uatActualEndDate', 'uat_tat'
            ]], use_container_width=True)
        else:
            st.success("No invalid date ranges found.")

#import plotly.graph_objects as go

    with tab2:
    # Stage map
        stage_map = {
        'requirement_tat_status': 'Requirement Gathering',
        'user_story_tat_status': 'User Story Writing',
        'estimation_tat_status': 'Estimation',
        'development_tat_status': 'Development',
        'sit_tat_status': 'SIT',
        'uat_tat_status': 'UAT',
        }

    # Melt and clean
        melted_df = filtered_df.melt(
        id_vars=['baName'],
        value_vars=stage_map.keys(),
        var_name='Stage',
        value_name='TAT_Status'
        )
        melted_df['Stage'] = melted_df['Stage'].map(stage_map)
        melted_df['TAT_Status'] = melted_df['TAT_Status'].str.strip().str.lower()

    # Filters
        status_filter = st.multiselect(
        "Filter by TAT Status",
        options=['In TAT', 'Out of TAT'],
        default=['In TAT']
        )
        stage_filter = st.multiselect(
        "Filter by Stage",
        options=list(stage_map.values()),
        default=list(stage_map.values())
        )

        status_filter_lower = [s.lower() for s in status_filter]

    # Filter
        filtered_df_stage = melted_df[
        (melted_df['TAT_Status'].isin(status_filter_lower)) &
        (melted_df['Stage'].isin(stage_filter))
        ]

    # Total tasks per stage (for %)
        total_per_stage = (
        melted_df[melted_df['Stage'].isin(stage_filter)]
        .groupby('Stage')
        .size()
        .reset_index(name='Total')
        )

    # In-TAT count per BA per stage
        summary_df = (
        filtered_df_stage.groupby(['Stage', 'baName'])
        .size()
        .reset_index(name='InTAT_Count')
        )

    # Merge total & compute %
        summary_df = summary_df.merge(total_per_stage, on='Stage', how='left')
        summary_df['InTAT_Percent'] = (summary_df['InTAT_Count'] / summary_df['Total']) * 100
        summary_df['InTAT_Percent_Label'] = summary_df['InTAT_Percent'].round(1).astype(str) + '%'

    # Get unique stages
        stage_order = summary_df['Stage'].unique().tolist()

    # Create sorted stacked bars manually
        fig = go.Figure()

        for stage in stage_order:
            stage_data = summary_df[summary_df['Stage'] == stage].copy()
            stage_data = stage_data.sort_values(by='InTAT_Percent', ascending=False)

            for _, row in stage_data.iterrows():
                fig.add_trace(go.Bar(
                x=[row['InTAT_Percent']],
                y=[stage],
                name=row['baName'],
                orientation='h',
                text=row['InTAT_Percent_Label'],
                textposition='inside',
                insidetextanchor='start',
                hovertemplate=f"Stage: {stage}<br>BA: {row['baName']}<br>In-TAT %: {row['InTAT_Percent']:.1f}%<extra></extra>",
                legendgroup=row['baName'],
                showlegend=(stage == stage_order[0])  # Only show legend once per BA
                ))

    # Layout
        fig.update_layout(
        barmode='stack',
        title='In-TAT % by Stage',
        xaxis_title='In-TAT %',
        yaxis_title='Stage',
        height=500,
        width=1000,
        margin=dict(t=30, l=10, r=40, b=90),
        legend_title_text='Business Analyst',
        plot_bgcolor='white',
        font=dict(size=13),
        )

        st.plotly_chart(fig, use_container_width=False)





 
        
        st.markdown("---")
        st.markdown(f"<h6 style='text-align: center; font-weight: bold;'>Data Dump</h6>", unsafe_allow_html=True)
        st.dataframe(filtered_df, use_container_width=True, height=min(460, len(filtered_df) * 40))

    with tab3:
        st.markdown("### 🤖 AI Assistant Chatbot")
        st.info("Ask questions about Projects, TAT, Deviations, or Project Status.")

    # Initialize session state for chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [{"role": "system", "content": (
                "You are a helpful assistant for analyzing Project SOP TAT data. "
                "Answer questions using the data summary provided. Focus on metrics like TAT, deviations, project status, and BA performance."
            )}]

    # Summarize the current filtered_df for context (limit size for token budget)
        if "data_context" not in st.session_state:
        # Only take necessary columns and top 5 rows
            context_df = filtered_df[[
                'projectId', 'projectSubStatus', 'projectSize', 'baName',
                'requirement_tat', 'user_story_tat', 'estimation_tat',
                'development_tat', 'sit_tat', 'uat_tat'
            ]].head(5)
            st.session_state.data_context = context_df.to_markdown(index=False)

            st.session_state.chat_history.append({
                "role": "system",
                "content": f"Here is a summary of the current project data:\n{st.session_state.data_context}"
            })

    # Display past messages
        for msg in st.session_state.chat_history[1:]:  # skip initial system message
            st.chat_message(msg["role"]).write(msg["content"])

    # Chat input
        user_input = st.chat_input("Ask something about the dashboard...")
        if user_input:
            st.chat_message("user").write(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with st.spinner("Thinking..."):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" based on your access
                        #messages=st.session_state.chat_history,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Hello!"}
                        ],
                        max_tokens=100
                        #temperature=0.5
                    )
                    reply = response.choices[0].message.content
                    

                except Exception as e:
                    reply = f"An error occurred: {e}"

            st.chat_message("assistant").write(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
