import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="H-1B Analysis Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div > select {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process the H-1B data"""
    try:
        # Load the Excel files
        lca = pd.read_excel('LCA_Disclosure.xlsx', sheet_name='LCA_Disclosure_Data_FY2025_Q2')
        # perm = pd.read_excel('PERM_Disclosure.xlsx', sheet_name='PERM_FY2025_Q2_old')  # Uncomment if needed
        
        # Filter LCA data to keep only the relevant columns
        lca_filtered = lca[['CASE_STATUS', 'VISA_CLASS', 'JOB_TITLE','SOC_CODE', 'SOC_TITLE',
                           'FULL_TIME_POSITION','EMPLOYER_NAME','EMPLOYER_CITY','EMPLOYER_STATE',
                           'EMPLOYER_COUNTRY', 'NAICS_CODE','WAGE_UNIT_OF_PAY','PREVAILING_WAGE',
                           'PW_UNIT_OF_PAY', 'SUPPORT_H1B']].copy()
        
        return lca_filtered
    except FileNotFoundError:
        st.warning("Data files not found. Using sample data for demonstration.")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration purposes"""
    np.random.seed(42)
    
    # Sample data generation
    employers = ['AMAZON.COM SERVICES LLC', 'MICROSOFT CORPORATION', 'GOOGLE LLC', 'APPLE INC', 
                'META PLATFORMS INC', 'TESLA INC', 'NETFLIX INC', 'UBER TECHNOLOGIES INC'] * 100
    
    states = ['CA', 'WA', 'TX', 'NY', 'MA', 'IL', 'FL', 'GA', 'NC', 'VA'] * 80
    cities = ['SAN FRANCISCO', 'SEATTLE', 'AUSTIN', 'NEW YORK', 'BOSTON', 'CHICAGO', 
              'MIAMI', 'ATLANTA', 'CHARLOTTE', 'RICHMOND'] * 80
    
    # Education NAICS codes (611xxx)
    education_naics = [611110, 611210, 611310, 611420, 611519, 611620, 611691, 611699]
    # Research NAICS codes
    research_naics = [541711, 541712]
    # Hospital NAICS codes (622xxx)
    hospital_naics = [622110, 622210, 622310]
    # Government research
    gov_naics = [927110, 927140]
    # Other NAICS codes
    other_naics = [541511, 541512, 518210, 334111, 336411]
    
    all_naics = education_naics + research_naics + hospital_naics + gov_naics + other_naics
    
    sample_data = pd.DataFrame({
        'CASE_STATUS': np.random.choice(['Certified', 'Denied', 'Withdrawn'], 800, p=[0.85, 0.10, 0.05]),
        'VISA_CLASS': ['H-1B'] * 800,
        'JOB_TITLE': np.random.choice(['SOFTWARE ENGINEER', 'DATA SCIENTIST', 'RESEARCH SCIENTIST', 
                                      'PROFESSOR', 'ANALYST', 'MANAGER'], 800),
        'SOC_CODE': np.random.choice(['15-1132', '15-1133', '19-1042', '25-1022'], 800),
        'SOC_TITLE': np.random.choice(['SOFTWARE DEVELOPERS', 'SOFTWARE ENGINEERS', 
                                      'MEDICAL SCIENTISTS', 'PROFESSORS'], 800),
        'FULL_TIME_POSITION': np.random.choice(['Y', 'N'], 800, p=[0.9, 0.1]),
        'EMPLOYER_NAME': np.random.choice(employers, 800),
        'EMPLOYER_CITY': np.random.choice(cities, 800),
        'EMPLOYER_STATE': np.random.choice(states, 800),
        'EMPLOYER_COUNTRY': ['UNITED STATES OF AMERICA'] * 800,
        'NAICS_CODE': np.random.choice(all_naics, 800),
        'WAGE_UNIT_OF_PAY': np.random.choice(['Year', 'Hour'], 800, p=[0.8, 0.2]),
        'PREVAILING_WAGE': np.random.uniform(50000, 200000, 800),
        'PW_UNIT_OF_PAY': np.random.choice(['Year', 'Hour'], 800, p=[0.8, 0.2]),
        'SUPPORT_H1B': np.random.choice(['Y', 'N'], 800, p=[0.95, 0.05])
    })
    
    return sample_data

def identify_cap_exempt_institutions(lca_filtered):
    """Identify cap-exempt institutions based on NAICS codes"""
    lca_filtered = lca_filtered.copy()
    lca_filtered['NAICS_CODE_str'] = lca_filtered['NAICS_CODE'].astype(str)
    
    cap_exempt_mask = (
        # Education institutions
        lca_filtered['NAICS_CODE_str'].str.startswith('611') |
        # Research institutions
        (lca_filtered['NAICS_CODE'] == 541711) |
        # Research and development in physical, engineering sciences
        (lca_filtered['NAICS_CODE'] == 541712) |
        # Hospitals (when affiliated with a university)
        lca_filtered['NAICS_CODE_str'].str.startswith('622') |
        # Space research and technology
        (lca_filtered['NAICS_CODE'] == 927110) |
        # Government institutions
        (lca_filtered['NAICS_CODE_str'].str.startswith('9271'))
    )
    
    cap_exempt_institutions = lca_filtered[cap_exempt_mask]
    return cap_exempt_institutions

def categorize_naics(code):
    """Categorize NAICS codes into institution types"""
    code_str = str(code)
    if code_str.startswith('611'):
        return 'Educational Services'
    elif code == 541711:
        return 'Biotech R&D'
    elif code == 541712:
        return 'Physical/Engineering R&D'
    elif code_str.startswith('622'):
        return 'Hospitals'
    elif code == 927110:
        return 'Space Research'
    elif code_str.startswith('9271'):
        return 'Government Research'
    else:
        return 'Other'

def create_employer_summary(lca_filtered):
    """Create employer summary with aggregated statistics"""
    employer_summary = lca_filtered.groupby(['EMPLOYER_NAME']).agg({
        'CASE_STATUS': ['count', lambda x: (x == 'Certified').sum(), lambda x: (x == 'Denied').sum()],
        'EMPLOYER_CITY': 'first',
        'EMPLOYER_STATE': 'first'
    }).reset_index()
    
    employer_summary.columns = ['Employer_Name', 'Total_Applications', 'Certified_Applications', 
                               'Denied_Applications', 'City', 'State']
    employer_summary['Certification_Rate'] = (
        employer_summary['Certified_Applications'] / employer_summary['Total_Applications'] * 100
    ).round(2)
    
    employer_summary['State'] = employer_summary['State'].fillna('UNKNOWN')
    employer_summary['City'] = employer_summary['City'].fillna('UNKNOWN')
    employer_summary = employer_summary.dropna(subset=['Employer_Name'])
    employer_summary['State'] = employer_summary['State'].astype(str).str.strip().str.upper()
    employer_summary['City'] = employer_summary['City'].astype(str).str.strip().str.title()
    employer_summary.loc[employer_summary['State'] == '', 'State'] = 'UNKNOWN'
    employer_summary.loc[employer_summary['City'] == '', 'City'] = 'UNKNOWN'
    employer_summary = employer_summary.sort_values('Total_Applications', ascending=False)
    
    return employer_summary

def create_state_summary(data):
    """Create state-level summary statistics"""
    summary = data.groupby('State').agg({
        'Total_Applications': 'sum',
        'Certified_Applications': 'sum', 
        'Denied_Applications': 'sum',
        'Employer_Name': 'count'
    }).reset_index()
    summary.columns = ['State', 'Total_Applications', 'Certified_Applications', 
                      'Denied_Applications', 'Employer_Count']
    summary['Certification_Rate'] = (
        summary['Certified_Applications'] / summary['Total_Applications'] * 100
    ).round(2)
    return summary.sort_values('Total_Applications', ascending=False)

def filter_data(data, states, top_n):
    """Filter data based on selected states"""
    if 'ALL STATES' in states or len(states) == 0:
        filtered_data = data.copy()
    else:
        filtered_data = data[data['State'].isin(states)].copy()
    return filtered_data.head(top_n)

def create_bar_chart(data, title_suffix=""):
    """Create horizontal bar chart"""
    fig = px.bar(
        data, 
        x='Total_Applications', 
        y='Employer_Name',
        orientation='h',
        title=f'Top {len(data)} Employers by H-1B Applications {title_suffix}',
        labels={'Total_Applications': 'Total Applications', 'Employer_Name': 'Employer'},
        color='Total_Applications',
        color_continuous_scale='Blues',
        text='Total_Applications',
        hover_data={'State': True, 'City': True, 'Certification_Rate': ':.2f'}
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        height=max(400, len(data) * 25),
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        margin=dict(l=250)
    )
    return fig

def create_stacked_chart(data, title_suffix=""):
    """Create stacked bar chart for case status"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Certified',
        x=data['Employer_Name'],
        y=data['Certified_Applications'],
        marker_color='#27ae60',
        hovertemplate='<b>%{x}</b><br>Certified: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Denied',
        x=data['Employer_Name'],
        y=data['Denied_Applications'],
        marker_color='#e74c3c',
        hovertemplate='<b>%{x}</b><br>Denied: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        barmode='stack',
        title=f'H-1B Status Distribution-VisaLens {title_suffix}',
        xaxis_title='Employer',
        yaxis_title='Number of Applications',
        xaxis_tickangle=-45,
        height=600,
        margin=dict(b=120)
    )
    return fig

def create_cert_rate_chart(data, title_suffix=""):
    """Create certification rate chart"""
    fig = px.bar(
        data,
        x='Certification_Rate',
        y='Employer_Name',
        orientation='h',
        title=f'H-1B Certification Rates {title_suffix}',
        labels={'Certification_Rate': 'Certification Rate (%)', 'Employer_Name': 'Employer'},
        color='Certification_Rate',
        color_continuous_scale='RdYlGn',
        text='Certification_Rate',
        hover_data={'State': True, 'Total_Applications': True}
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        height=max(400, len(data) * 25),
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=250)
    )
    return fig

def create_cap_exempt_stacked_chart(cap_exempt_institutions):
    """Create stacked bar chart for cap-exempt institutions by state and type"""
    # Add category column
    cap_exempt_institutions['Institution_Type'] = cap_exempt_institutions['NAICS_CODE'].apply(categorize_naics)
    
    # Create crosstab for stacked bar chart
    state_type_counts = cap_exempt_institutions.groupby(['EMPLOYER_STATE', 'Institution_Type']).size().reset_index(name='Count')
    
    # Get top 15 states by total count
    top_states = cap_exempt_institutions['EMPLOYER_STATE'].value_counts().head(15).index.tolist()
    state_type_counts_filtered = state_type_counts[state_type_counts['EMPLOYER_STATE'].isin(top_states)]
    
    # Create stacked bar chart
    fig = px.bar(
        state_type_counts_filtered,
        x='EMPLOYER_STATE',
        y='Count',
        color='Institution_Type',
        title='Cap-Exempt H-1B Applications by State and Institution Type (Top 15 States)',
        labels={'Count': 'Number of Applications', 'EMPLOYER_STATE': 'State'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=600,
        legend_title='Institution Type'
    )
    
    return fig

def create_cap_exempt_dropdown_chart(cap_exempt_institutions, selected_state):
    """Create dropdown chart for cap-exempt institutions by state"""
    def get_top_institutions_by_state(state, top_n=20):
        state_data = cap_exempt_institutions[cap_exempt_institutions['EMPLOYER_STATE'] == state]
        institution_counts = state_data['EMPLOYER_NAME'].value_counts().head(top_n).reset_index()
        institution_counts.columns = ['Institution_Name', 'Count']
        institution_counts['State'] = state
        return institution_counts
    
    # Get data for selected state
    state_data = get_top_institutions_by_state(selected_state, 20)
    
    if len(state_data) == 0:
        return None
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=state_data['Count'],
        y=state_data['Institution_Name'],
        orientation='h',
        name=selected_state,
        marker_color='skyblue',
        hovertemplate='<b>%{y}</b><br>Applications: %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Top 20 Cap-Exempt Institutions in {selected_state}',
        xaxis_title='Number of H-1B Applications',
        yaxis_title='Institution',
        yaxis={'categoryorder': 'total ascending'}, 
        height=800,
        margin=dict(l=300)
    )
    
    return fig

def main():
    # Title
    st.markdown('<h1 class="main-header">🏢 H-1B Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading and processing data...'):
        lca_filtered = load_data()
        cap_exempt_institutions = identify_cap_exempt_institutions(lca_filtered)
        employer_summary = create_employer_summary(lca_filtered)
        state_summary = create_state_summary(employer_summary)
    
    # Sidebar
    st.sidebar.header("📊 Analysis Controls")
    
    # Analysis section selection
    analysis_section = st.sidebar.radio(
        "Select Analysis Section:",
        ["General H-1B Analysis", "Cap-Exempt Institutions"]
    )
    
    if analysis_section == "General H-1B Analysis":
        # General H-1B Analysis Section
        st.markdown('<h2 class="sub-header">📈 General H-1B Employer Analysis</h2>', unsafe_allow_html=True)
        
        # Analysis type selection
        analysis_type = st.sidebar.radio(
            "Analysis Type:",
            ["Multi-State Analysis", "Single State Deep Dive", "State Overview"]
        )
        
        # Get available states
        available_states = ['ALL STATES'] + sorted(employer_summary['State'].unique().tolist())
        
        # State selection based on analysis type
        if analysis_type == "Single State Deep Dive":
            selected_states = [st.sidebar.selectbox(
                "Select State:",
                options=[s for s in available_states if s != 'ALL STATES'],
                index=0
            )]
        else:
            selected_states = st.sidebar.multiselect(
                "Select States:",
                options=available_states,
                default=['ALL STATES']
            )
        
        # Top N selection
        top_n = st.sidebar.slider(
            "Top N Employers:",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )
        
        # Chart type selection
        chart_type = st.sidebar.selectbox(
            "Chart Type:",
            ["Total Applications Bar", "Stacked Case Status", "Certification Rates"]
        )
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Employers", f"{len(employer_summary):,}")
        with col2:
            st.metric("Total Applications", f"{employer_summary['Total_Applications'].sum():,}")
        with col3:
            st.metric("Avg Cert Rate", f"{employer_summary['Certification_Rate'].mean():.1f}%")
        with col4:
            st.metric("Available States", f"{employer_summary['State'].nunique()}")
        
        st.markdown("---")
        
        # Generate and display charts
        if analysis_type == "State Overview":
            fig = px.bar(
                state_summary.head(25),
                x='State',
                y='Total_Applications',
                title='H-1B Applications by State (Top 25 States)',
                color='Certification_Rate',
                color_continuous_scale='RdYlGn',
                hover_data={'Employer_Count': True, 'Certification_Rate': ':.2f'},
                text='Total_Applications'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show state summary table
            st.subheader("📋 State Summary Statistics")
            st.dataframe(state_summary.head(15), use_container_width=True)
            
        elif analysis_type == "Single State Deep Dive" and len(selected_states) == 1:
            # Single state analysis
            state = selected_states[0]
            state_data = employer_summary[employer_summary['State'] == state].head(top_n)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_bar_chart(state_data, f"in {state}")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = create_cert_rate_chart(state_data, f"in {state}")
                st.plotly_chart(fig2, use_container_width=True)
            
            # State summary
            st.subheader(f"📊 Summary for {state}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Employers Analyzed", len(state_data))
            with col2:
                st.metric("Total Applications", f"{state_data['Total_Applications'].sum():,}")
            with col3:
                st.metric("Avg Cert Rate", f"{state_data['Certification_Rate'].mean():.2f}%")
            
            # Top employers table
            st.subheader(f"🏆 Top Employers in {state}")
            display_data = state_data[['Employer_Name', 'Total_Applications', 'Certification_Rate', 'City']].head(10)
            st.dataframe(display_data, use_container_width=True)
            
        else:
            # Multi-state analysis
            if 'ALL STATES' in selected_states:
                filtered_data = employer_summary.head(top_n)
                title_suffix = "(All States)"
            else:
                filtered_data = filter_data(employer_summary, selected_states, top_n)
                title_suffix = f"({', '.join(selected_states)})"
            
            # Create appropriate chart
            chart_mapping = {
                "Total Applications Bar": create_bar_chart,
                "Stacked Case Status": create_stacked_chart,
                "Certification Rates": create_cert_rate_chart
            }
            
            fig = chart_mapping[chart_type](filtered_data, title_suffix)
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis summary
            st.subheader("📋 Analysis Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Selected States:** {', '.join(selected_states) if 'ALL STATES' not in selected_states else 'All States'}")
                st.write(f"**Employers Analyzed:** {len(filtered_data):,}")
                st.write(f"**Total Applications:** {filtered_data['Total_Applications'].sum():,}")
                st.write(f"**Average Certification Rate:** {filtered_data['Certification_Rate'].mean():.2f}%")
            
            with col2:
                if len(filtered_data) > 0:
                    st.write("**Top 5 Employers:**")
                    top_5 = filtered_data[['Employer_Name', 'State', 'Total_Applications', 'Certification_Rate']].head(5)
                    st.dataframe(top_5, use_container_width=True)
    
    else:
        # Cap-Exempt Institutions Section
        st.markdown('<h2 class="sub-header">🎓 Cap-Exempt Institutions Analysis</h2>', unsafe_allow_html=True)
        
        # Cap-exempt summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Cap-Exempt Institutions", f"{cap_exempt_institutions['EMPLOYER_NAME'].nunique():,}")
        with col2:
            st.metric("Total Records", f"{len(lca_filtered):,}")
        with col3:
            st.metric("Cap-Exempt Records", f"{len(cap_exempt_institutions):,}")
        with col4:
            st.metric("Percentage Cap-Exempt", f"{len(cap_exempt_institutions)/len(lca_filtered)*100:.2f}%")
        
        st.markdown("---")
        
        # Chart type selection for cap-exempt
        cap_exempt_chart = st.sidebar.selectbox(
            "Cap-Exempt Chart Type:",
            ["Institution Type Distribution", "Top Institutions by State"]
        )
        
        if cap_exempt_chart == "Institution Type Distribution":
            # Display stacked bar chart
            fig = create_cap_exempt_stacked_chart(cap_exempt_institutions)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show institution type summary
            st.subheader("📊 Institution Type Summary")
            cap_exempt_institutions['Institution_Type'] = cap_exempt_institutions['NAICS_CODE'].apply(categorize_naics)
            type_summary = cap_exempt_institutions['Institution_Type'].value_counts().reset_index()
            type_summary.columns = ['Institution Type', 'Count']
            st.dataframe(type_summary, use_container_width=True)
            
        else:
            # Top institutions by state
            # State selection for cap-exempt
            all_cap_exempt_states = sorted(cap_exempt_institutions['EMPLOYER_STATE'].dropna().unique())
            selected_cap_exempt_state = st.sidebar.selectbox(
                "Select State:",
                options=all_cap_exempt_states,
                index=0 if all_cap_exempt_states else None
            )
            
            if selected_cap_exempt_state:
                fig = create_cap_exempt_dropdown_chart(cap_exempt_institutions, selected_cap_exempt_state)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show top institutions table
                    st.subheader(f"📋 Top Cap-Exempt Institutions in {selected_cap_exempt_state}")
                    state_data = cap_exempt_institutions[
                        cap_exempt_institutions['EMPLOYER_STATE'] == selected_cap_exempt_state
                    ]
                    institution_summary = state_data['EMPLOYER_NAME'].value_counts().head(10).reset_index()
                    institution_summary.columns = ['Institution Name', 'Applications']
                    st.dataframe(institution_summary, use_container_width=True)
                else:
                    st.warning(f"No cap-exempt institutions found in {selected_cap_exempt_state}")

if __name__ == "__main__":
    main()