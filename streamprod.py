import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="H-1B Analysis Dashboard",
    # page_icon="🏢",
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
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .success-banner {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SQLiteH1BReader:
    """SQLite reader for H-1B data analysis with caching"""
    
    def __init__(self, db_path="data/lca_disclosure.db"): 
        self.db_path = db_path
    
    def get_connection(self):
        """Get SQLite connection"""
        return sqlite3.connect(self.db_path)
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_employer_summary(_self, limit=None):
        """Get employer summary with SQL aggregation and caching"""
        query = """
        SELECT 
            employer_name as "Employer_Name",
            employer_city as "City",
            employer_state as "State",
            COUNT(*) as "Total_Applications",
            SUM(CASE WHEN case_status = 'CERTIFIED' THEN 1 ELSE 0 END) as "Certified_Applications",
            SUM(CASE WHEN case_status = 'DENIED' THEN 1 ELSE 0 END) as "Denied_Applications",
            ROUND(
                (SUM(CASE WHEN case_status = 'CERTIFIED' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2
            ) as "Certification_Rate"
        FROM lca_data
        WHERE employer_name != '' AND employer_name != 'NAN'
        GROUP BY employer_name, employer_city, employer_state
        ORDER BY Total_Applications DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        with _self.get_connection() as conn:
            return pd.read_sql_query(query, conn)
    
    @st.cache_data(ttl=3600)
    def get_state_summary(_self):
        """Get state-level summary with caching"""
        query = """
        SELECT 
            employer_state as "State",
            COUNT(*) as "Total_Applications",
            SUM(CASE WHEN case_status = 'CERTIFIED' THEN 1 ELSE 0 END) as "Certified_Applications",
            SUM(CASE WHEN case_status = 'DENIED' THEN 1 ELSE 0 END) as "Denied_Applications",
            COUNT(DISTINCT employer_name) as "Employer_Count",
            ROUND(
                (SUM(CASE WHEN case_status = 'CERTIFIED' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2
            ) as "Certification_Rate"
        FROM lca_data
        WHERE employer_state != '' AND employer_state != 'NAN'
        GROUP BY employer_state
        ORDER BY Total_Applications DESC
        """
        
        with _self.get_connection() as conn:
            return pd.read_sql_query(query, conn)
    
    @st.cache_data(ttl=3600)
    def get_filtered_employers(_self, states=None, top_n=20):
        """Get filtered employer data with caching"""
        where_clause = "WHERE employer_name != '' AND employer_name != 'NAN'"
        params = []
        
        if states and 'ALL STATES' not in states:
            placeholders = ','.join(['?' for _ in states])
            where_clause += f" AND employer_state IN ({placeholders})"
            params.extend(states)
        
        query = f"""
        SELECT 
            employer_name as "Employer_Name",
            employer_city as "City", 
            employer_state as "State",
            COUNT(*) as "Total_Applications",
            SUM(CASE WHEN case_status = 'CERTIFIED' THEN 1 ELSE 0 END) as "Certified_Applications",
            SUM(CASE WHEN case_status = 'DENIED' THEN 1 ELSE 0 END) as "Denied_Applications",
            ROUND(
                (SUM(CASE WHEN case_status = 'CERTIFIED' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2
            ) as "Certification_Rate"
        FROM lca_data
        {where_clause}
        GROUP BY employer_name, employer_city, employer_state
        ORDER BY Total_Applications DESC
        LIMIT ?
        """
        
        params.append(top_n)
        
        with _self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    @st.cache_data(ttl=3600)
    def get_cap_exempt_institutions(_self):
        """Get cap-exempt institutions with caching"""
        query = """
        SELECT 
            employer_name as "Employer_Name",
            employer_state as "State",
            naics_code as "NAICS_Code",
            COUNT(*) as "Applications",
            CASE 
                WHEN CAST(naics_code AS TEXT) LIKE '611%' THEN 'Educational Services'
                WHEN naics_code = 541711 THEN 'Biotech R&D'
                WHEN naics_code = 541712 THEN 'Physical/Engineering R&D'
                WHEN CAST(naics_code AS TEXT) LIKE '622%' THEN 'Hospitals'
                WHEN naics_code = 927110 THEN 'Space Research'
                WHEN CAST(naics_code AS TEXT) LIKE '9271%' THEN 'Government Research'
                ELSE 'Other'
            END as "Institution_Type"
        FROM lca_data
        WHERE 
            CAST(naics_code AS TEXT) LIKE '611%'  -- Education
            OR naics_code = 541711  -- Biotech R&D
            OR naics_code = 541712  -- Physical/Engineering R&D
            OR CAST(naics_code AS TEXT) LIKE '622%'  -- Hospitals
            OR naics_code = 927110  -- Space research
            OR CAST(naics_code AS TEXT) LIKE '9271%'  -- Government research
        GROUP BY employer_name, employer_state, naics_code
        ORDER BY Applications DESC
        """
        
        with _self.get_connection() as conn:
            return pd.read_sql_query(query, conn)
    
    @st.cache_data(ttl=3600)
    def get_top_job_titles(_self, limit=20):
        """Get top job titles with caching"""
        query = f"""
        SELECT 
            job_title as "Job_Title",
            COUNT(*) as "Total_Applications",
            SUM(CASE WHEN case_status = 'CERTIFIED' THEN 1 ELSE 0 END) as "Certified_Applications",
            ROUND(
                (SUM(CASE WHEN case_status = 'CERTIFIED' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2
            ) as "Certification_Rate"
        FROM lca_data
        WHERE job_title != '' AND job_title != 'NAN'
        GROUP BY job_title
        ORDER BY Total_Applications DESC
        LIMIT {limit}
        """
        
        with _self.get_connection() as conn:
            return pd.read_sql_query(query, conn)
    
    def get_database_stats(self):
        """Get database statistics"""
        with self.get_connection() as conn:
            stats = {}
            
            # Total records
            stats['total_records'] = conn.execute("SELECT COUNT(*) FROM lca_data").fetchone()[0]
            
            # Unique employers
            stats['unique_employers'] = conn.execute(
                "SELECT COUNT(DISTINCT employer_name) FROM lca_data WHERE employer_name != ''"
            ).fetchone()[0]
            
            # Unique states
            stats['unique_states'] = conn.execute(
                "SELECT COUNT(DISTINCT employer_state) FROM lca_data WHERE employer_state != ''"
            ).fetchone()[0]
            
            # Database file size
            if Path(self.db_path).exists():
                stats['file_size_mb'] = Path(self.db_path).stat().st_size / (1024 * 1024)
            else:
                stats['file_size_mb'] = 0
            
            # Case status distribution
            case_status = conn.execute("""
                SELECT case_status, COUNT(*) 
                FROM lca_data 
                GROUP BY case_status 
                ORDER BY COUNT(*) DESC
            """).fetchall()
            stats['case_status_dist'] = dict(case_status)
            
            return stats

@st.cache_resource
def get_database_reader():
    """Initialize database reader with caching"""
    return SQLiteH1BReader()

def create_charts(data, chart_type, title_suffix=""):
    """Create various chart types for data visualization"""
    
    if chart_type == "Total Applications Bar":
        fig = px.bar(
            data, 
            x='Total_Applications', 
            y='Employer_Name',
            orientation='h',
            title=f'Top {len(data)} Employers by H-1B Applications {title_suffix}',
            color='Total_Applications',
            color_continuous_scale='Blues',
            text='Total_Applications',
            hover_data=['State', 'Certification_Rate']
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=max(400, len(data) * 25),
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=250),
            showlegend=False
        )
    
    elif chart_type == "Certification Rates":
        fig = px.bar(
            data,
            x='Certification_Rate',
            y='Employer_Name',
            orientation='h',
            title=f'H-1B Certification Rates {title_suffix}',
            color='Certification_Rate',
            color_continuous_scale='RdYlGn',
            text='Certification_Rate',
            hover_data=['State', 'Total_Applications']
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            height=max(400, len(data) * 25),
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=250),
            showlegend=False
        )
    
    elif chart_type == "Stacked Case Status":
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Certified',
            x=data['Employer_Name'],
            y=data['Certified_Applications'],
            marker_color='#27ae60'
        ))
        fig.add_trace(go.Bar(
            name='Denied',
            x=data['Employer_Name'],
            y=data['Denied_Applications'],
            marker_color='#e74c3c'
        ))
        fig.update_layout(
            barmode='stack',
            title=f'H-1B Case Status Distribution {title_suffix}',
            xaxis_title='Employer',
            yaxis_title='Number of Applications',
            xaxis_tickangle=-45,
            height=600,
            margin=dict(b=120)
        )
    
    return fig

def check_database_exists():
    """Check if the SQLite database exists and has data"""
    db_path = "data/lca_disclosure.db"
    
    if not Path(db_path).exists():
        return False, "Database file not found"
    
    try:
        reader = SQLiteH1BReader(db_path)
        stats = reader.get_database_stats()
        if stats['total_records'] == 0:
            return False, "Database exists but is empty"
        return True, f"Database ready with {stats['total_records']:,} records"
    except Exception as e:
        return False, f"Database error: {str(e)}"

def show_setup_instructions():
    """Show setup instructions when database is not found"""
    st.error(" Database not found!")
    
    st.markdown("""
    ###  Setup Required
    
    Please run the migration script first to create the database from your Excel file.
    """)
    
    with st.expander(" Detailed Setup Instructions", expanded=True):
        st.markdown("""
        **Step 1: Prepare your Excel file**
        - Place your H-1B LCA disclosure Excel file in the project directory
        - Note the exact file name and sheet name
        
        **Step 2: Update migration script**
        - Open `migrate_h1b_data.py`
        - Update these lines:
        ```python
        EXCEL_FILE_PATH = "your_file_name.xlsx"
        SHEET_NAME = "your_sheet_name"
        ```
        
        **Step 3: Run migration**
        ```bash
        python migrate_h1b_data.py
        ```
        
        **Step 4: Refresh this page**
        - Once migration completes successfully, refresh this dashboard
        """)
    
    st.info(" The migration script only needs to be run once. After that, this dashboard will load instantly!")

def main():
    """Main Streamlit application"""
    
    # Title and header
    st.markdown('<h1 class="main-header"> H-1B Analysis Dashboard</h1>', unsafe_allow_html=True)
    # st.markdown("### Professional H-1B visa application data analysis and insights")
    
    # Check if database exists
    db_exists, db_message = check_database_exists()
    
    if not db_exists:
        show_setup_instructions()
        return
    
    # Database is ready - show success message
    st.markdown(f'<div class="success-banner"> {db_message}</div>', unsafe_allow_html=True)
    
    # Initialize database reader
    db_reader = get_database_reader()
    
    # Load initial data for metrics
    with st.spinner("Loading database statistics..."):
        stats = db_reader.get_database_stats()
    
    # Display key metrics
    st.markdown("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Records", 
            value=f"{stats['total_records']:,}",
            help="Total number of H-1B applications in the database"
        )
    with col2:
        st.metric(
            label="Unique Employers", 
            value=f"{stats['unique_employers']:,}",
            help="Number of distinct employers in the dataset"
        )
    with col3:
        st.metric(
            label="States Covered", 
            value=f"{stats['unique_states']}",
            help="Number of states with H-1B applications"
        )
    with col4:
        st.metric(
            label="Database Size", 
            value=f"{stats['file_size_mb']:.1f} MB",
            help="Size of the SQLite database file"
        )
    
    # Case status distribution
    if stats.get('case_status_dist'):
        st.markdown("### Case Status Distribution")
        status_df = pd.DataFrame(list(stats['case_status_dist'].items()), 
                                columns=['Status', 'Count'])
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.pie(status_df, values='Count', names='Status', 
                        title='H-1B Application Status Distribution',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(status_df, use_container_width=True)
    
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header(" Analysis Controls")
    
    # Analysis section selection
    analysis_section = st.sidebar.radio(
        "Select Analysis Type:",
        ["Employer Analysis", "Cap-Exempt Institutions", "Job Title Analysis"]
    )
    
    if analysis_section == "Employer Analysis":
        # Employer Analysis Section
        st.markdown('<h2 class="sub-header">Employer Analysis</h2>', unsafe_allow_html=True)
        
        # Analysis type selection
        analysis_type = st.sidebar.radio(
            "Analysis Scope:",
            ["Multi-State Analysis", "Single State Deep Dive", "State Overview"]
        )
        
        if analysis_type == "State Overview":
            # State overview analysis
            with st.spinner("Loading state data..."):
                state_summary = db_reader.get_state_summary()
            
            # State overview chart
            fig = px.bar(
                state_summary.head(25),
                x='State',
                y='Total_Applications',
                title='H-1B Applications by State (Top 25)',
                color='Certification_Rate',
                color_continuous_scale='RdYlGn',
                hover_data=['Employer_Count', 'Certification_Rate'],
                text='Total_Applications'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(height=600, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # State summary table
            st.subheader("State Summary Details")
            
            # Add search functionality
            search_state = st.text_input(" Search for a specific state:", 
                                       placeholder="Enter state abbreviation (e.g., CA, NY, TX)")
            
            display_df = state_summary.copy()
            if search_state:
                display_df = display_df[display_df['State'].str.contains(search_state.upper(), na=False)]
            
            st.dataframe(display_df.head(20), use_container_width=True)
        
        else:
            # Multi-state or single state analysis
            with st.spinner("Loading employer data..."):
                state_summary = db_reader.get_state_summary()
            
            # State selection
            available_states = ['ALL STATES'] + sorted(state_summary['State'].unique().tolist())
            
            if analysis_type == "Single State Deep Dive":
                selected_states = [st.sidebar.selectbox(
                    "Select State:",
                    options=[s for s in available_states if s != 'ALL STATES']
                )]
            else:
                selected_states = st.sidebar.multiselect(
                    "Select States:",
                    options=available_states,
                    default=['ALL STATES']
                )
            
            # Top N selection
            top_n = st.sidebar.slider("Top N Employers:", 5, 50, 20, 5)
            
            # Chart type selection
            chart_type = st.sidebar.selectbox(
                "Chart Type:",
                ["Total Applications Bar", "Certification Rates", "Stacked Case Status"]
            )
            
            # Get filtered data
            with st.spinner("Filtering employer data..."):
                if 'ALL STATES' in selected_states or not selected_states:
                    filtered_data = db_reader.get_employer_summary(top_n)
                    title_suffix = "(All States)"
                else:
                    filtered_data = db_reader.get_filtered_employers(selected_states, top_n)
                    title_suffix = f"({', '.join(selected_states)})"
            
            # Create and display chart
            if len(filtered_data) > 0:
                fig = create_charts(filtered_data, chart_type, title_suffix)
                st.plotly_chart(fig, use_container_width=True)
                
                # Analysis summary
                st.subheader("Analysis Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Selected States:** {', '.join(selected_states) if 'ALL STATES' not in selected_states else 'All States'}")
                    st.write(f"**Employers Analyzed:** {len(filtered_data):,}")
                    st.write(f"**Total Applications:** {filtered_data['Total_Applications'].sum():,}")
                    st.write(f"**Average Certification Rate:** {filtered_data['Certification_Rate'].mean():.2f}%")
                
                with col2:
                    st.subheader("Top 5 Employers")
                    top_5_display = filtered_data[['Employer_Name', 'State', 'Total_Applications', 'Certification_Rate']].head(5)
                    st.dataframe(top_5_display, use_container_width=True)
                
                # Detailed table with search
                st.subheader("Detailed Results")
                search_employer = st.text_input(" Search for specific employer:", 
                                              placeholder="Enter employer name or part of it")
                
                display_data = filtered_data.copy()
                if search_employer:
                    display_data = display_data[
                        display_data['Employer_Name'].str.contains(search_employer.upper(), na=False)
                    ]
                
                st.dataframe(display_data[['Employer_Name', 'State', 'City', 'Total_Applications', 'Certification_Rate']], 
                           use_container_width=True)
            else:
                st.warning("No data found for selected criteria")
    
    elif analysis_section == "🎓 Cap-Exempt Institutions":
        # Cap-Exempt Institutions Analysis
        st.markdown('<h2 class="sub-header">🎓 Cap-Exempt Institutions Analysis</h2>', unsafe_allow_html=True)
        
        with st.spinner("Loading cap-exempt institution data..."):
            cap_exempt_data = db_reader.get_cap_exempt_institutions()
        
        if len(cap_exempt_data) > 0:
            # Cap-exempt summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cap-Exempt Institutions", f"{cap_exempt_data['Employer_Name'].nunique():,}")
            with col2:
                st.metric("Total Applications", f"{cap_exempt_data['Applications'].sum():,}")
            with col3:
                st.metric("Institution Types", f"{cap_exempt_data['Institution_Type'].nunique()}")
            
            # Institution type distribution
            type_summary = cap_exempt_data.groupby('Institution_Type')['Applications'].sum().reset_index()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.pie(
                    type_summary,
                    values='Applications',
                    names='Institution_Type',
                    title='Cap-Exempt H-1B Applications by Institution Type',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Institution Types")
                st.dataframe(type_summary, use_container_width=True)
            
            # State-wise cap-exempt analysis
            st.subheader("Cap-Exempt Institutions by State")
            
            cap_exempt_states = sorted(cap_exempt_data['State'].dropna().unique())
            selected_state = st.selectbox(
                "Select State for Detailed View:",
                options=['All States'] + cap_exempt_states
            )
            
            if selected_state == 'All States':
                # Show top states
                state_summary = cap_exempt_data.groupby('State')['Applications'].sum().reset_index()
                state_summary = state_summary.sort_values('Applications', ascending=False).head(15)
                
                fig = px.bar(
                    state_summary,
                    x='State',
                    y='Applications',
                    title='Top 15 States by Cap-Exempt H-1B Applications',
                    color='Applications',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Show institutions in selected state
                state_data = cap_exempt_data[cap_exempt_data['State'] == selected_state]
                top_institutions = state_data.nlargest(20, 'Applications')
                
                if len(top_institutions) > 0:
                    fig = px.bar(
                        top_institutions,
                        x='Applications',
                        y='Employer_Name',
                        orientation='h',
                        title=f'Top 20 Cap-Exempt Institutions in {selected_state}',
                        color='Institution_Type',
                        hover_data=['NAICS_Code']
                    )
                    fig.update_layout(
                        height=800,
                        yaxis={'categoryorder': 'total ascending'},
                        margin=dict(l=300)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No cap-exempt institutions found in {selected_state}")
            
            # Detailed cap-exempt table
            st.subheader("Top Cap-Exempt Institutions")
            
            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                institution_type_filter = st.selectbox(
                    "Filter by Institution Type:",
                    options=['All Types'] + sorted(cap_exempt_data['Institution_Type'].unique())
                )
            with col2:
                state_filter = st.selectbox(
                    "Filter by State:",
                    options=['All States'] + sorted(cap_exempt_data['State'].unique())
                )
            
            # Apply filters
            filtered_cap_exempt = cap_exempt_data.copy()
            if institution_type_filter != 'All Types':
                filtered_cap_exempt = filtered_cap_exempt[filtered_cap_exempt['Institution_Type'] == institution_type_filter]
            if state_filter != 'All States':
                filtered_cap_exempt = filtered_cap_exempt[filtered_cap_exempt['State'] == state_filter]
            
            display_data = filtered_cap_exempt.nlargest(25, 'Applications')
            st.dataframe(
                display_data[['Employer_Name', 'State', 'Institution_Type', 'Applications']],
                use_container_width=True
            )
        else:
            st.info("No cap-exempt institutions found in the current dataset")
    
    else:
        # Job Title Analysis
        st.markdown('<h2 class="sub-header">💼 Job Title Analysis</h2>', unsafe_allow_html=True)
        
        with st.spinner("Loading job title data..."):
            job_title_data = db_reader.get_top_job_titles(30)
        
        if len(job_title_data) > 0:
            # Job title metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Job Titles", f"{len(job_title_data):,}")
            with col2:
                st.metric("Total Applications", f"{job_title_data['Total_Applications'].sum():,}")
            with col3:
                st.metric("Avg Certification Rate", f"{job_title_data['Certification_Rate'].mean():.1f}%")
            
            # Job title chart selection
            chart_option = st.radio(
                "Select Chart Type:",
                ["Total Applications", "Certification Rates"],
                horizontal=True
            )
            
            if chart_option == "Total Applications":
                fig = px.bar(
                    job_title_data.head(20),
                    x='Total_Applications',
                    y='Job_Title',
                    orientation='h',
                    title='Top 20 Job Titles by H-1B Applications',
                    color='Total_Applications',
                    color_continuous_scale='Blues',
                    text='Total_Applications'
                )
            else:
                fig = px.bar(
                    job_title_data.head(20),
                    x='Certification_Rate',
                    y='Job_Title',
                    orientation='h',
                    title='Top 20 Job Titles by Certification Rate',
                    color='Certification_Rate',
                    color_continuous_scale='RdYlGn',
                    text='Certification_Rate'
                )
                fig.update_traces(texttemplate='%{text:.1f}%')
            
            fig.update_traces(textposition='outside')
            fig.update_layout(
                height=600,
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=200)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Job title table
            st.subheader("Job Title Details")
            st.dataframe(job_title_data, use_container_width=True)
        else:
            st.info("No job title data found in the current dataset")
    
    # Footer
    st.markdown("---")
    st.markdown("### About This Dashboard")
    st.info("""
    This dashboard analyzes H-1B visa application data using a high-performance SQLite database. 
    The data includes employer information, geographic distribution, case status, and cap-exempt institution analysis.
    
    **Data Sources**: U.S. Department of Labor LCA Disclosure Data
    **Technology**: Streamlit + SQLite + Plotly
    """)

if __name__ == "__main__":
    main()