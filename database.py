import pandas as pd
import sqlite3
from pathlib import Path
import logging
import sys


# Configurations
# lca = pd.read_excel('LCA_Disclosure.xlsx',sheet_name='LCA_Disclosure_Data_FY2025_Q2')

EXCEL_FILE_PATH = "LCA_Disclosure.xlsx"
SHEET_NAME = "LCA_Disclosure_Data_FY2025_Q2"
DB_FILE_PATH = "data/lca_disclosure.db"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Migrator:
    def __init__(self, db_path = "data/lca_disclosure.db"):
        self.db_path = db_path

        # Ensure data directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def create_database(self):
        conn = sqlite3.connect(self.db_path)

        # Fixed: Drop the correct table name
        conn.execute("DROP TABLE IF EXISTS lca_data")
        
        # Create lca table
        conn.execute('''
            CREATE TABLE lca_data(
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     case_status TEXT NOT NULL,
                     visa_class TEXT,
                     job_title TEXT,
                     soc_code TEXT,
                     soc_title TEXT,
                     full_time_position TEXT,
                     employer_name TEXT NOT NULL,
                     employer_city TEXT,
                     employer_state TEXT,
                     employer_country TEXT,
                     naics_code INTEGER,
                     wage_unit_of_pay TEXT,
                     prevailing_wage REAL,
                     pw_unit_of_pay TEXT,
                     support_h1b TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
                     ''')
        
        # Fixed: performance indexes (CREATE INDEX, not CREATE_INDEX)
        indexes = [
            "CREATE INDEX idx_employer_name ON lca_data(employer_name)",
            "CREATE INDEX idx_employer_state ON lca_data(employer_state)",
            "CREATE INDEX idx_case_status ON lca_data(case_status)",
            "CREATE INDEX idx_naics_code ON lca_data(naics_code)",
            "CREATE INDEX idx_state_name ON lca_data(employer_state, employer_name)",
            "CREATE INDEX idx_status_employer ON lca_data(case_status, employer_name)"
        ]

        for index_sql in indexes:
            conn.execute(index_sql)

        conn.commit()
        conn.close()
        logger.info("Database schema created successfully.")

    def clean_and_prepare_data(self, df):
        logger.info("Cleaning and preparing data...")
        # column mapping from Excel to DB
        column_mapping = {
            'CASE_STATUS' : 'case_status',
            'VISA_CLASS' : 'visa_class',
            'JOB_TITLE' : 'job_title',
            'SOC_CODE' : 'soc_code',
            'SOC_TITLE' : 'soc_title',
            'FULL_TIME_POSITION' : 'full_time_position',
            'EMPLOYER_NAME' : 'employer_name',
            'EMPLOYER_CITY' : 'employer_city',
            'EMPLOYER_STATE' : 'employer_state',
            'EMPLOYER_COUNTRY' : 'employer_country',
            'NAICS_CODE' : 'naics_code',
            'WAGE_UNIT_OF_PAY' : 'wage_unit_of_pay',
            'PREVAILING_WAGE' : 'prevailing_wage',
            'PW_UNIT_OF_PAY' : 'pw_unit_of_pay',
            'SUPPORT_H1B' : 'support_h1b'
        }

        # selecting available columns
        available_columns = [col for col in column_mapping.keys() if col in df.columns]
        logger.info(f"Available columns in the dataset: {available_columns}")

        if len(available_columns) == 0:
            logger.error("No expected columns found in Excel file.")
            logger.error(f"Available columns in Excel : {list(df.columns)}")
            raise ValueError("Excel File doesn't contain expected columns.")
        
        df_filtered = df[available_columns].copy()
        df_filtered = df_filtered.rename(columns = column_mapping)

        # Data Cleaning
        logger.info("cleaning text fields")

        # Focusing on employer_name
        df_filtered['employer_name'] = df_filtered['employer_name'].astype(str).str.strip().str.upper()
        df_filtered = df_filtered[df_filtered['employer_name'] != '']
        df_filtered = df_filtered[df_filtered['employer_name'] != 'N/A']

        # clean state
        if 'employer_state' in df_filtered.columns:
            df_filtered['employer_state'] = df_filtered['employer_state'].astype(str).str.strip().str.upper()

        # clean case status
        if 'case_status' in df_filtered.columns:
            df_filtered['case_status'] = df_filtered['case_status'].astype(str).str.strip().str.upper()
        
        # clean city
        if 'employer_city' in df_filtered.columns:
            df_filtered['employer_city'] = df_filtered['employer_city'].astype(str).str.strip().str.upper()

        # NAICS code
        if 'naics_code' in df_filtered.columns:
            df_filtered['naics_code'] = pd.to_numeric(df_filtered['naics_code'], errors='coerce')
        
        # Prevailing wage
        if 'prevailing_wage' in df_filtered.columns:
            df_filtered['prevailing_wage'] = pd.to_numeric(df_filtered['prevailing_wage'], errors='coerce')

        # FILL IN missing values with appropriate defaults
        fill_values = {
            'employer_city': '',
            'employer_state': '',
            'job_title': '',
            'soc_code': '',
            'soc_title': '',
            'visa_class': '',
            'full_time_position': '',
            'wage_unit_of_pay': '',
            'pw_unit_of_pay': '',
            'support_h1b': '',
            'employer_country': 'USA',
        }

        for column, value in fill_values.items():
            if column in df_filtered.columns:
                df_filtered[column] = df_filtered[column].fillna(value)

        logger.info(f"Data cleaned : {len(df_filtered)} records ready for insertion.")
        return df_filtered
    
    def migrate_excel_to_db(self, excel_file_path, sheet_name):
        try:
            logger.info(f"Reading Excel file: {excel_file_path}, sheet: {sheet_name}")

            if not Path(excel_file_path).exists():
                logger.error(f"Excel file not found: {excel_file_path}")
                logger.error("Please ensure the file exists and the path is correct.")
                return False
            
            # Read Excel file
            logger.info("Loading data from Excel...")
            try:
                df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
                logger.info(f"Excel file loaded successfully with {len(df)} records.")
            except Exception as e:
                logger.error(f"Error reading Excel file: {str(e)}")
                logger.error("Please check the file path and sheet name.")
                return False
            
            # create a database 
            self.create_database()
            
            # Clean and prepare data
            df_cleaned = self.clean_and_prepare_data(df)

            # insert data into db
            logger.info("Inserting data into database...")
            conn = sqlite3.connect(self.db_path)

            # using chunks to avoid memory issues
            chunk_size = 10000
            # Fixed: proper parentheses for chunk calculation
            total_chunks = (len(df_cleaned) + chunk_size - 1) // chunk_size

            for i, chunk_start in enumerate(range(0, len(df_cleaned), chunk_size)):
                chunk_end = min(chunk_start + chunk_size, len(df_cleaned))
                chunk = df_cleaned.iloc[chunk_start:chunk_end]
                
                chunk.to_sql('lca_data', conn, if_exists='append', index=False, method='multi')
                # Fixed: variable name typo (df_cleaned not df_clean)
                logger.info(f"Inserted chunk {i+1}/{total_chunks} ({chunk_end}/{len(df_cleaned)} rows)")
            
            conn.commit()
            conn.close()
            
            # Verify migration
            self.verify_migration()
            
            logger.info("Migration completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            return False

    def verify_migration(self):
        """Verify the migration by checking the number of records in the database."""
        conn = sqlite3.connect(self.db_path)
        # basic stats
        total_rows = conn.execute("SELECT COUNT(*) FROM lca_data").fetchone()[0]
        unique_employers = conn.execute("SELECT COUNT(DISTINCT employer_name) FROM lca_data").fetchone()[0]
        unique_states = conn.execute("SELECT COUNT(DISTINCT employer_state) FROM lca_data WHERE employer_state != ''").fetchone()[0]

        # Get case status distribution
        status_dist = conn.execute("""
            SELECT case_status, COUNT(*) 
            FROM lca_data 
            GROUP BY case_status 
            ORDER BY COUNT(*) DESC
        """).fetchall()
        
        # Get file size
        file_size_mb = Path(self.db_path).stat().st_size / (1024 * 1024)
        
        conn.close()
        
        logger.info("📊 Migration Verification:")
        logger.info(f"   • Total records: {total_rows:,}")
        logger.info(f"   • Unique employers: {unique_employers:,}")
        logger.info(f"   • States covered: {unique_states}")
        logger.info(f"   • Database size: {file_size_mb:.1f} MB")
        logger.info(f"   • Case status distribution: {dict(status_dist)}")
        
        return {
            'total_records': total_rows,
            'unique_employers': unique_employers,
            'unique_states': unique_states,
            'file_size_mb': file_size_mb
        }


def main():
    """Main migration function"""
    print("🏢 H-1B Data Migration Tool")
    print("=" * 40)
    
    # Display configuration
    print(f"📁 Excel file: {EXCEL_FILE_PATH}")
    print(f"📄 Sheet name: {SHEET_NAME}")
    print(f"🗄️ Database: {DB_FILE_PATH}")
    print()
    
    # Check if Excel file exists
    if not Path(EXCEL_FILE_PATH).exists():
        print(f"❌ Error: Excel file not found: {EXCEL_FILE_PATH}")
        print()
        print("📝 Setup Instructions:")
        print("1. Place your H-1B Excel file in this directory")
        print("2. Update EXCEL_FILE_PATH in this script")
        print("3. Update SHEET_NAME if needed")
        print("4. Run this script again")
        return 1
    
    # initialize migrator
    migrator = Migrator(DB_FILE_PATH)

    # Run migration
    print("🚀 Starting migration...")
    success = migrator.migrate_excel_to_db(EXCEL_FILE_PATH, SHEET_NAME)
    
    if success:
        print("\n✅ Migration completed successfully!")
        print(f"💾 Database created at: {DB_FILE_PATH}")
        return 0
    else:
        print("\n❌ Migration failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())