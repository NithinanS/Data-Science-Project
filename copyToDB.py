import psycopg2
import glob
import os
import pandas as pd

# Database connection parameters
DB_HOST = "localhost"
DB_NAME = "Data Science"
DB_USER = "postgres"
DB_PASSWORD = "root"

# Directory containing CSV files
CSV_DIR = "./FilteredDataWithYear/"

# Connect to the database
try:
    conn = psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )
    print("Connected to the database successfully.")
except Exception as e:
    print("Error connecting to the database:", e)
    exit()

# Function to create a table dynamically
# Function to create a table dynamically
def create_table_from_csv(cursor, tablename, df):
    columns = df.columns
    sql_create = f"CREATE TABLE IF NOT EXISTS {tablename} ("
    for column in columns:
        # Ensure column names are SQL-safe
        safe_column = column.replace(" ", "_").replace("-", "_").replace(".", "_")
        sql_create += f"{safe_column} TEXT, "  # Use TEXT for unlimited string length
    sql_create = sql_create.rstrip(", ") + ");"
    cursor.execute(sql_create)


# Function to insert data into the table
def insert_data_from_csv(cursor, tablename, filepath):
    # Use COPY for efficient bulk insert
    with open(filepath, "r", encoding="utf-8") as f:
        cursor.copy_expert(f"COPY {tablename} FROM STDIN WITH CSV HEADER", f)

# Automate CSV import
def import_csv_to_db():
    cursor = conn.cursor()
    for filepath in glob.glob(os.path.join(CSV_DIR, "*.csv")):
        try:
            # Get the table name from the file name
            tablename = os.path.basename(filepath).replace(".csv", "").lower()

            # Read CSV into a DataFrame
            df = pd.read_csv(filepath)

            # Create table if it doesn't exist
            create_table_from_csv(cursor, tablename, df)
            conn.commit()
            print(f"Table '{tablename}' created or already exists.")

            # Insert data into the table
            insert_data_from_csv(cursor, tablename, filepath)
            conn.commit()
            print(f"Data from '{filepath}' inserted into '{tablename}' successfully.")
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
        finally:
            conn.rollback()  # Ensure no partial commits on errors
    cursor.close()

# Run the import process
import_csv_to_db()

# Close the connection
conn.close()
