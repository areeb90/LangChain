import sqlite3
from faker import Faker
import random

# Initialize Faker for generating random data
fake = Faker()

# Connect to SQLite
conn = sqlite3.connect('test_large.db')
cursor = conn.cursor()

# Drop table if it already exists
cursor.execute("DROP TABLE IF EXISTS STUDENT")

# Create table with many columns
table = """
CREATE TABLE STUDENT (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    NAME TEXT,
    AGE INTEGER,
    GENDER TEXT,
    CLASS TEXT,
    SECTION TEXT,
    GPA REAL,
    ENROLLMENT_DATE DATE,
    EMAIL TEXT,
    PHONE_NUMBER TEXT,
    CITY TEXT
);
"""
cursor.execute(table)

# Helper function to generate a student row
def generate_student():
    name = fake.name()
    age = random.randint(18, 30)
    gender = random.choice(['Male', 'Female', 'Other'])
    class_name = random.choice(['Data Science', 'DevOps', 'Cybersecurity', 'AI', 'Web Development'])
    section = random.choice(['A', 'B', 'C', 'D'])
    gpa = round(random.uniform(2.0, 4.0), 2)
    enrollment_date = fake.date_between(start_date='-4y', end_date='today')
    email = fake.email()
    phone = fake.phone_number()
    city = fake.city()
    return (name, age, gender, class_name, section, gpa, enrollment_date, email, phone, city)

# Insert many rows
num_records = 1000  # You can increase this up to 10,000+ if needed
students = [generate_student() for _ in range(num_records)]

cursor.executemany('''
    INSERT INTO STUDENT (
        NAME, AGE, GENDER, CLASS, SECTION, GPA, ENROLLMENT_DATE, EMAIL, PHONE_NUMBER, CITY
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', students)

# Commit and close
conn.commit()
print(f"{num_records} records inserted into STUDENT table.")
conn.close()
