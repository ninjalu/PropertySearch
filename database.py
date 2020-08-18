import sqlite3
from sqlite3 import DatabaseError

# create and connect to database
conn = sqlite3.connect('../database/rightmove.db')

# create a table
c = conn.cursor()

c.execute("""CREATE TABLE rightmove (
        id  TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        price INTEGER NOT NULL,
        description TEXT NOT NULL 

    ) ;""")

# commit command
conn.commit()

# close connection
conn.close()
