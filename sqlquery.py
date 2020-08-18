import sqlite3

conn = sqlite3.connect('../database/rightmove.db')
c = conn.cursor()

c.execute("SELECT  COUNT(title) AS count FROM rightmove")
print(c.fetchall())

# c.execute("DELETE FROM rightmove")

conn.commit()
conn.close()
