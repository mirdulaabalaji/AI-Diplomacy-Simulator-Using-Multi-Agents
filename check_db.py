import sqlite3
import os

db_path = os.path.join("memory", "negotiation_memory.db")
conn = sqlite3.connect(db_path)

print("=== ALL AGENT STATS ===")
rows = conn.execute("SELECT * FROM agent_stats").fetchall()
for row in rows:
    print(row)

print("\n=== COUNTRY B SPECIFICALLY ===")
b_rows = conn.execute(
    "SELECT * FROM agent_stats WHERE agent_name LIKE '%B%' OR agent_id = 'B'"
).fetchall()
print(b_rows if b_rows else "NO ROWS FOUND FOR COUNTRY B")

print("\n=== ALL SIMULATIONS ===")
sims = conn.execute("SELECT id, run_at, outcome, rounds FROM simulations").fetchall()
for s in sims:
    print(s)

conn.close()