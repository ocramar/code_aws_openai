#!/usr/bin/env python3
import psycopg2

conn = psycopg2.connect(
    dbname="chembl_35",
    user="chembl",
    password="STK01nicolae$",
    host="localhost",
    port=5432
)
cur = conn.cursor()
cur.execute("""
    SELECT assay_organism, COUNT(*) AS cnt
      FROM assays
     GROUP BY assay_organism
     ORDER BY cnt DESC
     LIMIT 22;
""")
for organism, cnt in cur.fetchall():
    print(f"{str(organism):20s} {cnt}")
cur.close()
conn.close()