# from chembl_webresource_client.new_client import new_client
# import pandas as pd
# pd.set_option('display.width', None)    # don’t wrap
# pd.set_option('display.max_columns', None)

# # grab a single molecule record

# rows = []
# for num in range(1503, 1504):                 # 1503 → 1512 inclusive
#     chembl_id = f'CHEMBL{num}'
#     mol = new_client.molecule.get(chembl_id)

#     rows.append({
#         "chembl_id":        chembl_id,
#         "pref_name":        mol.get("pref_name"),
#         "full_mwt":         mol.get("molecule_properties", {}).get("full_mwt"),
#         "max_phase":        mol.get("max_phase"),
#         "therapeutic_flag": mol.get("therapeutic_flag"),
#         "trade names:"  :   mol.get("trade_names", []),
#     })

# df = pd.DataFrame(rows)
# print(df.to_string(index=False))

from chembl_webresource_client.new_client import new_client

mol = new_client.molecule.get('CHEMBL1503')

trade_names = [
    s['synonyms']
    for s in mol.get('molecule_synonyms', [])
    if s.get('syn_type') == 'TRADE_NAME'
]

print(trade_names)
