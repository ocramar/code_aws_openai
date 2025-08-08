import json
import boto3
import psycopg2
from psycopg2.extras import RealDictCursor
import ast
import numpy as np
from numpy.linalg import norm

bedrock_runtime = boto3.client("bedrock-runtime")
lambda_client = boto3.client('lambda')
s3 = boto3.client('s3')
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
q = None

def lambda_handler(event, context):
    global q
    question = event.get('question', None)
    schema_id = event.get('schema_id', 'vitic_lhasa')
    if question is None:
        return build_response(400, 'Exact', None, 'No question provided', None)
    q = question
    sql_query = ''
    response = get_schema(question)
    schema = response.get('schema')
    #print(schema)
    sql_prompt_file = response.get('sql_prompt_file')
    narrative_prompt_file = response.get('narrative_prompt_file')
    try:
        print('Converting the question into SQL')
        sql_query = generateSqlStatement(question, schema, schema_id, sql_prompt_file)
    except Exception as e:
        return build_response(400, 'Exact', None, f'Error converting question to SQL: {e}', narrative_prompt_file)

    if sql_query is None or sql_query == '':
        return build_response(400, 'Exact', None, 'Error converting question to SQL', narrative_prompt_file)

    print('Running the SQL query')
    print(sql_query)
    response = run_query(sql_query)

    if response is None or len(response) == 0:
    # if response is None or len(response) == 0 or not validate_response(response, question, schema):
        print('No results, so attempting to modify SQL')
        print('Identifying chemical names in the question')
        compound_names = identify_chemical_names(question)
        if len(compound_names) > 0:
            print('Found the following chemical names')
            print(compound_names)
            name_to_smiles_dict = {}
            print('Attempting to convert to SMILES')
            for compound in compound_names:
                try:
                    smiles = get_smiles(compound)
                    if smiles != compound:
                        name_to_smiles_dict[compound] = smiles
                except Exception as e:
                    return build_response(400, 'Modified', None, f'Error getting SMILES for compound: {e}', narrative_prompt_file)

            if len(name_to_smiles_dict) == 0:
                return build_response(400, 'Modified', None, 'No results found', narrative_prompt_file)
            print('Converted the following chemical names to SMILES')
            print(name_to_smiles_dict)
            modified_query = modify_query(sql_query, name_to_smiles_dict)
            print('Running the modified query with exact SMILES strings')
            print(modified_query)
            response = run_query(modified_query)

            if response is None or len(response) == 0:
                print('No results, so attempting to modify the SQL using similarity')
                return modify_for_similarity(modified_query)
            else:
                return build_response(200, 'Modified', modified_query, response, narrative_prompt_file)
    
        return build_response(400, 'Modified', None, 'Unable to extract SMILES strings', narrative_prompt_file)

    return build_response(200, 'Exact', sql_query, response, narrative_prompt_file)

def convert_to_sql(question, schema):
    payload = {
        'question': question,
        'schemaId': 'vitic_lhasa',
        'schema': schema
    }

    response = lambda_client.invoke(
        FunctionName='lhasa-genai-development-s-CombinedRagFusionChatSql-aeB3VICi0vaP',
        Payload=json.dumps(payload)
    )

    response_payload = json.loads(response['Payload'].read())

    return response_payload.get('response')

def get_smiles(compound):
    payload = {
        'value': compound,
        'output': 'SMILES'
    }

    response = lambda_client.invoke(
        FunctionName='parse_structure',
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )

    response_payload = json.loads(response['Payload'].read())

    return response_payload.get('response')

def run_query(sql_query):
    host = "lhasa-genai-development-s-viticdatabaseinstancea0f-tamqzv49vqny.coxu0bzg5vzg.eu-west-2.rds.amazonaws.com" #"your-rds-endpoint"
    port = "5432"
    dbname = "viticdb"
    user = "root"
    password = "SOavplKcRx,.Z_UF73YWww2yWyiOVO"

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
    except psycopg2.Error as e:
        return {
            'error': str(e)
        }

    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        cursor.execute(sql_query)
        result = {
            'result': cursor.fetchall()
        }
    except psycopg2.Error as e:
        result = {
            'error': str(e)
        }

    cursor.close()
    conn.close()

    return result.get('result')

def identify_chemical_names(question):
    prompt = f"""
        You are an expert at identifying chemical names in scientific questions. You are given a question and need to identify and return these chemical names.

        Question:
        {question}

        You must not provide any preamble or explanation. The response should be returned in an array that can be iterated over by a lambda.
        Be very careful when identifying compounds, do not include concepts or subjects.
        If you cannot identify any chemical compounds, return an empty array

        Example question: Give me the phototoxicity of paracetamol and benzene and eggs
        Expected response: ['paracetamol', 'benzene']
        """
    response = invoke_model(prompt)
    return ast.literal_eval(json.loads(response['body'].read())['content'][0]['text'])

def modify_for_similarity(sql):
    prompt = f"""
        You are an expert at modifying SQL queries. You are given an existing SQL query.

        SQL query:
        {sql}

        Your task is to replace specific predicates in the SQL query with an alternative.
        You are looking for predicates of this form "WHERE mol @= mol_from_smiles('<a value that must be used in the modified sql>'::cstring)
        When a predicate like this is found, it must be replaced with the following: "WHERE tanimoto_sml(rdkit_fp(mol_from_smiles('<a value that must be used in the modified sql>'::cstring)), fingerprint) > 0.6"
        The user is only interested in the SQL query, do not provide any preamble or explanation for your work.
        The SQL query will be used in subsequent function calls so must be provided in a form that can be used immediately.
        Do not return the mol column in the response, it contains information that cannot be parsed by the user.

        Example SQL query: SELECT * FROM vitic_lhasa WHERE mol @= mol_from_smiles('CCO'::cstring)
        Expected response: SELECT * FROM vitic_lhasa WHERE tanimoto_sml(rdkit_fp(mol_from_smiles('<a value that must be used in the modified sql>'::cstring)), fingerprint) > 0.6
        
        ** IMPORTANT **
        When making this replacement, do not make any assumptions about any other substitutions to be made.
        If the sql performs a join on the structures table, add the mol column if it is missing from the existing SQL.
        You must not make the following substitution: WHERE s.structure @= mol_from_smiles it must ALWAYS be WHERE s.mol @= mol_from_smiles
        Only make replacements as detailed above
        """
    model_response = invoke_model(prompt)
    new_sql = json.loads(model_response['body'].read())['content'][0]['text']
    print('Running the modified query with similarity search')
    print(new_sql)
    response = run_query(new_sql)
    if response is None or len(response) == 0:
        return build_response(200, 'Modified', sql, 'No results', narrative_prompt_file)
    else:
        return build_response(200, 'Similarity', new_sql, response, narrative_prompt_file)

def modify_query(sql, compound_name_dict):
    prompt = f"""
        You are an expert at modifying SQL queries. You are given an existing SQL query and a set of key value pairs.

        SQL query:
        {sql}

        Key value pairs:
        {compound_name_dict}

        Your task is to replace predicates in the SQL query using the set of key value pairs.
        If the predicate looks at the common_name column, replace it with a function call using the value in the key value pair.
        If the predicate does not look at the common_name column, do not modify it.
        The user is only interested in the SQL query, do not provide any preamble or explanation for your work.
        The SQL query will be used in subsequent function calls so must be provided in a form that can be used immediately.

        Example SQL query: SELECT * FROM vitic_lhasa WHERE common_name = 'ethanol' AND common_name = 'benzene'
        Example key value pairs: 'ethanol': 'CCO', 'benzene': 'c1ccccc1'
        Expected response: SELECT * FROM vitic_lhasa WHERE mol @= mol_from_smiles('CCO'::cstring) AND mol @= mol_from_smiles('c1ccccc1'::cstring)
        
        ** IMPORTANT **
        When making this replacement, it is very important that the mol column is used, absolutely do not use the structure column in the updated query.
        """
    response = invoke_model(prompt)
    return json.loads(response['body'].read())['content'][0]['text']

def validate_response(response, question, schema):
    prompt = f"""
        You are an expert at validating SQL responses. You are given a schema, a question, and a response.

        Schema:
        {schema}

        Question:
        {question}

        Response:
        {response}

        Your task is to determine if the response is valid. A valid response is one that contains results, it is not empty
        and it is relevant for the question asked, considering the schema.
        If the response is valid, return true. If the response is invalid, return false.
        Do not provide any preamble or explanation for your work.
        """
    response = invoke_model(prompt)
    return json.loads(response['body'].read())['content'][0]['text'].lower() == 'true'

def get_schema(question, schema_id='vitic_lhasa'):
    host = "lhasa-genai-development-s-viticdatabaseinstancea0f-tamqzv49vqny.coxu0bzg5vzg.eu-west-2.rds.amazonaws.com"
    port = "5432"
    dbname = "viticdb"
    user = "root"
    password = "SOavplKcRx,.Z_UF73YWww2yWyiOVO"

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
    except psycopg2.Error as e:
        print("Error: Could not make connection to the Postgres database")
        print(e)
        exit()
    response = {}
    cursor = conn.cursor()
    subschema = choose_subschema_with_keyword_boost(question)
    print(f"Identified subschema: {subschema}")
    if subschema == 'GENERAL':
        result = buildSchemaString(schema_id, cursor)
        response['sql_prompt_file'] = 'vitic-prompt-v2.txt'
        response['narrative_prompt_file'] = 'CARCINOGENICITY-narrative_prompt.txt'
    else:
        result = buildSubSchemaString(schema_id, cursor, subschema)
        response['sql_prompt_file'] = subschema + '-prompt.txt'
        response['narrative_prompt_file'] = subschema + '-narrative_prompt.txt'
    response['schema'] = result
    cursor.close()
    conn.close()

    return response

def buildSchemaString(schema_id, cursor):

    cursor.execute("select META_CODE, META_VALUE from " + schema_id + ".vdb_db_metadata where META_KEY = 'TABLE_DESC'")
    tables = cursor.fetchall()
    cursor.execute("select META_CODE, META_VALUE from " + schema_id + ".vdb_db_metadata where META_KEY = 'COL_DESC';")
    columns = cursor.fetchall()
    cursor.execute("select META_CODE, META_VALUE from " + schema_id + ".vdb_db_metadata where META_KEY = 'PARENT_TABLE';")
    supplementary = cursor.fetchall()
    cursor.execute("select table_name, column_name, data_type from information_schema.columns where table_schema = '" + schema_id + "';")
    information = cursor.fetchall()

    parent_map = {}
    index = 0
    while index < len(supplementary):
        parent_map[supplementary[index][0]] = supplementary[index][1]
        index += 1

    table_map = {}
    index = 0
    while index < len(tables):
        entry = {
            'alias': tables[index][1],
            'columns': ['luid NUMERIC PRIMARY KEY',
                        'vdb_created_date TIMESTAMP WITHOUT TIME ZONE',
                        'vdb_modified_date TIMESTAMP WITHOUT TIME ZONE',
                        'vdb_created_by CHARACTER VARYING',
                        'vdb_modified_by CHARACTER VARYING']
        }
        if tables[index][0] in parent_map.keys():
            entry['columns'] = entry['columns'] + ['parent_luid NUMERIC FOREIGN KEY LINKING TO "' + parent_map[tables[index][0]] + '" TABLE THROUGH "' + parent_map[tables[index][0]] + '.luid" COLUMN']
        else:
            entry['columns'] = entry['columns'] + ['structure_luid NUMERIC FOREIGN KEY LINKING TO "structures" TABLE THROUGH "structures.luid" COLUMN']
        table_map[tables[index][0]] = entry
        index += 1

    information_map = {}
    index = 0
    while index < len(information):
        if information[index][0] not in information_map:
            information_map[information[index][0]] = {}
        information_map[information[index][0]][information[index][1]] = information[index][2]
        index += 1

    index = 0
    while index < len(columns):
        tablesplit = columns[index][0].rsplit('.', 1)[0]
        split = columns[index][0].split('.')
        if split[2] != 'LUID':
            table = split[1].lower()
            column = split[2].lower()
            if table in information_map and tablesplit in table_map:
                table_map[tablesplit]['columns'] = table_map[tablesplit]['columns'] + [column + ', column type: ' + information_map[table][column].upper() + ', column alias: "' + columns[index][1] + '"']
        index += 1

    schemamap = {}

    schemastring = ''
    for key in table_map.keys():
        tableline = 'table_name: ' + key.lower() + ', table alias: ' + table_map[key]['alias'] + ', '
        for col in table_map[key]['columns']:
            tableline = tableline + 'column_name: ' + col + ', '
        tableline = tableline[:-2] + '\n'
        schemastring = schemastring + tableline

    return schemastring

def build_response(statusCode, type, sql, response, prompt_file=None):
    global q
    summary = 'No available summary'
    metadata = []
    if statusCode == 200:
        summary = create_summary(sql, response, prompt_file)
        try:
            for row in response:
                metadata.append({"type":"vitic", "data": dict(row)})
        except:
            metadata = []
    return {
        'question': q,
        'response': summary,
        'metadata': metadata
    }

def create_summary(sql, sql_result, prompt_file):
    try:
        # Load prompt template from S3
        row_count = len(sql_result)
        prompt_response = s3.get_object(Bucket='vitic-systemprompt', Key=prompt_file)
        prompt_template = prompt_response['Body'].read().decode('utf-8')

        # Fill in placeholders in the prompt
        narrative_prompt = prompt_template.format(sql=sql, sql_result=sql_result,row_count=row_count)

        # Invoke model
        bedrock_response = invoke_model(narrative_prompt)
        raw_response = bedrock_response['body'].read().decode('utf-8')
        parsed = json.loads(raw_response)

        # Extract and format model output
        if 'content' in parsed:
            summary_text = parsed['content'][0]['text']
        elif 'results' in parsed:
            summary_text = parsed['results'][0]['outputText']
        else:
            return 'Unexpected model response format.'

        print("Response, start_________:")
        rearranged = rearrange_summary(summary_text)
        print(format_response_markdown(rearranged))
        print("___________Response, end")
        return format_response_markdown(rearranged)

    except Exception as e:
        return f'Unable to generate summary: {e}' 
        
def rearrange_summary(summary_text):
    lines = summary_text.strip().split("\n")
    compound_line = ""
    bullets = []
    conclusion_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # 1. Capture the compound line
        if stripped.startswith("**Compound Evaluated**:"):
            compound_line = line
            continue
        # 2. Normalize numbered bullets into dash bullets
        if stripped and stripped[0].isdigit() and stripped[1] == ".":
            bullets.append("- " + stripped[3:].strip())
        elif stripped.startswith("-"):
            bullets.append(stripped)
        elif stripped.startswith("**Study Findings**:"):
            continue  # Ignore misplaced header
        elif stripped:  # Any paragraph text (conclusion)
            conclusion_lines.append(stripped)
    # 3. Compose the rearranged result with formatting
    rearranged = f"""{compound_line}
---

{" ".join(conclusion_lines)}

**Study Findings**:

{chr(10).join(bullets)}"""

    return rearranged.strip()

def format_response_markdown(raw_text):
    lines = raw_text.split("\n")
    formatted_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped and stripped[0].isdigit() and stripped[1] == '.':
            # Convert "1. text" into "- text"
            formatted_lines.append(f"- {stripped[3:].strip()}")
        else:
            formatted_lines.append(line)
    return "\n".join(formatted_lines)
    
def invoke_model(prompt):
    return  bedrock_runtime.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8192,
                "temperature": 0,
            }
        ),
    )

def buildSubSchemaString(schema_id, cursor, sub_schema):
    cursor.execute("((SELECT a.meta_code, a.meta_value FROM " + schema_id + ".vdb_db_metadata a JOIN (SELECT meta_code, meta_value FROM " + schema_id + ".vdb_db_metadata WHERE meta_value='" + schema_id.upper() + "." + sub_schema.upper() + "') AS b ON a.meta_code = b.meta_code WHERE a.meta_key = 'TABLE_DESC')) UNION (SELECT META_CODE, META_VALUE FROM " + schema_id + ".vdb_db_metadata WHERE META_KEY = 'TABLE_DESC' AND meta_code IN ('" + schema_id.upper() + "." + sub_schema.upper() + "', '" + schema_id.upper() + ".STRUCTURES', '" + schema_id.upper() + ".SYNONYMS'))")
    tables = cursor.fetchall()
    cursor.execute("select META_CODE, META_VALUE from " + schema_id + ".vdb_db_metadata where META_KEY = 'COL_DESC';")
    columns = cursor.fetchall()
    cursor.execute("select META_CODE, META_VALUE from " + schema_id + ".vdb_db_metadata where META_KEY = 'PARENT_TABLE';")
    supplementary = cursor.fetchall()
    cursor.execute("select table_name, column_name, data_type from information_schema.columns where table_schema = '" + schema_id + "';")
    information = cursor.fetchall()

    parent_map = {}
    index = 0
    while index < len(supplementary):
        parent_map[supplementary[index][0]] = supplementary[index][1]
        index += 1

    table_map = {}
    index = 0
    while index < len(tables):
        entry = {
            'alias': tables[index][1],
            'columns': ['luid NUMERIC PRIMARY KEY',
                        'vdb_created_date TIMESTAMP WITHOUT TIME ZONE',
                        'vdb_modified_date TIMESTAMP WITHOUT TIME ZONE',
                        'vdb_created_by CHARACTER VARYING',
                        'vdb_modified_by CHARACTER VARYING']
        }
        if tables[index][0] in parent_map.keys():
            entry['columns'] = entry['columns'] + ['parent_luid NUMERIC FOREIGN KEY LINKING TO "' + parent_map[tables[index][0]] + '" TABLE THROUGH "' + parent_map[tables[index][0]] + '.luid" COLUMN']
        else:
            entry['columns'] = entry['columns'] + ['structure_luid NUMERIC FOREIGN KEY LINKING TO "structures" TABLE THROUGH "structures.luid" COLUMN']
        table_map[tables[index][0]] = entry
        index += 1

    information_map = {}
    index = 0
    while index < len(information):
        if information[index][0] not in information_map:
            information_map[information[index][0]] = {}
        information_map[information[index][0]][information[index][1]] = information[index][2]
        index += 1

    index = 0
    while index < len(columns):
        tablesplit = columns[index][0].rsplit('.', 1)[0]
        split = columns[index][0].split('.')
        if split[2] != 'LUID':
            table = split[1].lower()
            column = split[2].lower()
            if table in information_map and tablesplit in table_map:
                #table_map[tablesplit]['columns'] = table_map[tablesplit]['columns'] + [column + ', column type: ' + information_map[table][column].upper() + ', column alias: "' + columns[index][1] + '"']
                table_map[tablesplit]['columns'] = table_map[tablesplit]['columns'] + [column + ', column type: ' + information_map[table][column].upper()]
        index += 1

    schemamap = {}

    schemastring = ''
    for key in table_map.keys():
        #tableline = 'table name: ' + key.lower() + ', table alias: ' + table_map[key]['alias'] + ', '
        tableline = 'table name: ' + key.lower() + ', '
        for col in table_map[key]['columns']:
            tableline = tableline + 'column name: ' + col + ', '
        tableline = tableline[:-2] + '\n'
        schemastring = schemastring + tableline
    sample_columns_map = {
        'CARCINOGENICITY': ['testtype', 'species', 'strain', 'route'],
        'TOGENETICINVIVOTAB': ['test_type', 'species', 'strain', 'route_of_administration'],
        'TOGENETICINVITROTAB': ['test_type', 'species', 'strain'],
        'TOREPEATEDDOSETAB': ['test_type', 'species', 'strain', 'route']
    }

    sub_schema_upper = sub_schema.upper()
    schema_id_upper = schema_id.upper()

    if sub_schema_upper in sample_columns_map:
        try:
            schemastring += f"\nMost common values for table {sub_schema_upper.lower()}\n"
            for col in sample_columns_map[sub_schema_upper]:
                query = f"""
                    SELECT {col}, COUNT(*) as freq
                    FROM {schema_id_upper}.{sub_schema_upper}
                    WHERE {col} IS NOT NULL
                    GROUP BY {col}
                    ORDER BY freq DESC
                    LIMIT 9;
                """
                cursor.execute(query)
                rows = cursor.fetchall()
                schemastring += f" Column: {col},  Values:\n"
                for value, freq in rows:
                    display_value = str(value).replace('\n', ' ') if value is not None else 'NULL'
                    schemastring += f"  {display_value}, \n"
        except Exception as e:
            schemastring += f"\n[Error fetching top values for {sub_schema_upper}]: {str(e)}\n"    
    return schemastring

def choose_subschema_with_keyword_boost(user_question: str) -> str:
    subschema_definitions = {
        "CARCINOGENICITY": ("Data related to carcinogenicity, cancer, lymphoma and carcinogens in studies."),
        "TOGENETICINVIVOTAB": ("This schema contains important in vivo genotoxicity studies"),
        "TOGENETICINVITROTAB": ( "This schema contains important in vitro genotoxicity studies"),
        "TOREPEATEDDOSETAB": ("repeated dose studies")
    }
    question_vec = embed_with_cohere([user_question])[0]
    keyword_boosts = {
        "TOGENETICINVITROTAB": ["genotoxicity","genotoxic","mutagen","mutagenicity","mutagenitic","mutations","in vitro", "ames", "ames test", "bacterial reverse mutation", "micronucleus (in vitro)","micronucleus (in vitro)","positive in vitro", "GreenScreen GC", "CYP1A","DNA damage", "chromosome aberration", "mouse lymphoma assay", "sister chromatid exchange assay", "aneuploidy test", "micronucleus assay", "mammalian cell gene mutation assay", "unscheduled dna synthesis assay", "bacterial mutagenicity test", "cytogenetic assay", "hprt assay", "polyploidy assay", "ames ii assay", "yeast gene mutation assay", "greenscreen gc assay", "dna damage and repair assay", "dna repair assay"],
        "TOGENETICINVIVOTAB": ["genotoxicity","genotoxic","mutagen","mutagenicity","mutagenitic","mutations","in vivo","UDS","micronucleus assay", "chromosome aberration", "cytogenetic assay", "unscheduled dna synthesis assay", "sex-linked recessive lethal", "dominant lethal assay", "sister chromatid exchange assay", "comet assay", "transgenic models", "alkaline elution assay", "dna damage and repair assay", "pig-a assay", "gene mutation assay", "dna binding assay", "host mediated assay", "somatic mutation and recombination test (smart)", "transgenic rodent assay"],
        "CARCINOGENICITY": ["cancer", "carcinogen", "tumor","tumors","hyperplasia", "preneoplastic","hypertrophy","experiment time","exposure time" "organ weight increase","testicular atrophy", "ATP4A", "AhR binding", "cross-species relevance", "gastric cancer", "testicular atrophy", "chronic toxicity", "lymphoma","chronic", "18 month", "overall", "subchronic", "lifetime", "assay","initiation-promotion assay", "neonatal assay", "transgenic rodent assay", "photocarcinogenicity assay", "2 year study", "transplacental assay", "initiation assay", "promotion assay", "multi-generation "],
        "TOREPEATEDDOSETAB":  ["repeated dose","repeat dose"]
    }

    max_score = -1
    best_match = None

    for schema, description in subschema_definitions.items():
        desc_vec = embed_with_cohere([description])[0]
        score = np.dot(question_vec, desc_vec) / (norm(question_vec) * norm(desc_vec))

        # Add keyword boost
        for keyword in keyword_boosts.get(schema, []):
            if keyword in user_question.lower():
                score += 0.1

        if score > max_score:
            max_score = score
            best_match = schema

    return best_match if max_score > 0.15 else "GENERAL"

def embed_with_cohere(texts):
    response = bedrock_runtime.invoke_model(
        modelId="cohere.embed-multilingual-v3",
        body=json.dumps({
            "texts": texts,
            "input_type": "search_document"
        }),
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(response['body'].read())
    return [np.array(vec) for vec in result['embeddings']]

def generateSqlStatement(question, schema, schema_id, prompt_file):
    MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
    prompt_response = s3.get_object(Bucket='vitic-systemprompt', Key=prompt_file)
    prompt_text = prompt_response['Body'].read().decode('utf-8')
    prompt = prompt_text.format(schema=schema, question=question, schema_id=schema_id)
    response = bedrock_runtime.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8192,
                "temperature": 0,
            }
        ),
    )
    return json.loads(response['body'].read())['content'][0]['text']