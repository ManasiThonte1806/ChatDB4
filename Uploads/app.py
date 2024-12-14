from flask import Flask, request, jsonify, render_template
import os
import pymongo
import pandas as pd
import spacy
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging
import re
import json
from datetime import datetime
from dateutil import parser as date_parser
from pymongo import MongoClient


HOST = 'localhost'  
USER = 'root'
PASSWORD = ''  # Set to an empty string for no password
DATABASE = 'Sql_db'
app = Flask(__name__)

# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")


# Configure the upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create SQLAlchemy engine
# engine = create_engine('mysql+pymysql://root:Manasi%401806@localhost/Sql_db')
engine = create_engine(f'mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}/{DATABASE}')


# Configure MongoDB connection
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["chatdb_nosql"]


# Schema storage for datasets
schemas = {}

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload-database", methods=["POST"])
def upload_database():
    
    # Endpoint to upload CSV or JSON files and infer schemas.
    
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    if file.filename.endswith(".csv"):
        df = pd.read_csv(file_path)
        schemas[file.filename] = {"type": "SQL", "columns": list(df.columns)}
    elif file.filename.endswith(".json"):
        with open(file_path, "r") as f:
            try:
                json_data = json.load(f)
                if isinstance(json_data, list) and len(json_data) > 0:
                    first_doc = json_data[0]
                elif isinstance(json_data, dict):
                    first_doc = json_data
                else:
                    return jsonify({"error": "Invalid JSON format"}), 400
                
                schemas[file.filename] = {"type": "NoSQL", "keys": list(first_doc.keys())}
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON format"}), 400

    return jsonify({"message": f"File '{file.filename}' uploaded successfully.", "schema": schemas[file.filename]})
        


@app.route("/schema", methods=["POST"])
def get_schema():
    """
    Fetch schemas for all datasets under a selected database.
    """
    data = request.get_json()
    db_type = data.get("db_type")
    dataset_name = data.get("dataset_name")  # Here, dataset_name refers to the database (e.g., students_db)

    if not db_type or not dataset_name:
        return jsonify({"error": "Database type and dataset name are required."}), 400

    try:
        if db_type == "SQL":
            schemas = get_all_schemas(dataset_name)
            return jsonify({"schemas": schemas})
            
        elif db_type == "NoSQL":
            schema = get_mongo_schema(dataset_name)
            return {"schema": schema}
        else:
            return jsonify({"error": "Unsupported database type."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_all_schemas(database_name):
    """
    Fetch schemas for all tables in the given database.
    """
    with engine.connect() as connection:
        # Get all tables in the database
        result = connection.execute(text(f"SHOW TABLES IN {database_name}"))
        tables = [row[0] for row in result]

        schemas = {}
        schemas_only_names = {}
        for table in tables:
            table_result = connection.execute(text(f"DESCRIBE {database_name}.{table}"))
            columns = [{"name": row[0], "type": row[1]} for row in table_result]
            schemas_only_names[table] = {row[0] for row in table_result}
            schemas[table] = {"columns": columns}

        return schemas


def get_mongo_schema(database_name):
    """
    Get detailed schemas (keys and inferred data types) of all collections in a MongoDB database.
    """
    db = mongo_client[database_name]  # Access the specified database
    collections = db.list_collection_names()  # List all collections in the database
    database_schema = {}

    for collection_name in collections:
        collection = db[collection_name]
        sample_docs = collection.find().limit(1)  # Retrieve up to 1 sample documents for better inference
        schema = {}

        for doc in sample_docs:
            for key, value in doc.items():
                # Infer data type and store the most general type if keys are repeated
                inferred_type = type(value).__name__
                if key not in schema:
                    schema[key] = inferred_type
                elif schema[key] != inferred_type:
                    schema[key] = "Mixed"  # Mark as mixed if types vary

        database_schema[collection_name] = schema
    # print(database_schema)

    return database_schema



@app.route("/execute-query", methods=["POST"])
def execute_query():
    
    data = request.get_json()
    db_type = data.get("db_type")
    natural_query = data.get("query")
    dataset_name = data.get("dataset_name")

    if not db_type or not natural_query or not dataset_name:
        return jsonify({"error": "Database type, query, and dataset name are required."}), 400

    try:
        if db_type == "SQL":
            # Always use the predefined database
            table_schema = get_all_schemas(dataset_name)

            table_schema = {
            "students": ["StudentID", "FirstName", "LastName", "Email", "Major", "AID", "Marks","CourseID"],
            "instructors": ["AID", "iname", "CourseID"],
            "courses": ["CourseID", "cname", "CreditHours"]
            }

            
            if "example" in natural_query and all(keyword not in natural_query for keyword in ["groupby", "limit", "orderby", "join"]):
                natural_queries = [
                    "Show the courses table",
                    "Show the instructors table",
                    "Show the students table"]
                sql_query = []
                for query in natural_queries:
                    # print("Query:",query)
                    sql_q = natural_to_sql(query, table_schema)
                    sql_query.append("\n" + sql_q)
                return jsonify({
                    "sql_query": sql_query
                })

            elif "groupby" in natural_query:
                natural_queries = [
                    "Find count of students in each major"
                ]
                sql_query = []
                for query in natural_queries:
                    # print("Query:",query)
                    sql_q = natural_to_sql(query, table_schema)
                    sql_query.append("\n" + sql_q)
                return jsonify({
                    "sql_query": sql_query
                })
            elif "orderby" in natural_query:
                natural_queries = [
                    "Show the first name last name of students with highest marks as first"
                ]
                sql_query = []
                for query in natural_queries:
                    # print("Query:",query)
                    sql_q = natural_to_sql(query, table_schema)
                    sql_query.append("\n" + sql_q)
                return jsonify({
                    "sql_query": sql_query
                })
            elif "limit" in natural_query:
                natural_queries = [
                    "Show the first name last name of top 5 students"
                ]
                sql_query = []
                for query in natural_queries:
                    # print("Query:",query)
                    sql_q = natural_to_sql(query, table_schema)
                    sql_query.append("\n" + sql_q)
                return jsonify({
                    "sql_query": sql_query
                })
            elif "join" in natural_query:
                natural_queries = [
                    "Give the student ID, firstname from students whose marks are between 45 and 67 in the course named genetics"
                ]
                sql_query = []
                for query in natural_queries:
                    # print("Query:",query)
                    sql_q = natural_to_sql(query, table_schema)
                    sql_query.append("\n" + sql_q)
                return jsonify({
                    "sql_query": sql_query
                })
            else:
                sql_query = natural_to_sql(natural_query, table_schema)
                # print(sql_query)
                return jsonify({
                    "sql_query": sql_query  # Send the generated SQL query in the response
                })

            # if natural_query == "Give example queries" or natural_query == "give me example queries" or natural_query == "Give some example queries":
            #     natural_queries = [
            #         "Show the courses table",
            #         "Show the instructors table",
            #         "Show the students table"
            #     ]
            #     sql_query = []
            #     for query in natural_queries:
            #         sql_q = natural_to_sql(query, table_schema)
            #         sql_query.append("\n" + sql_q)
            #     return jsonify({
            #         "sql_query": sql_query
            #     })
            # else:
            # # print(natural_query)
            #     sql_query = natural_to_sql(natural_query, table_schema)
            #     # print(sql_query)
            #     return jsonify({
            #         "sql_query": sql_query  # Send the generated SQL query in the response
            #     })

        elif db_type == "NoSQL":
            processed_query = preprocess_query(natural_query)
            response = generate_mongodb_command_string(processed_query)
            if "error" in response:
                return jsonify(response), 400

            collection = mongo_db[dataset_name]
            mongo_command = response.get("mongodb_command")
            if not mongo_command:
                return jsonify({"error": "Failed to generate MongoDB command."}), 500

            filter_query = response.get("filter", {})
            results = list(collection.find(filter_query))
            for result in results:
                result["_id"] = str(result["_id"])
            return jsonify({"mongodb_command": response.get("mongodb_command")})


        else:
            return jsonify({"error": "Unsupported database type."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# SQL query execution logic   
action_synonyms = {
    'find': 'get',
    'retrieve': 'get',
    'show': 'get',
    'display': 'get',
    'list': 'get',
    'give': 'get',
    'fetch': 'get',
}

operator_synonyms = {
    'equals': 'is',
    'equal to': 'is',
    'are greater than': '>',
    'is more than': '>',
    'is less than': '<',
    'is under': '<',
    'are under': '<',
    'are over': '>',
    'are above': '>',
    'are below': '<',
    'are between': 'BETWEEN',
    'is between':'BETWEEN',
    'scored between': 'BETWEEN',
    'not equal to': '!=',
    'not equals': '!=',
    'starts with':'starts like',
    'begin with': 'begin like',
    'starts with prefix': 'prefix like',
    'ends with suffix': 'suffix like',
    'ends with': 'ends like',
    'contains': 'contains like',
    'includes': 'includes like'
}

logical_synonyms = {
    'and also': 'and',
    'as well as': 'and',
    'along with': 'and',
    'or else': 'or',
    'in the': 'and'
}

aggregate_synonyms = {
    'total': 'sum',
    'summarize': 'sum',
    'average': 'avg',
    'mean': 'avg',
    'number of': 'count',
    'no.of': 'count',
    'count of': 'count',
}

# Attribute synonyms
attribute_synonyms = {
    'student id': 'StudentID',
    'student number': 'StudentID',
    'StudentID': 'StudentID',
    'firstname': 'FirstName',
    'first name': 'FirstName',
    'lastname': 'LastName',
    'last name': 'LastName',
    'email address': 'Email',
    'email id': 'Email',
    'major name': 'Major',
    'major': 'Major',
    'advisor id': 'AID',
    'id of advisor': 'AID',
    'marks obtained': 'Marks',
    'marks': 'Marks',
    'instructors name': 'iname',
    'name of instructors': 'iname',
    'course id': 'CourseID',
    'courses name': 'cname',
    'course names':'cname',
    'credit hours': 'CreditHours',
    'has taken course': 'cname is',
    'course named': 'cname is'
}

def preprocess_query_sql(natural_query):
    # Convert to lowercase
    query = natural_query.lower()
    
    # Replace action synonyms
    for synonym, standard in action_synonyms.items():
        query = re.sub(r'\b' + re.escape(synonym) + r'\b', standard, query)
    
    # Replace operator synonyms
    for synonym, standard in operator_synonyms.items():
        query = re.sub(r'\b' + re.escape(synonym) + r'\b', standard, query)
    
    # Replace logical operator synonyms
    for synonym, standard in logical_synonyms.items():
        query = re.sub(r'\b' + re.escape(synonym) + r'\b', standard, query)
    
    # Replace aggregate function synonyms
    for synonym, standard in aggregate_synonyms.items():
        query = re.sub(r'\b' + re.escape(synonym) + r'\b', standard, query)
    
    # Replace attribute synonyms
    for synonym, standard in attribute_synonyms.items():
        query = re.sub(r'\b' + re.escape(synonym) + r'\b', standard, query)
    
    # Handle possessives (e.g., "student's major" -> "student major")
    query = re.sub(r"'s\b", '', query)
    
    # Remove unnecessary words or phrases
    filler_phrases = ['please', 'could you', 'would you', 'can you', 'i want to', 'show me', 'give me', 'list me', 'tell me']
    for phrase in filler_phrases:
        query = query.replace(phrase, '')
    
    # Remove extra spaces
    query = ' '.join(query.split())
    
    return query

def natural_to_sql(natural_query, table_schema):

    query = preprocess_query_sql(natural_query)

    print("Preprocessed Query:", query)
    print()
    doc = nlp(query)
    
    action = "SELECT" if 'get' in query else "Unsupported action."

    if action == "Unsupported action.":
        return "Hi Please give relevant input"
    
    columns, aggregate_function = extract_columns(doc, table_schema)

    if not columns:
        columns = '*'
    tables = extract_tables(doc, table_schema)
    
    # Build the SQL query
    sql_query = f"{action} "

    sql_query += f"{', '.join(columns)} "
    
    sql_query += f"FROM {tables['main']} "
    
    # Add JOINs if any
    join_clause = None
    if tables['joins']:
        join_clause = handle_joins(doc, tables, table_schema)
    if join_clause:
        sql_query += f"{join_clause} "
    
    # Add WHERE clause if any
    where_condition = extract_where(doc, table_schema)
    if where_condition:
        # Remove leading 'AND' or 'OR' if present
        if where_condition.startswith('AND '):
            where_condition = where_condition[4:]
        elif where_condition.startswith('OR '):
            where_condition = where_condition[3:]
        sql_query += f"WHERE {where_condition} "
    
    # Add GROUP BY clause if any
    group_by_clause = extract_group_by(doc, table_schema)
    if group_by_clause:
        sql_query += f"GROUP BY {group_by_clause} "
    
    # Add ORDER BY clause if needed
    order_column, order_direction = extract_order_by(doc, table_schema)
    if order_column and order_direction:
        sql_query += f"ORDER BY {order_column} {order_direction} "
    
    # Add LIMIT clause if any
    limit_clause = extract_limit(doc)
    if limit_clause:
        sql_query += f"LIMIT {limit_clause} "
    
    # Clean up and return the query
    sql_query = sql_query.strip()
    # Uncomment the next line to see the generated SQL query
    # print("Generated SQL Query:", sql_query + ";")
    return sql_query + ";"

def extract_columns(doc, table_schema):
    columns = []
    aggregate_functions = []
    cols_without_tables = []
    # Find the action verb (e.g., 'get')
    action_token = None
    for token in doc:
        if token.lemma_ == 'get':
            action_token = token
            break

    # print("Action token:", action_token)
    
    # Gather all tokens related to selecting columns
    tokens_to_check = []
    if action_token:
        # Traverse all children of the action to gather direct objects and other related information
        for child in action_token.children:
            # print("Child of Action token:", child)
            if child.dep_ in ('dobj', 'attr', 'pobj', 'conj', 'nsubj', 'nmod'):
                # Traverse the subtree to collect relevant tokens, excluding prepositional phrases
                for t in child.subtree:
                    if not any(ancestor.text.lower() in ['with', 'by', 'having'] for ancestor in t.ancestors):
                        tokens_to_check.append(t)
    else:
        tokens_to_check = doc

    # Also add the tokens that are connected by logical operators like 'or' or 'and'
    for token in doc:
        if token.dep_ in ('conj', 'cc') and token.head in tokens_to_check:
            tokens_to_check.append(token)
            for t in token.subtree:
                if t not in tokens_to_check:
                    tokens_to_check.append(t)

    # print("Tokens to check:", tokens_to_check)

    # Extract columns based on tokens
    for table, columns_in_table in table_schema.items():
        for column in columns_in_table:
            for token in tokens_to_check:
                if column.lower() == token.text.lower() and column.lower() not in cols_without_tables:
                    cols_without_tables.append(f"{column.lower()}")
                    columns.append(f"{table}.{column}")

    # Remove duplicate columns
    # cols_without_tables = list(set(cols_without_tables))
    columns = list(set(columns))

    # Handle aggregate functions (SUM, AVG, COUNT)
    if 'sum' in doc.text or 'total' in doc.text:
        column = extract_column_after_aggregate(doc, table_schema, 'sum') or extract_column_after_aggregate(doc, table_schema, 'total') 
        if column:
            aggregate_functions.append(f"SUM({column})")
    if 'average' in doc.text or 'avg' in doc.text:
        column = extract_column_after_aggregate(doc, table_schema, 'average') or extract_column_after_aggregate(doc, table_schema, 'avg')
        if column:
            aggregate_functions.append(f"AVG({column})")
    if 'count' in doc.text or 'no.of' in doc.text:
        column = extract_column_after_aggregate(doc, table_schema, 'count') or extract_column_after_aggregate(doc, table_schema, 'no.of')
        aggregate_functions.append(f"COUNT({column})" if column else "COUNT(*)")

    # If aggregate functions are found, replace columns with aggregate functions
    if aggregate_functions:
        columns.extend(aggregate_functions)
    
    # print(columns)
    return columns, ', '.join(aggregate_functions)

def extract_column_after_aggregate(doc, table_schema, aggregate_word):
    for token in doc:
        if token.text.lower() == aggregate_word:
            next_tokens = [t for t in doc[token.i + 1 : token.i + 3]]
            for next_token in next_tokens:
                for table, columns in table_schema.items():
                    if next_token.text.lower() in [col.lower() for col in columns]:
                        return f"{table}.{next_token.text}"
    return None

def extract_tables(doc, table_schema):
    tables = {'main': '', 'joins': []}

    # First, check for explicit table mentions
    for table in table_schema.keys():
        if table.lower() in doc.text.lower():
            tables['main'] = table
            break

    # If no table is explicitly mentioned, infer from columns
    if not tables['main']:
        mentioned_columns = extract_columns_from_query(doc, table_schema)
        if mentioned_columns:
            tables['main'] = mentioned_columns[0].split('.')[0]

    # print("main_table:",tables['main'])
    
    main_table_col = []
    for table,columns in table_schema.items():
        if table == tables['main']:
            for col in columns:
                main_table_col.append(f"{col}")
    # print("main_table_cols:",main_table_col)

    # Find any tables needed for joins
    mentioned_columns = extract_columns_from_query(doc, table_schema)
    # print("Mentioned_columns:",mentioned_columns)

    # if all of the mentioned cols not in main_table_cols then join should perform
    cols = []
    for value in mentioned_columns:
        cols.append(value.split('.')[1])

    cols = list(set(cols))
    # print("Mentioned_cols2:",cols)

    are_all_present = all(item in  main_table_col for item in cols)

    # print(are_all_present)

    if not are_all_present:
        for value in mentioned_columns:
            if value not in main_table_col:
                other_table = value.split('.')[0]
                # print("Other_tables:",other_table)
                if other_table != tables['main'] and other_table not in tables['joins']:
                    tables['joins'].append(other_table)

    return tables

def extract_columns_from_query(doc, table_schema):
    mentioned_columns = []

    for token in doc:
        for table, columns in table_schema.items():
            for col in columns:
                if token.text.lower() == col.lower():
                    mentioned_columns.append(f"{table}.{col}")

    return mentioned_columns

def handle_joins(doc, tables, table_schema):
    join_clauses = []
    main_table = tables['main']
    join_tables = tables['joins']
    joined_tables = set()
    joined_tables.add(main_table)

    while join_tables:
        join_made = False
        for join_table in join_tables:
            for from_table in joined_tables:
                common_key = get_common_key(from_table, join_table, table_schema)
                if common_key:
                    join_clause = f"JOIN {join_table} ON {from_table}.{common_key} = {join_table}.{common_key}"
                    join_clauses.append(join_clause)
                    joined_tables.add(join_table)
                    join_tables.remove(join_table)
                    join_made = True
                    break
            if join_made:
                break
        if not join_made:
            # No way to join remaining tables
            break

    return " ".join(join_clauses)

def get_common_key(table1, table2, table_schema):
    for column1 in table_schema[table1]:
        for column2 in table_schema[table2]:
            if column1 == column2:
                return column1
    return None

def extract_where(doc, table_schema):
    where_conditions = []
    relational_ops = {
        'equals': '=', 
        'is': '=', 
        'greaterthan': '>',
        'morethan': '>',
        'lessthan': '<',
        'less': '<',
        'more': '>',
        '>': '>',
        '<': '<',
        '=': '=',
        'like': 'LIKE'
    }

    i = 0
    while i < len(doc):
        is_num=False
        token = doc[i]
        
        if token.text.lower() in ['and', 'or']:
            where_conditions.append(token.text.upper())
            i += 1
            continue
        
        if token.text.lower() == 'between':  # Handle BETWEEN functionality
            left_token = doc[i - 1] if i - 1 >= 0 else None
            left_operand = left_token.text if left_token else ''
            between_value1 = doc[i + 1].text if i + 1 < len(doc) else ''
            and_token = doc[i + 2] if i + 2 < len(doc) else None
            between_value2 = doc[i + 3].text if i + 3 < len(doc) else ''
            
            if and_token and and_token.text.lower() == 'and':
                for table, columns in table_schema.items():
                    if left_operand.lower() in [col.lower() for col in columns]:
                        left_operand = f"{table}.{left_operand}"
                        break
                where_conditions.append(f"{left_operand} BETWEEN {between_value1} AND {between_value2}")
                i += 4
                continue

        if token.text.lower() == 'like':  # Handle LIKE functionality with NLP context
            left_token = doc[i - 2] if i - 2 >= 0 else None
            left_operand = left_token.text if left_token else ''
            pattern_token = doc[i + 1] if i + 1 < len(doc) else None
            pattern = pattern_token.text if pattern_token else ''

            # print("Pattern:",pattern)

            # Determine context from previous or next tokens
            previous_text = doc[i - 1].text.lower() if i - 1 >= 0 else ''
            if previous_text in ['starts', 'begins', 'prefix']:
                pattern = f"{pattern}%"
            elif previous_text in ['ends', 'suffix']:
                pattern = f"%{pattern}"
            elif previous_text in ['contains', 'includes']:
                pattern = f"%{pattern}%"

            for table, columns in table_schema.items():
                if left_operand.lower() in [col.lower() for col in columns]:
                    left_operand = f"{table}.{left_operand}"
                    break
            where_conditions.append(f"{left_operand} LIKE '{pattern}'")
            i += 2 
            continue

        if token.text.lower() in relational_ops.keys() or token.text in relational_ops.values():
            operator = relational_ops.get(token.text.lower(), token.text)
            left_token = doc[i - 1] if i - 1 >= 0 else None
            left_operand = left_token.text if left_token else ''
            right_token = doc[i + 1] if i + 1 < len(doc) else None
            if right_token.pos_ == "NUM" or right_token.like_num:
                is_num = True
            right_operand = right_token.text if right_token else ''
            
            for table, columns in table_schema.items():
                for col in columns:
                    if left_operand.lower() == col.lower():
                        left_operand = f"{table}.{col}"
                        break
                # if left_operand.lower() in [col.lower() for col in columns]:
                #     left_operand = f"{table}.{left_operand}"
                #     break 

            if is_num:
                 where_conditions.append(f"{left_operand} {operator} {right_operand}")
            else:
                where_conditions.append(f"{left_operand} {operator} '{right_operand}'")
            i += 2
            continue
        
        i += 1

    # print("Where_conditions:", where_conditions)
    return ' '.join(where_conditions)


def extract_group_by(doc, table_schema):
    group_by_columns = []

    for token in doc:
        if token.text.lower() in ["group", "by", "each"]:
            next_tokens = [doc[token.i + i] for i in range(1, 4) if token.i + i < len(doc)]
            for next_token in next_tokens:
                for table, columns in table_schema.items():
                    for col in columns:
                        if next_token.text.lower() == col.lower():
                            group_by_columns.append(f"{table}.{col}")
                            break

    group_by_clause = ", ".join(group_by_columns)
    return group_by_clause

def extract_order_by(doc, table_schema):
    order_column = None
    order_direction = None
    for token in doc:
        if token.text.lower() in ["highest", "largest", "top", "maximum"]:
            order_direction = "DESC"
        elif token.text.lower() in ["lowest", "smallest", "minimum"]:
            order_direction = "ASC"
        
        if order_direction:
            for table, col_list in table_schema.items():
                for col in col_list:
                    if col.lower() in [t.text.lower() for t in token.subtree]:
                        order_column = f"{table}.{col}"
                        break
            if order_column:
                break

    # If order_column is still None, try to find 'Marks' or 'Score' in the query
    if not order_column:
        for token in doc:
            if token.text.lower() in ['marks', 'score']:
                for table, col_list in table_schema.items():
                    if token.text in col_list:
                        order_column = f"{table}.{token.text}"
                        break
                if order_column:
                    break

    return order_column, order_direction

def extract_limit(doc):
    for token in doc:
        if token.text.lower() in ["limit", "top", "first"]:
            # Look for a number after the token
            next_tokens = [doc[token.i + i] for i in range(1, 3) if token.i + i < len(doc)]
            for next_token in next_tokens:
                if next_token.pos_ == "NUM" or next_token.like_num:
                    return next_token.text
            # Look for a number before the token
            prev_tokens = [doc[token.i - i] for i in range(1, 3) if token.i - i >= 0]
            for prev_token in prev_tokens:
                if prev_token.pos_ == "NUM" or prev_token.like_num:
                    return prev_token.text
    return None




# NoSQL query execution logic

# Intent and keyword mappings
INTENT_KEYWORDS = {
    "find": ["find", "search", "retrieve", "get", "show", "display", "list", "fetch"],
    "insert": ["insert", "add", "create", "new"],
    "update": ["update", "modify", "change", "set"],
    "delete": ["delete", "remove", "drop"],
    "aggregate": ["sum", "average", "avg", "aggregate", "total", "count", "max", "min", "group by", "group", "number"],
    "sort": ["sort", "order", "arrange", "highest", "lowest", "max", "min", "top", "bottom", "most", "least"],
    "filter": ["where", "filter", "with", "having", "in", "on", "by", "between", "before", "after", "and", "or", "not"],
    "limit": ["limit", "top", "first", "last"],
    "project": ["show", "display", "project", "select", "only"],
    "join": ["join", "lookup", "merge", "include", "combine", "together"],
}

LOGICAL_OPERATORS = {
    "and": "$and",
    "or": "$or",
    "not": "$not",
    "nor": "$nor"
}

COMPARISON_OPERATORS = {
    "is greater than or equal to": "$gte",
    "is less than or equal to": "$lte",
    "is greater than": "$gt",
    "is less than": "$lt",
    "is not equal to": "$ne",
    "is equal to": "$eq",
    "equals": "$eq",
    "equal to": "$eq",
    "equal": "$eq",
    "greater than or equal to": "$gte",
    "less than or equal to": "$lte",
    "greater than": "$gt",
    "less than": "$lt",
    "not equal to": "$ne",
    "before": "$lt",
    "after": "$gt",
    "between": "$between",
    "in": "$in",
    "is": "$eq",
    ">=": "$gte",
    "<=": "$lte",
    ">": "$gt",
    "<": "$lt",
    "!=": "$ne",
    "=": "$eq",
    "than": "",  # To handle phrases like 'greater than'
}


def suggest_query_format(user_query):
    return (
        "To improve results, try using standardized queries like:\n"
        "- 'Find all customers where state is California'\n"
        "- 'Find customers in California state'\n"
        "- 'Group customers by state'\n"
        "- 'Sort customers by city'\n"
        "- 'Update customers set email to example@example.com where first name is John'\n"
        "- 'Delete orders where customer id is in 1,2,3'\n"
        "- 'Find top 5 products with highest list price'\n"
        "- 'Count the number of orders per customer'\n"
        "- 'Find customers where city is New York or Los Angeles'\n"
        "- 'Join customers and orders on customer_id'"
    )

def map_attribute(phrase, attributes):
    # Map phrases to attribute names, accounting for underscores and spaces
    attribute_map = {}
    for attr in attributes:
        attr_normalized = attr.replace('_', ' ').lower()
        attribute_map[attr_normalized] = attr

    # Add custom mappings if necessary
    custom_mappings = {
        'total amount': 'total_amount',
        'amount': 'total_amount',
        'order total': 'order_total',
        'price': 'list_price',
        'first name': 'first_name',
        'last name': 'last_name',
        'required date': 'required_date',
        'product category': 'category_id',
        'category id': 'category_id',
        'product name': 'product_name',
        'zip code': 'zip_code',
        'customer id': 'customer_id',
        'product id': 'product_id',
        'order id': 'order_id',
        'list price': 'list_price',
        'model year': 'model_year',
        'email': 'email',  # Added email mapping
        'name': 'name',
        'order status': 'order_status',
        'orders': 'orders',
        'products': 'products',
        'customer': 'customer_id',  # Map 'customer' to 'customer_id'
    }
    attribute_map.update(custom_mappings)
    return attribute_map.get(phrase.lower())

def preprocess_query(query):
    query = re.sub(r'\bi\s+d\b', 'id', query, flags=re.IGNORECASE)
    doc = nlp(query)
    lemmas = [token.lemma_.lower() for token in doc]
    pos_tags = [token.pos_ for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    logging.info(f"Tokens: {[token.text for token in doc]}")
    logging.info(f"Lemmas: {lemmas}")
    logging.info(f"POS Tags: {pos_tags}")
    logging.info(f"Entities: {entities}")

    return {
        'tokens': [token.text for token in doc],
        'lemmas': lemmas,
        'pos_tags': pos_tags,
        'entities': entities,
        'doc': doc
    }

def generate_mongodb_command_string(processed_query):
    logging.info(f"Processing query: {processed_query}")
    tokens = [token.lower() for token in processed_query['tokens']]
    lemmas = processed_query['lemmas']
    doc = processed_query['doc']

    # Identify collection
    collection = None
    database = None

    # Get all databases
    databases = mongo_client.list_database_names()
    for db_name in databases:
        db = mongo_client[db_name]
        collections = db.list_collection_names()
        for col in collections:
            if col.lower() in tokens or col.lower() in lemmas:
                collection = col
                database = db_name
                break
        if collection:
            break

    if not collection:
        # Try to infer collection from known mappings
        for token in tokens:
            potential_col = token.lower()
            if potential_col in ['customers', 'orders', 'products']:
                collection = potential_col
                database = 'chatdb_nosql'
                print(f"collection:{collection}")
                break

    if not collection:
        logging.error("Collection not found.")
        return {"error": "Collection not found", "suggestion": suggest_query_format(' '.join(tokens))}
    logging.info(f"Identified collection: {collection} in database: {database}")

    # Connect to the identified database
    mongo_db = mongo_client[database]

    # Fetch collection attributes
    try:
        sample_doc = mongo_db[collection].find_one()
        if sample_doc:
            attributes = list(sample_doc.keys())
        else:
            logging.error("No documents found in collection.")
            return {"error": f"No documents found in {collection}."}
    except Exception as e:
        logging.error(f"Error accessing collection attributes: {str(e)}")
        return {"error": str(e)}
    logging.info(f"Attributes in collection '{collection}': {attributes}")

    # Add custom attributes if necessary
    if collection == 'customers':
        if 'email' not in attributes:
            attributes.append('email')
        if 'first_name' not in attributes:
            attributes.append('first_name')
    if collection == 'products':
        if 'list_price' not in attributes:
            attributes.append('list_price')
    if collection == 'orders':
        if 'customer_id' not in attributes:
            attributes.append('customer_id')

    # Extract filters
    filters = extract_filters(processed_query, attributes)

    # Match intents
    intents = match_intents(processed_query)
    logging.info(f"Matched intents: {intents}")

    # Build MongoDB command string
    mongo_command = ""
    if 'join' in intents:
        # Handle join operation using $lookup
        tokens_lower = [t.lower() for t in processed_query['tokens']]
        and_indices = [i for i, x in enumerate(tokens_lower) if x == 'and']
        on_indices = [i for i, x in enumerate(tokens_lower) if x == 'on']

        if and_indices and on_indices:
            first_collection = collection
            second_collection = tokens_lower[and_indices[0] + 1]
            join_field = tokens_lower[on_indices[0] + 1]

            mongo_command = f"db.{first_collection}.aggregate([{{'$lookup': {{'from': '{second_collection}', 'localField': '{join_field}', 'foreignField': '{join_field}', 'as': '{second_collection}'}}}}])"
        else:
            return {"error": "Unable to construct join operation."}

    elif 'find' in intents or 'count' in tokens or 'count' in lemmas:
        filter_str = json.dumps(filters) if filters else "{}"
        mongo_command = f"db.{collection}.find({filter_str})"

        # Handle count
        if 'count' in tokens or 'count' in lemmas:
            # Check if it's a group count or total count
            if 'per' in tokens or 'by' in tokens:
                # Group count
                group_field = None
                for idx, token in enumerate(doc):
                    if token.text.lower() in ['per', 'by'] and idx + 1 < len(doc):
                        potential_field = ' '.join([doc[i].text.lower() for i in range(idx + 1, len(doc))])
                        group_field = map_attribute(potential_field.strip(), attributes)
                        break
                if group_field:
                    pipeline = []
                    if filters:
                        pipeline.append({"$match": filters})
                    pipeline.append({
                        "$group": {
                            "_id": f"${group_field}",
                            "count": {"$sum": 1}
                        }
                    })
                    mongo_command = f"db.{collection}.aggregate({json.dumps(pipeline)})"
                else:
                    return {"error": "Unable to determine grouping field for count."}
            else:
                # Total count
                mongo_command += ".countDocuments()"
        # Handle sorting
        if 'sort' in intents or any(token in ['top', 'highest', 'lowest', 'most', 'least'] for token in tokens):
            sort_field = None
            sort_order = -1  # Default to descending
            for idx, token in enumerate(doc):
                if token.text.lower() in INTENT_KEYWORDS['sort'] or token.text.lower() in ['top', 'highest', 'most']:
                    sort_order = -1
                elif token.text.lower() in ['lowest', 'min', 'minimum', 'least']:
                    sort_order = 1

                if token.text.lower() in ['with', 'by']:
                    if idx + 1 < len(doc):
                        potential_field = ' '.join([doc[i].text.lower() for i in range(idx + 1, len(doc))])
                        sort_field = map_attribute(potential_field.strip(), attributes)
                        break
                elif token.text.lower() in ['highest', 'lowest', 'most', 'least', 'top']:
                    if idx + 2 < len(doc):
                        potential_field = ' '.join([doc[i].text.lower() for i in range(idx + 2, len(doc))])
                        sort_field = map_attribute(potential_field.strip(), attributes)
                        break
            if sort_field:
                mongo_command += f".sort({{{json.dumps(sort_field)}: {sort_order}}})"

        # Handle limit
        if 'limit' in intents or 'top' in tokens:
            limit_value = None
            for idx, token in enumerate(doc):
                if token.text.lower() in INTENT_KEYWORDS['limit'] or token.text.lower() == 'top':
                    # Look for number after 'top' or 'limit'
                    if idx + 1 < len(doc) and doc[idx + 1].pos_ == 'NUM':
                        limit_value = int(doc[idx + 1].text)
                    elif idx + 2 < len(doc) and doc[idx + 2].pos_ == 'NUM':
                        limit_value = int(doc[idx + 2].text)
                    break
            if limit_value:
                mongo_command += f".limit({limit_value})"

    elif 'update' in intents:
        # Handle update command
        update_fields = {}
        set_idx = None
        where_idx = None
        for idx, token in enumerate(doc):
            if token.text.lower() == 'set':
                set_idx = idx
            elif token.text.lower() == 'where':
                where_idx = idx

        if set_idx is not None:
            i = set_idx + 1
            while i < len(doc) and (where_idx is None or i < where_idx):
                attr_name = None
                max_attr_len = min(3, len(doc) - i)
                for l in range(max_attr_len, 0, -1):
                    potential_attr = ' '.join([doc[i + k].text.lower() for k in range(l)])
                    mapped_attr = map_attribute(potential_attr, attributes)
                    if mapped_attr:
                        attr_name = mapped_attr
                        i += l
                        break
                if attr_name:
                    # Look for 'to' and then value
                    if i < len(doc) and doc[i].text.lower() == 'to':
                        i += 1
                        value_tokens = []
                        while i < len(doc) and doc[i].text.lower() not in ['where'] and doc[i].pos_ not in ['PUNCT']:
                            value_tokens.append(doc[i].text)
                            i += 1
                        value = ' '.join(value_tokens).strip()

                        # Check for arithmetic operation
                        if '*' in value:
                            parts = value.split('*')
                            if len(parts) == 2:
                                field_part = parts[0].strip()
                                number_part = parts[1].strip()
                                try:
                                    multiplier = float(number_part)
                                    update_fields["$mul"] = {attr_name: multiplier}
                                except ValueError:
                                    pass
                        else:
                            # Regular update
                            if value.isdigit():
                                value = int(value)
                            else:
                                try:
                                    value = float(value)
                                except ValueError:
                                    value = value.strip('"')  # Remove quotes if any
                            update_fields.setdefault("$set", {})[attr_name] = value
                    else:
                        i += 1
                else:
                    i += 1

            # Extract filters after 'where'
            if where_idx is not None:
                filter_doc = nlp(' '.join([token.text for token in doc[where_idx+1:]]))
                filter_processed_query = {
                    'tokens': [token.text for token in filter_doc],
                    'lemmas': [token.lemma_.lower() for token in filter_doc],
                    'doc': filter_doc,
                    'entities': [(ent.text, ent.label_) for ent in filter_doc.ents]
                }
                filters = extract_filters(filter_processed_query, attributes)
            else:
                filters = {}

            if update_fields:
                update_str = json.dumps(update_fields)
                filter_str = json.dumps(filters) if filters else "{}"
                mongo_command = f"db.{collection}.updateMany({filter_str}, {update_str})"
            else:
                return {"error": "No fields provided to update."}

    elif 'delete' in intents:
        # Handle delete command
        filters = extract_filters(processed_query, attributes)
        if not filters:
            return {"error": "No filters provided for delete operation to prevent deleting all documents."}
        filter_str = json.dumps(filters)
        mongo_command = f"db.{collection}.deleteMany({filter_str})"

    elif 'aggregate' in intents:
        # Handle aggregation commands
        group_field = None
        agg_operation = "$sum"
        agg_field = 1  # Default to count
        for idx, token in enumerate(doc):
            if token.text.lower() in ['per', 'by']:
                if idx + 1 < len(doc):
                    potential_field = ' '.join([doc[i].text.lower() for i in range(idx + 1, len(doc))])
                    group_field = map_attribute(potential_field.strip(), attributes)
                    break

        if group_field:
            pipeline = []
            filters = extract_filters(processed_query, attributes)
            if filters:
                pipeline.append({"$match": filters})
            group_stage = {
                "$group": {
                    "_id": f"${group_field}",
                    "count": {agg_operation: agg_field}
                }
            }
            pipeline.append(group_stage)
            mongo_command = f"db.{collection}.aggregate({json.dumps(pipeline)})"
        else:
            return {"error": "Unable to determine grouping field for count."}
    else:
        return {"error": "Unsupported query type", "suggestion": suggest_query_format(' '.join(tokens))}

    # Create a natural language description
    nl_description = "Generated MongoDB command for your query."
    # print (mongo_command)
    if mongo_command:
        return {
            "mongodb_command": mongo_command,
            "natural_language_description": "Generated Mongo command for your query."
        }
    else:
        return {"error": "Failed to generate MongoDB command"}

def match_intents(processed_query):
    lemmas = processed_query['lemmas']
    intents = []
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in lemmas for keyword in keywords):
            intents.append(intent)
    return intents

def extract_filters(processed_query, attributes):
    filters = {}
    doc = processed_query['doc']

    conditions = []
    logical_ops = []
    i = 0

    while i < len(doc):
        token = doc[i]
        token_text = token.text.lower()

        # handling for 'or'
        if token_text == "or":
            # Use entities to construct the filter directly for 'or' conditions
            entities = processed_query['entities']
            or_conditions = []
            for entity in entities:
                if entity[1] == 'GPE':  # Assuming GPE (Geopolitical Entity) for locations like cities
                    or_conditions.append({"city": entity[0]})
            if or_conditions:
                filters = {"$or": or_conditions}
                return filters

        # handling for 'between' operator
        if token_text == "between":
            attr_name = None
            # Look back for attribute
            max_attr_len = min(3, i)
            for l in range(max_attr_len, 0, -1):
                potential_attr = ' '.join([doc[i - l].text.lower() for l in range(l, 0, -1)])
                mapped_attr = map_attribute(potential_attr.strip(), attributes)
                if mapped_attr:
                    attr_name = mapped_attr
                    break
            if attr_name and i + 2 < len(doc):
                if doc[i + 1].pos_ == "NUM" and doc[i + 2].text.lower() == "and" and doc[i + 3].pos_ == "NUM":
                    start_value = int(doc[i + 1].text)
                    end_value = int(doc[i + 3].text)
                    condition = {attr_name: {"$gte": start_value, "$lte": end_value}}
                    conditions.append(condition)
                    i += 4
                    continue

        #  handling for 'in' operator
        if token_text == "in":
            attr_name = None
            # Look back for attribute
            max_attr_len = min(3, i)
            for l in range(max_attr_len, 0, -1):
                potential_attr = ' '.join([doc[i - l].text.lower() for l in range(l, 0, -1)])
                mapped_attr = map_attribute(potential_attr.strip(), attributes)
                if mapped_attr:
                    attr_name = mapped_attr
                    break
            if attr_name and i + 1 < len(doc):
                # Collect value tokens
                value_tokens = []
                i += 1
                while i < len(doc) and doc[i].text.lower() not in LOGICAL_OPERATORS:
                    value_tokens.append(doc[i].text)
                    i += 1
                value = ' '.join(value_tokens).strip()
                # Split value by commas or 'and'
                values = re.split(r'[\,\s]+', value)
                values = [int(v) if v.isdigit() else v.strip('"') for v in values if v and v.lower() != 'and']
                condition = {attr_name: {"$in": values}}
                conditions.append(condition)
                continue

        # Check for logical operators
        if token_text in LOGICAL_OPERATORS:
            logical_ops.append(LOGICAL_OPERATORS[token_text])
            i += 1
            continue

        # Map attribute phrases
        attr_name = None
        max_attr_len = min(3, len(doc) - i)
        for l in range(max_attr_len, 0, -1):
            potential_attr = ' '.join([doc[i + k].text.lower() for k in range(l)])
            mapped_attr = map_attribute(potential_attr, attributes)
            if mapped_attr:
                attr_name = mapped_attr
                i += l
                break

        if attr_name:
            # Handle comparison operators
            operator = "$eq"
            if i < len(doc):
                # Check for comparison phrases
                max_op_len = min(3, len(doc) - i)
                for l in range(max_op_len, 0, -1):
                    potential_op = ' '.join([doc[i + k].text.lower() for k in range(l)])
                    if potential_op in COMPARISON_OPERATORS:
                        operator = COMPARISON_OPERATORS[potential_op]
                        i += l
                        break
                    elif doc[i].text.lower() in COMPARISON_OPERATORS:
                        operator = COMPARISON_OPERATORS[doc[i].text.lower()]
                        i += 1
                        break
            # Collect value
            value_tokens = []
            while i < len(doc) and doc[i].text.lower() not in LOGICAL_OPERATORS and doc[i].text.lower() not in COMPARISON_OPERATORS:
                value_tokens.append(doc[i].text)
                i += 1

            value = ''.join(value_tokens).strip()

            # Handle 'between' operator
            if operator == "$between":
                nums = re.findall(r'\d+', value)
                if len(nums) == 2:
                    condition = {attr_name: {"$gte": int(nums[0]), "$lte": int(nums[1])}}
                else:
                    condition = {attr_name: {"$eq": value}}
            # Handle 'in' operator
            elif operator == "$in":
                values = re.split(r'[\,\s]+', value)
                values = [int(v) if v.isdigit() else v.strip('"') for v in values if v and v.lower() != 'and']
                condition = {attr_name: {"$in": values}}
            else:
                # Handle date parsing for 'before' and 'after'
                if operator in ["$lt", "$gt", "$lte", "$gte"]:
                    try:
                        value_date = date_parser.parse(value, fuzzy=True)
                        value = value_date.isoformat()
                    except ValueError:
                        pass  # Not a date
                else:
                    # Convert numerical values
                    if value.isdigit():
                        value = int(value)
                    else:
                        try:
                            value = float(value)
                        except ValueError:
                            value = value.strip('"')  # Remove quotes if any
                condition = {attr_name: {operator: value}}

            conditions.append(condition)
        else:
            i += 1

    # Combine conditions with logical operators
    if conditions:
        if len(conditions) == 1:
            filters = conditions[0]
        else:
            # Combine conditions based on logical operators
            if logical_ops and logical_ops[0] == '$or':
                filters = {"$or": conditions}
            else:
                filters = {"$and": conditions}
    else:
        filters = {}

    logging.info(f"Extracted filters: {filters}")
    return filters



if __name__ == "__main__":
    app.run(debug=True)
