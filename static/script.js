let selectedDbType = "";
let selectedDataSource = "";

function navigateFromPage1() {
    const dbTypeSelect = document.getElementById('dbTypeSelect');
    const dataSourceSelect = document.getElementById('dataSourceSelect');

    selectedDbType = dbTypeSelect.value;
    selectedDataSource = dataSourceSelect.value;

    if (!selectedDbType || !selectedDataSource) {
        alert('Please select both a database type and a data source.');
        return;
    }

    if (selectedDataSource === "Available") {
        populateAvailableDatasets();
        showPage('page2Available');
    } else if (selectedDataSource === "Upload") {
        showPage('page2Upload');
    }
}

function populateAvailableDatasets() {
    const datasetSelect = document.getElementById('datasetSelect');
    datasetSelect.innerHTML = '<option value="">Choose from available datasets</option>';

    const datasets = selectedDbType === "SQL" 
        ? ["Sql_db"] 
        : ["chatdb_nosql"];

    datasets.forEach(dataset => {
        const option = document.createElement('option');
        option.value = dataset;
        option.textContent = dataset;
        datasetSelect.appendChild(option);
    });
}

function finalizeSelection() {
    const datasetSelect = document.getElementById('datasetSelect');
    const datasetName = datasetSelect.value;

    if (!datasetName) {
        alert('Please select a dataset.');
        return;
    }

    // Fetch schemas for all tables in the selected database
    


    fetch('/schema', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
            db_type: selectedDbType,  // Pass the selected database type (SQL/NoSQL)
            dataset_name: datasetName // Pass the dataset name (database name)
        })
    })
    .then(response => response.json())
    .then(data => {
        // Check if the response contains SQL schema data
        if (selectedDbType === "SQL") {
            displaySchemas(data.schemas);  // Display the schema for SQL databases
        } 
        // Check if the response contains NoSQL schema data
        else if (selectedDbType === "NoSQL") {
            displaySchema(data.schema);  // Display the schema for NoSQL databases
        } 
        // If an error message is returned in the response
        else {
            alert(`Error: ${data.error}`);  // Display the error message
        }
    })
    .catch(error => {
        // Handle network or other errors
        console.error('Error fetching schema:', error);
        alert('An error occurred while fetching the schema.');
    });


    showPage('page3');
}
function displayUploadedSchema(schema) {
    const schemaDisplay = document.getElementById('schemaDisplay');
    schemaDisplay.innerHTML = '<h3>Uploaded File Schema:</h3>';

    if (schema.type === "SQL") {
        schemaDisplay.innerHTML += '<ul>';
        for (const column of schema.columns) {
            schemaDisplay.innerHTML += `<li>${column}</li>`;
        }
        schemaDisplay.innerHTML += '</ul>';
    } else if (schema.type === "NoSQL") {
        schemaDisplay.innerHTML += '<ul>';
        for (const key of schema.keys) {
            schemaDisplay.innerHTML += `<li>${key}</li>`;
        }
        schemaDisplay.innerHTML += '</ul>';
    }

    showPage('page3');
}
function displaySchemas(schemas) {
    const schemaDisplay = document.getElementById('schemaDisplay');
    schemaDisplay.innerHTML = `
        <h3>Schemas for Selected Database:</h3>
        ${Object.keys(schemas).map(table => `
            <div>
                <h4>Table: ${table}</h4>
                <ul>
                    ${schemas[table].columns.map(col => `<li>${col.name}: ${col.type}</li>`).join("")}
                </ul>
            </div>
        `).join("")}
    `;
}

function displaySchema(schema) {
    const schemaDisplay = document.getElementById('schemaDisplay');
    schemaDisplay.innerHTML = '<h3>NoSQL Schema:</h3>';
    for (const [collection, fields] of Object.entries(schema)) {
        schemaDisplay.innerHTML += `<h4>${collection}</h4><ul>`;
        for (const [field, type] of Object.entries(fields)) {
            schemaDisplay.innerHTML += `<li>${field}: ${type}</li>`;
        }
        schemaDisplay.innerHTML += '</ul>';
    }
}

function showPage(pageId) {
    const pages = document.querySelectorAll('.page');
    pages.forEach(page => {
        page.style.display = 'none';
    });
    document.getElementById(pageId).style.display = 'block';
}

document.getElementById('userInput').addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        sendQuery();
    }
});

function sendQuery() {
    const userInput = document.getElementById('userInput');
    const chatOutput = document.getElementById('chatOutput');
    const query = userInput.value.trim();
    const datasetSelect = document.getElementById('datasetSelect');
    const selectedDataset = datasetSelect.value;

    if (query && selectedDataset) {
        // Display the user's input in the chat window
        chatOutput.innerHTML += `<p><strong>You:</strong> ${query}</p>`;

        fetch('/execute-query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                query: query, 
                db_type: selectedDbType, 
                dataset_name: selectedDataset 
            })
        })
        .then(response => response.json())
        .then(data => {
            // Handle the response and update the chat output
            if (data.results) {
                chatOutput.innerHTML += `<p><strong>ChatDB:</strong> ${JSON.stringify(data.results, null, 2)}</p>`;
            }
            if (data.sql_query) {
                chatOutput.innerHTML += `<p><strong>ChatDB:</strong> ${data.sql_query}</p>`;
            } 
            if (data.mongodb_command) {
                chatOutput.innerHTML += `<p><strong>ChatDB:</strong> ${data.mongodb_command}</p>`;
            } 
            if (data.error) {
                chatOutput.innerHTML += `<p><strong>ChatDB:</strong> Error: ${data.error}</p>`;
            }

            chatOutput.scrollTop = chatOutput.scrollHeight;
        })
        .catch(error => {
            console.error('Error:', error);
            chatOutput.innerHTML += `<p><strong>ChatDB:</strong> An error occurred while processing the query.</p>`;
        });

        userInput.value = '';
    }
}

// document.getElementById('proceedToQueryBtn').addEventListener('click', function() {
//     const chatOutput = document.getElementById('chatOutput');
//     const userInput = document.getElementById('userInput');
//     const query = userInput.value.trim();

//     if (!query) {
//         alert('Please enter a query.');
//         return;
//     }

//     fetch('/execute-query', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json'
//         },
//         body: JSON.stringify({
//             query: query,
//             db_type: selectedDbType,
//             dataset_name: selectedDataset
//         })
//     })
//     .then(response => response.json())
//     .then(data => {
//         chatOutput.innerHTML += `
// <p><strong>You:</strong> ${query}</p>
// <p><strong>ChatDB:</strong> ${JSON.stringify(data, null, 2)}</p>
// `;
//         chatOutput.scrollTop = chatOutput.scrollHeight;
//     })
//     .catch(error => {
//         console.error('Error:', error);
//         chatOutput.innerHTML += `
// <p><strong>ChatDB:</strong> Error: An error occurred while processing the query.</p>
// `;
//     });

//     userInput.value = '';
// });


document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a file to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload-database', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        if (data.schema) {
            displayUploadedSchema(data.schema);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while uploading the file.');
    });
});

