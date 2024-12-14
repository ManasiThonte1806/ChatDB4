# **ChatDB4**

ChatDB4 is a user-friendly application designed to generate queries for SQL and NoSQL databases. Follow the steps below to set up and run the application.

## **Getting Started**

### **How to Run**

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd ChatDB4
2. Start the application
   ```bash
   python app.py
3. The terminal will display the hosting address (e.g., http://127.0.0.1:5000).
4. Open the provided address in your browser to access ChatDB.

## **File Structure**

  The repository is organized as follows:
  ```bash
  ChatDB4/
  ├── Data/                 # contains the csv and json data files
  │   ├── customers.json
  │   ├── database-courses.csv
  │   ├── database-instructors.csv
  │   ├── database-students.csv
  │   ├── orders.json
  │   └── products.json
  ├── Uploads/
  │   └── app.py            # Application logic for the web browser
  ├── static/               # Frontend assets
  │   ├── script.js
  │   └── style.css
  ├── templates/
  │   └── index.html        # Main HTML file for the web page
  └── ReadME.md             # Project documentation

