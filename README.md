#  Intelligent Hybrid Recommendation System

An interactive, data-driven web application built with Python and Streamlit that parses user profile registries and logs historical feedback interactions. This project demonstrates advanced backend data pipeline integration, dynamic UI components, and strict data relation management using unique identifiers.

## 🚀 Live Demo
* [Click here to view the live interactive application!](https://streamlit.app)

## 🛠️ Tech Stack & Architecture
* *Frontend UI:* Streamlit (leveraging flexible screen-stretch components)
* *Backend Language:* Python 3
* *Data Storage:* Relational Flat-file CSV database (users.csv, feedback.csv)
* *Environment Management:* Python Virtual Environments (.venv)

## 📦 Core Engineering Features
* *Universally Unique ID (UUID) System:* Implemented a secure, decentralized 8-character alpha-numeric unique token system (user_id) to manage transactional profiles instead of serial indices.
* *User Registry Databank:* Dynamically parses raw profile sheets to resolve algorithmic compatibility parameters.
* *Historical Interaction Logs:* Captures system behaviors and records bidirectional network actions across multi-relational flat tables cleanly.
* *ID-to-Name Mapping Engine:* Built an embedded dictionary parsing lookup loop in ui_app.py to seamlessly convert internal data tokens into friendly human names on the interface view layer.

## 💻 How to Run Locally

1. Clone this repository:
bash
git clone https://github.com


2. Navigate into the project folder:
bash
cd user-feedback-dashboard


3. Install the required dependencies:
bash
pip install -r requirements.txt


4. Start the Streamlit server:
bash
streamlit run ui_app.py
