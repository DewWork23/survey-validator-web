# Survey Validator Web Application

A web application for validating and processing survey responses, with support for IP whitelisting and geographic validation.

## Features

- Upload and process community and incentive survey responses
- IP address validation and whitelisting
- Geographic distance validation
- Detailed reporting and statistics
- Excel export with multiple sheets
- User authentication and session management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/survey-validator-web.git
cd survey-validator-web
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Unix/MacOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python run.py
```

The application will be available at `http://localhost:5000`

## Configuration

The application uses a JSON configuration file for validation settings. A default configuration is provided in `app/static/config/default_whitelist.json`. You can upload a custom configuration file through the web interface.

## Requirements

- Python 3.8+
- Flask 2.3.3
- Flask-Login 0.6.2
- Flask-WTF 1.1.1
- pandas 2.2.3
- numpy 2.2.5
- openpyxl 3.1.2
- XlsxWriter 3.1.2

## License

[Your chosen license] 