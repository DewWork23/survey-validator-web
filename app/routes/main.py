import os
import pandas as pd
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_from_directory, current_app, session, Response, send_file
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import sys
import json
import xlsxwriter
from io import BytesIO, StringIO
import csv
import tempfile
import traceback
import difflib
import uuid
import numpy as np
import collections

# Import validate_surveys from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from validate_surveys import process_surveys, load_config, DEFAULT_CONFIG
from app.forms import MatchForm

main = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_default_config():
    """Load the default configuration with whitelist"""
    default_config_path = os.path.join(current_app.root_path, 'static', 'config', 'default_whitelist.json')
    if os.path.exists(default_config_path):
        return load_config(default_config_path)
    return DEFAULT_CONFIG.copy()

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/dashboard')
@login_required
def dashboard():
    # Get list of previous validations
    results_dir = current_app.config['RESULTS_FOLDER']
    validations = []
    
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.endswith('.txt') and 'validation_report' in filename:
                file_path = os.path.join(results_dir, filename)
                file_stat = os.stat(file_path)
                validations.append({
                    'filename': filename,
                    'date': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'path': url_for('main.download_file', filename=filename)
                })
    
    # Sort by date (newest first)
    validations.sort(key=lambda x: x['date'], reverse=True)
    
    return render_template('dashboard.html', validations=validations)

@main.route('/validate', methods=['GET', 'POST'])
@login_required
def validate():
    if request.method == 'POST':
        # Check if files were uploaded
        if 'community_file' not in request.files or 'incentive_file' not in request.files:
            flash('Both community and incentive survey files are required')
            return redirect(request.url)
        
        community_file = request.files['community_file']
        incentive_file = request.files['incentive_file']
        
        # Check if files were selected
        if community_file.filename == '' or incentive_file.filename == '':
            flash('No files selected')
            return redirect(request.url)
        
        # Check if files are allowed
        if not allowed_file(community_file.filename) or not allowed_file(incentive_file.filename):
            flash('Only CSV files are allowed')
            return redirect(request.url)
        
        # Get date range
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        
        if not start_date or not end_date:
            flash('Start and end dates are required')
            return redirect(request.url)
        
        # Save uploaded files
        community_filename = secure_filename(community_file.filename)
        incentive_filename = secure_filename(incentive_file.filename)
        
        community_path = os.path.join(current_app.config['UPLOAD_FOLDER'], community_filename)
        incentive_path = os.path.join(current_app.config['UPLOAD_FOLDER'], incentive_filename)
        
        community_file.save(community_path)
        incentive_file.save(incentive_path)
        
        # Load config with default whitelist
        config = load_default_config()
        
        # Check if config file was uploaded
        if 'config_file' in request.files and request.files['config_file'].filename != '':
            config_file = request.files['config_file']
            if config_file.filename.endswith('.json'):
                config_filename = secure_filename(config_file.filename)
                config_path = os.path.join(current_app.config['UPLOAD_FOLDER'], config_filename)
                config_file.save(config_path)
                user_config = load_config(config_path)
                # Merge user config with default config
                config.update(user_config)
            else:
                flash('Configuration file must be a JSON file')
                return redirect(request.url)
        
        # Process surveys
        try:
            # Clean up the CSV before reading
            with open(community_path, 'r', encoding='utf-8', errors='ignore') as infile, open(os.path.join(current_app.config['UPLOAD_FOLDER'], f"cleaned_{community_filename}"), 'w', encoding='utf-8') as outfile:
                for line in infile:
                    # Remove problematic characters or fix quoting here if needed
                    outfile.write(line.replace('"', '\"'))

            # Now read the cleaned file
            community_df = pd.read_csv(os.path.join(current_app.config['UPLOAD_FOLDER'], f"cleaned_{community_filename}"), sep='\t')
            
            # Clean up the CSV before reading
            with open(incentive_path, 'r', encoding='utf-8', errors='ignore') as infile, open(os.path.join(current_app.config['UPLOAD_FOLDER'], f"cleaned_{incentive_filename}"), 'w', encoding='utf-8') as outfile:
                for line in infile:
                    # Remove problematic characters or fix quoting here if needed
                    outfile.write(line.replace('"', '\"'))

            # Now read the cleaned file
            incentive_df = pd.read_csv(os.path.join(current_app.config['UPLOAD_FOLDER'], f"cleaned_{incentive_filename}"), sep='\t')
            
            eligible_respondents, ineligible_respondents, detailed_statistics = process_surveys(
                community_path, incentive_path, start_date, end_date, config
            )
            
            # Generate timestamp for output files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            
            # Save results
            results_dir = current_app.config['RESULTS_FOLDER']
            
            # Save eligible respondents to CSV
            if eligible_respondents:
                eligible_df = pd.DataFrame(eligible_respondents)
                eligible_filename = f"eligible_respondents_{start_date}_to_{end_date}_{timestamp}.csv"
                eligible_path = os.path.join(results_dir, eligible_filename)
                eligible_df.to_csv(eligible_path, index=False)
                
                # Create Excel workbook with multiple tabs
                excel_filename = f"survey_results_{start_date}_to_{end_date}_{timestamp}.xlsx"
                excel_path = os.path.join(results_dir, excel_filename)
                
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    # Convert Distance column to numeric for filtering
                    eligible_df['distance_numeric'] = eligible_df['Distance'].str.replace(' miles', '').astype(float)
                    
                    # Create "Within Range" tab - ALL respondents within distance threshold
                    distance_threshold = config.get('distance_threshold', 50)
                    within_range = eligible_df[eligible_df['distance_numeric'] <= distance_threshold]
                    within_range = within_range.drop(columns=['distance_numeric'])
                    within_range.to_excel(writer, sheet_name='Within Range', index=False)
                    
                    # Create "Whitelisted Only" tab - ONLY respondents outside distance threshold eligible due to whitelist
                    whitelisted_only = eligible_df[
                        (eligible_df['distance_numeric'] > distance_threshold) & 
                        (eligible_df['IPWhitelisted'] == 'Yes')
                    ]
                    whitelisted_only = whitelisted_only.drop(columns=['distance_numeric'])
                    whitelisted_only.to_excel(writer, sheet_name='Whitelisted Only', index=False)
                    
                    # Create a tab for all whitelisted respondents
                    all_whitelisted = eligible_df[eligible_df['IPWhitelisted'] == 'Yes']
                    all_whitelisted = all_whitelisted.drop(columns=['distance_numeric'])
                    all_whitelisted.to_excel(writer, sheet_name='All Whitelisted', index=False)
                    
                    # Add a summary tab
                    summary_data = {
                        'Category': [
                            'Total Eligible', 
                            f'Within {distance_threshold} miles', 
                            f'Outside {distance_threshold} miles (Whitelisted Only)',
                            'Total Whitelisted (any distance)'
                        ],
                        'Count': [
                            len(eligible_df),
                            len(within_range),
                            len(whitelisted_only),
                            len(all_whitelisted)
                        ]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Export the original data with all columns
                    eligible_df = eligible_df.drop(columns=['distance_numeric'])
                    eligible_df.to_excel(writer, sheet_name='All Eligible', index=False)
            
            # Save ineligible respondents
            if ineligible_respondents:
                ineligible_df = pd.DataFrame(ineligible_respondents)
                ineligible_filename = f"ineligible_respondents_{start_date}_to_{end_date}_{timestamp}.csv"
                ineligible_path = os.path.join(results_dir, ineligible_filename)
                ineligible_df.to_csv(ineligible_path, index=False)
            
            # Save validation report
            report_filename = f"validation_report_{start_date}_to_{end_date}_{timestamp}.txt"
            report_path = os.path.join(results_dir, report_filename)
            
            with open(report_path, 'w') as f:
                f.write("SURVEY VALIDATION AND ELIGIBILITY ANALYSIS\n")
                f.write("========================================\n\n")
                f.write(f"Analysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"Date range: {start_date} to {end_date}\n\n")
                
                f.write("Validation Criteria:\n")
                f.write("-------------------\n")
                f.write(f"Distance threshold: {config.get('distance_threshold', 50)} miles\n")
                f.write(f"Minimum completion time: {config.get('min_completion_time', 60)} seconds\n")
                f.write(f"Captcha threshold: {config.get('captcha_threshold', 0.5)}\n")
                f.write(f"Validation failure threshold: {config.get('validation_failure_threshold', 3)}\n")
                f.write(f"Heavily shared IP threshold: {config.get('heavily_shared_ip_threshold', 10)}\n")
                f.write(f"Maximum whitelist distance: {config.get('max_whitelist_distance', 400)} miles\n")
                f.write(f"IP whitelist entries: {len(config.get('ip_whitelist', []))}\n\n")
                
                f.write("Summary Statistics:\n")
                f.write("-------------------\n")
                f.write(f"Total incentive survey responses: {detailed_statistics.get('total_incentive', 0)}\n")
                f.write(f"Total community survey responses: {detailed_statistics.get('total_community', 0)}\n")
                f.write(f"Respondents who only completed incentive survey: {detailed_statistics.get('incentive_only', 0)}\n")
                
                f.write("\nMatching Statistics:\n")
                f.write("-------------------\n")
                f.write(f"Total IP matches found: {detailed_statistics.get('total_ip_matches', 0)}\n")
                f.write(f"Unique community surveys involved in matches: {detailed_statistics.get('unique_community_matched', 0)}\n")
                
                f.write("\nEligibility Results:\n")
                f.write("-------------------\n")
                f.write(f"Eligible respondents for incentives: {detailed_statistics.get('eligible', 0)}\n")
                f.write(f"Ineligible matched respondents: {detailed_statistics.get('ineligible', 0)}\n")
                
                if 'whitelist_matches' in detailed_statistics:
                    f.write("\nWhitelist Analysis:\n")
                    f.write("------------------\n")
                    f.write(f"Total respondents with whitelisted IPs: {detailed_statistics.get('whitelist_matches', 0)}\n")
                    f.write(f"Eligible respondents with whitelisted IPs: {detailed_statistics.get('whitelist_eligible', 0)}\n")
            
            # Redirect to results page with Excel file info
            return redirect(url_for('main.results', report=report_filename, excel=excel_filename if eligible_respondents else None))
            
        except Exception as e:
            flash(f'Error processing surveys: {str(e)}')
            return redirect(request.url)
    
    return render_template('validate.html')

@main.route('/results')
@login_required
def results():
    report = request.args.get('report')
    if not report:
        return redirect(url_for('main.dashboard'))
    
    report_path = os.path.join(current_app.config['RESULTS_FOLDER'], report)
    if not os.path.exists(report_path):
        flash('Report not found')
        return redirect(url_for('main.dashboard'))
    
    # Read report content
    with open(report_path, 'r') as f:
        report_content = f.read()
    
    # Get related files
    base_name = report.replace('validation_report_', '').replace('.txt', '')
    eligible_file = f"eligible_respondents_{base_name}.csv"
    ineligible_file = f"ineligible_respondents_{base_name}.csv"
    excel_file = f"survey_results_{base_name}.xlsx"
    
    eligible_path = os.path.join(current_app.config['RESULTS_FOLDER'], eligible_file)
    ineligible_path = os.path.join(current_app.config['RESULTS_FOLDER'], ineligible_file)
    excel_path = os.path.join(current_app.config['RESULTS_FOLDER'], excel_file)
    
    files = {
        'report': {
            'name': report,
            'exists': True,
            'path': url_for('main.download_file', filename=report)
        },
        'eligible': {
            'name': eligible_file,
            'exists': os.path.exists(eligible_path),
            'path': url_for('main.download_file', filename=eligible_file) if os.path.exists(eligible_path) else None
        },
        'ineligible': {
            'name': ineligible_file,
            'exists': os.path.exists(ineligible_path),
            'path': url_for('main.download_file', filename=ineligible_file) if os.path.exists(ineligible_path) else None
        },
        'excel': {
            'name': excel_file,
            'exists': os.path.exists(excel_path),
            'path': url_for('main.download_file', filename=excel_file) if os.path.exists(excel_path) else None
        }
    }
    
    return render_template('results.html', report_content=report_content, files=files)

@main.route('/download/<filename>')
@login_required
def download_file(filename):
    return send_from_directory(current_app.config['RESULTS_FOLDER'], filename, as_attachment=True)

@main.route('/match', methods=['GET', 'POST'])
def match():
    form = MatchForm()
    if form.validate_on_submit():
        try:
            # Get uploaded files
            eligibility_file = form.eligibility_file.data
            survey_file = form.survey_file.data
            
            # Debug print: print the sheet name received from the form
            print(f"Sheet name received from form: '{form.sheet_name.data}'", flush=True)

            # Save uploaded file to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                eligibility_file.save(tmp)
                tmp_path = tmp.name

            print("Temp file path:", tmp_path, flush=True)

            # Now use the temp file path for pandas
            xls = pd.ExcelFile(tmp_path)
            print("Sheet names found in uploaded file:", xls.sheet_names, flush=True)

            eligibility_df = pd.read_excel(
                tmp_path,
                sheet_name=form.sheet_name.data,
                header=0  # Use the first row as header
            )
            print("Eligibility DataFrame columns:", eligibility_df.columns.tolist(), flush=True)
            if 'IPAddress' in eligibility_df.columns:
                print("First 10 unique eligibility IPs:", eligibility_df['IPAddress'].dropna().unique()[:10], flush=True)
            if 'timestamp' in eligibility_df.columns:
                print("First 5 eligibility timestamps:", eligibility_df['timestamp'].dropna().head(5).tolist(), flush=True)
            print("First 5 rows of eligibility_df:\n", eligibility_df.head(5), flush=True)
            
            # Read survey data from CSV
            try:
                # Save uploaded survey file to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_csv:
                    survey_file.save(tmp_csv)
                    tmp_csv_path = tmp_csv.name

                # Clean up the CSV before reading
                cleaned_csv_path = tmp_csv_path + "_cleaned.csv"
                with open(tmp_csv_path, 'r', encoding='utf-8', errors='ignore') as infile, open(cleaned_csv_path, 'w', encoding='utf-8') as outfile:
                    for line in infile:
                        # Remove problematic characters or fix quoting here if needed
                        outfile.write(line.replace('"', '\"'))

                # Now read the cleaned file
                survey_df = pd.read_csv(cleaned_csv_path, header=0, skiprows=[1,2])  # Use first row as header, skip next two

                # Debug: Print first 10 unique IPs and first 5 timestamps from survey_df
                if 'IPAddress' in survey_df.columns:
                    print("First 10 unique survey IPs:", survey_df['IPAddress'].dropna().unique()[:10], flush=True)
                if 'timestamp' in survey_df.columns:
                    print("First 5 survey timestamps:", survey_df['timestamp'].dropna().head(5).tolist(), flush=True)
                print("First 5 rows of survey_df:\n", survey_df.head(5), flush=True)
            except Exception as e:
                print("Error reading CSV:", e)
                flash('Error reading survey file', 'error')
                return render_template('match.html', form=form)
            
            # Initialize results dictionary without email-related fields
            results = {
                'matched_count': 0,
                'total_count': len(eligibility_df),
                'match_percentage': 0,
                'matched_data': [],
                'unmatched_ips': []  # Track unmatched IPs instead of emails
            }
            
            # Define the fields to include from both eligibility and survey DataFrames
            eligibility_fields = [
                'ResponseId', 'RecipientEmail', 'RecipientFirstName', 'RecipientLastName', 'SubmissionDate', 'IPAddress'
            ]
            survey_fields = [
                'ResponseId', 'StartDate', 'Q_RecaptchaScore', 'Q_RelevantIDDuplicate', 'Q_RelevantIDDuplicateScore', 'Q_RelevantIDFraudScore'
            ]
            # Add all Q* columns from survey_df
            survey_fields += [col for col in survey_df.columns if col.startswith('Q')]

            # Update matching logic to build a full row for each match
            results['matched_data'] = []
            for _, eligible_row in eligibility_df.iterrows():
                ip_matches = survey_df[survey_df['IPAddress'] == eligible_row['IPAddress']]
                if ip_matches.empty:
                    results['unmatched_ips'].append(eligible_row['IPAddress'])
                    continue
                if len(ip_matches) == 1:
                    match_row = ip_matches.iloc[0]
                else:
                    # Use time window if multiple matches
                    if 'timestamp' in eligible_row and 'timestamp' in ip_matches.columns:
                        time_diff = abs(ip_matches['timestamp'] - eligible_row['timestamp'])
                        time_matches = ip_matches[time_diff <= time_window]
                        if not time_matches.empty:
                            closest_idx = time_diff[time_diff <= time_window].idxmin()
                            match_row = ip_matches.loc[closest_idx]
                        else:
                            results['unmatched_ips'].append(eligible_row['IPAddress'])
                            continue
                    else:
                        match_row = ip_matches.iloc[0]
                # Build the combined row
                combined_row = {}
                for field in eligibility_fields:
                    combined_row[f'Eligibility_{field}'] = eligible_row.get(field, '')
                for field in survey_fields:
                    combined_row[f'Survey_{field}'] = match_row.get(field, '')
                results['matched_data'].append(combined_row)
                results['matched_count'] += 1
            
            # Save matched data to a temporary CSV file instead of session
            tmp_csv_path = os.path.join(tempfile.gettempdir(), f"matched_data_{uuid.uuid4().hex}.csv")
            fieldnames = list(results['matched_data'][0].keys()) if results['matched_data'] else eligibility_fields + survey_fields
            # Replace nan with 'N/A' in matched_data
            cleaned_matched_data = []
            for row in results['matched_data']:
                cleaned_row = {k: ('N/A' if (v is None or (isinstance(v, float) and np.isnan(v)) or str(v) == 'nan') else v) for k, v in row.items()}
                cleaned_matched_data.append(cleaned_row)
            with open(tmp_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(cleaned_matched_data)
            session['matched_data_file'] = tmp_csv_path
            session['csv_fieldnames'] = fieldnames
            print(f"DEBUG: matched data written to {tmp_csv_path}", flush=True)
            
            # Calculate match percentage
            results['match_percentage'] = (results['matched_count'] / results['total_count']) * 100
            
            # Calculate race/ethnicity summary from matched data
            race_col = 'Survey_Q28'  # Adjust if your column name is different
            race_counts = collections.Counter(row.get(race_col, 'N/A') for row in cleaned_matched_data)
            total = sum(race_counts.values())
            race_summary = [
                {
                    'Race/Ethnicity': race,
                    'Count': count,
                    'Percent': round((count / total) * 100, 1) if total > 0 else 0
                }
                for race, count in race_counts.items()
            ]
            race_summary = sorted(race_summary, key=lambda x: -x['Count'])
            
            return render_template('match.html', form=form, results=results, race_summary=race_summary)
            
        except Exception as e:
            print("Exception occurred in /match route:", flush=True)
            traceback.print_exc()
            flash(f'Error processing files: {str(e)}', 'error')
            return render_template('match.html', form=form)
    
    return render_template('match.html', form=form)

@main.route('/download_matched_data')
def download_matched_data():
    tmp_csv_path = session.get('matched_data_file')
    if not tmp_csv_path or not os.path.exists(tmp_csv_path):
        flash('No matched data available for download', 'error')
        return redirect(url_for('main.match'))
    return send_file(tmp_csv_path, mimetype='text/csv', as_attachment=True, download_name='matched_data.csv')