import os
import pandas as pd
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_from_directory, current_app, session, Response
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import sys
import json
import xlsxwriter
from io import BytesIO, StringIO
import csv

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
            
            # Read eligibility data from Excel
            eligibility_df = pd.read_excel(
                eligibility_file,
                sheet_name=form.sheet_name.data,
                skiprows=1 if form.skip_header.data else 0
            )
            
            # Read survey data from CSV
            survey_df = pd.read_csv(survey_file, sep='\t', engine='python')
            
            # Initialize results dictionary
            results = {
                'matched_count': 0,
                'total_count': len(eligibility_df),
                'match_percentage': 0,
                'matched_data': [],
                'unmatched_emails': []
            }
            
            # Perform matching based on selected method
            if form.matching_method.data == 'ip_time':
                # Convert timestamps to datetime if needed
                if 'SubmissionDate' in eligibility_df.columns:
                    eligibility_df['timestamp'] = pd.to_datetime(eligibility_df['SubmissionDate'])
                if 'Start Date' in survey_df.columns:
                    survey_df['timestamp'] = pd.to_datetime(survey_df['Start Date'])
                
                # Match based on IP and time window
                time_window = pd.Timedelta(minutes=form.time_window.data)
                for _, eligible_row in eligibility_df.iterrows():
                    ip_matches = survey_df[survey_df['IP Address'] == eligible_row['IPAddress']]
                    if not ip_matches.empty:
                        time_diff = abs(ip_matches['timestamp'] - eligible_row['timestamp'])
                        time_matches = ip_matches[time_diff <= time_window]
                        if not time_matches.empty:
                            results['matched_data'].append({
                                'eligibility_email': eligible_row['RecipientEmail'],
                                'eligibility_ip': eligible_row['IPAddress'],
                                'survey_response_id': time_matches.iloc[0]['Response ID'],
                                'match_type': 'IP + Time'
                            })
                            results['matched_count'] += 1
                        else:
                            results['unmatched_emails'].append(eligible_row['RecipientEmail'])
                    else:
                        results['unmatched_emails'].append(eligible_row['RecipientEmail'])
            
            elif form.matching_method.data == 'ip_only':
                # Match based on IP only
                for _, eligible_row in eligibility_df.iterrows():
                    ip_matches = survey_df[survey_df['IP Address'] == eligible_row['IPAddress']]
                    if not ip_matches.empty:
                        results['matched_data'].append({
                            'eligibility_email': eligible_row['RecipientEmail'],
                            'eligibility_ip': eligible_row['IPAddress'],
                            'survey_response_id': ip_matches.iloc[0]['Response ID'],
                            'match_type': 'IP Only'
                        })
                        results['matched_count'] += 1
                    else:
                        results['unmatched_emails'].append(eligible_row['RecipientEmail'])
            
            elif form.matching_method.data == 'email':
                # Match based on email
                for _, eligible_row in eligibility_df.iterrows():
                    email_matches = survey_df[survey_df['RecipientEmail'].str.lower() == eligible_row['RecipientEmail'].str.lower()]
                    if not email_matches.empty:
                        results['matched_data'].append({
                            'eligibility_email': eligible_row['RecipientEmail'],
                            'eligibility_ip': eligible_row['IPAddress'],
                            'survey_response_id': email_matches.iloc[0]['Response ID'],
                            'match_type': 'Email'
                        })
                        results['matched_count'] += 1
                    else:
                        results['unmatched_emails'].append(eligible_row['RecipientEmail'])
            
            elif form.matching_method.data == 'response_id':
                # Match based on response ID
                for _, eligible_row in eligibility_df.iterrows():
                    id_matches = survey_df[survey_df['Response ID'] == eligible_row['ResponseId']]
                    if not id_matches.empty:
                        results['matched_data'].append({
                            'eligibility_email': eligible_row['RecipientEmail'],
                            'eligibility_ip': eligible_row['IPAddress'],
                            'survey_response_id': id_matches.iloc[0]['Response ID'],
                            'match_type': 'Response ID'
                        })
                        results['matched_count'] += 1
                    else:
                        results['unmatched_emails'].append(eligible_row['RecipientEmail'])
            
            # Calculate match percentage
            results['match_percentage'] = (results['matched_count'] / results['total_count']) * 100
            
            # Store results in session for download
            session['matched_data'] = results['matched_data']
            
            return render_template('match.html', form=form, results=results)
            
        except Exception as e:
            flash(f'Error processing files: {str(e)}', 'error')
            return render_template('match.html', form=form)
    
    return render_template('match.html', form=form)

@main.route('/download_matched_data')
def download_matched_data():
    if 'matched_data' not in session:
        flash('No matched data available for download', 'error')
        return redirect(url_for('main.match'))
    
    # Create CSV file
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=['eligibility_email', 'eligibility_ip', 'survey_response_id', 'match_type'])
    writer.writeheader()
    writer.writerows(session['matched_data'])
    
    # Prepare response
    output.seek(0)
    return Response(
        output,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=matched_data.csv'}
    )