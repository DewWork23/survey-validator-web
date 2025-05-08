from flask import Blueprint, render_template, request, flash, redirect, url_for, session
import pandas as pd
import plotly.express as px
import plotly.utils
import json
import os
import tempfile
from werkzeug.utils import secure_filename

bp = Blueprint('demographics', __name__)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/demographics', methods=['GET', 'POST'])
def demographics():
    if request.method == 'POST':
        # Step 1: File upload
        if 'surveyData' in request.files and request.files['surveyData'].filename != '':
            survey_file = request.files['surveyData']
            if not allowed_file(survey_file.filename):
                flash('Only CSV files are allowed')
                return redirect(request.url)
            try:
                df = pd.read_csv(survey_file, sep=',', on_bad_lines='skip', engine='python', skiprows=[1,2])
                # Save to a temp file and store path in session
                temp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                df.to_csv(temp.name, index=False)
                session['demographics_tempfile'] = temp.name
                temp.close()
                # Show dropdown for column selection
                return render_template('demographics.html', columns=df.columns, show_column_select=True)
            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)
        # Step 2: Column selection
        elif 'race_column' in request.form and 'demographics_tempfile' in session:
            race_column = request.form['race_column']
            temp_path = session.get('demographics_tempfile')
            try:
                df = pd.read_csv(temp_path)
                # Clean up temp file
                os.remove(temp_path)
                session.pop('demographics_tempfile', None)
                # Remove metadata values from the selected column
                metadata_values = ['{"ImportId":"QID28"}', 'Which of the following best describes your racial background?']
                df = df[~df[race_column].isin(metadata_values)]
                # Group and count by selected column
                demo_counts = df[race_column].value_counts(dropna=False).reset_index()
                demo_counts.columns = ['race_ethnicity', 'count']
                # Calculate percentages
                total = demo_counts['count'].sum()
                demo_counts['percent'] = (demo_counts['count'] / total * 100).round(1)
                # Create chart
                demographics_fig = px.pie(demo_counts, values='count', names='race_ethnicity',
                                         title='Survey Participants by Race/Ethnicity')
                demographics_json = json.dumps(demographics_fig, cls=plotly.utils.PlotlyJSONEncoder)
                # Pass summary to template
                demo_summary = demo_counts.to_dict(orient='records')
                return render_template('demographics.html', show_chart=True, demographics_chart=demographics_json, demo_summary=demo_summary)
            except Exception as e:
                flash(f'Error processing data: {str(e)}')
                return redirect(request.url)
        else:
            flash('Please upload a file')
            return redirect(request.url)
    return render_template('demographics.html') 