from flask_wtf import FlaskForm
from wtforms import StringField, FileField, SelectField, IntegerField, BooleanField
from wtforms.validators import DataRequired, Optional, NumberRange

class MatchForm(FlaskForm):
    eligibility_file = FileField('Eligibility Spreadsheet (XLSX)', validators=[DataRequired()])
    survey_file = FileField('Survey Data (CSV)', validators=[DataRequired()])
    sheet_name = StringField('Excel Sheet Name', validators=[DataRequired()])
    matching_method = SelectField('Matching Method', 
                                choices=[
                                    ('ip_time', 'IP Address + Time Window'),
                                    ('ip_only', 'IP Address Only'),
                                    ('email', 'Email Match'),
                                    ('response_id', 'Response ID Match')
                                ],
                                validators=[DataRequired()])
    time_window = IntegerField('Time Window (minutes)', 
                              default=10,
                              validators=[Optional(), NumberRange(min=1, max=1440)])
    skip_header = BooleanField('Skip first row (metadata)', default=True) 