#!/usr/bin/env python3
"""
Improved Survey Validation Script with IP Whitelist Support and Enhanced Reporting

This script analyzes community and incentive survey responses to determine
eligibility for incentives, with support for IP whitelisting to validate
respondents from mobile carriers or ISPs while still enforcing reasonable
geographic limits.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import os
import sys
import argparse
import logging
import json
import re
import ipaddress
import tempfile
import csv
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('survey_validator')

# Default configuration
DEFAULT_CONFIG = {
    "distance_threshold": 50,  # miles
    "min_completion_time": 60,  # seconds
    "captcha_threshold": 0.5,
    "heavily_shared_ip_threshold": 10,
    "validation_failure_threshold": 3,
    "max_whitelist_distance": 400,  # Maximum distance in miles for whitelisted IPs
    "ip_whitelist": []  # Default empty whitelist
}

# Geographic constants
ROBESON_CENTER = (34.6390, -79.1003)  # Robeson County center coordinates
DEGREES_TO_MILES = 69  # 1 degree ≈ 69 miles

bad_lines = []

def log_bad_line(bad_line):
    bad_lines.append(bad_line)
    print(f"Skipped bad line: {bad_line}")

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Validate survey responses for incentive eligibility')
    
    # File arguments
    parser.add_argument('--community', type=str, default="communitysurvey.csv",
                        help='Path to community survey CSV file (default: communitysurvey.csv)')
    parser.add_argument('--incentive', type=str, default="incentivesurvey.csv",
                        help='Path to incentive survey CSV file (default: incentivesurvey.csv)')
    
    # Date range arguments
    parser.add_argument('--start-date', type=str, 
                        help='Start date for filtering (YYYY-MM-DD, default: 7 days ago)')
    parser.add_argument('--end-date', type=str,
                        help='End date for filtering (YYYY-MM-DD, default: today)')
    
    # Output directory
    parser.add_argument('--output-dir', type=str, default=".",
                        help='Directory for output files (default: current directory)')
    
    # Configuration file
    parser.add_argument('--config', type=str,
                        help='Path to JSON configuration file')
    
    # Verbose mode
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default dates if not provided
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    if not args.end_date:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    return args


def load_config(config_path=None):
    """Load configuration from file or use defaults"""
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                config.update(user_config)
                logger.info(f"Loaded configuration from {config_path}")
                
                # Validate and log whitelist configuration
                if 'ip_whitelist' in config and config['ip_whitelist']:
                    logger.info(f"IP whitelist loaded with {len(config['ip_whitelist'])} entries")
                    
                    # Validate each whitelist entry
                    valid_entries = []
                    for i, entry in enumerate(config['ip_whitelist']):
                        try:
                            # Check if it's a CIDR notation
                            if '/' in entry:
                                ipaddress.ip_network(entry, strict=False)
                                valid_entries.append(entry)
                            else:
                                # Check if it's a single IP
                                ipaddress.ip_address(entry)
                                valid_entries.append(entry)
                        except ValueError as ve:
                            logger.warning(f"Skipping invalid whitelist entry #{i+1} '{entry}': {str(ve)}")
                    
                    # Update whitelist with only valid entries
                    config['ip_whitelist'] = valid_entries
                    logger.info(f"IP whitelist validated: {len(valid_entries)} valid entries")
                    
                    # Log first few entries as a sample
                    if valid_entries:
                        sample = valid_entries[:3]
                        logger.info(f"Sample whitelist entries: {', '.join(sample)}")
                        if len(valid_entries) > 3:
                            logger.info(f"... and {len(valid_entries) - 3} more entries")
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            logger.info("Using default configuration")
    else:
        logger.info("Using default configuration")
    
    return config


def find_survey_files(directory="."):
    """Find survey files in the directory based on naming patterns"""
    files = os.listdir(directory)
    
    community_patterns = [
        r'community.*\.csv',
        r'.*community.*\.csv',
        r'.*survey.*community.*\.csv'
    ]
    
    incentive_patterns = [
        r'incentive.*\.csv',
        r'.*incentive.*\.csv',
        r'.*survey.*incentive.*\.csv'
    ]
    
    community_files = []
    incentive_files = []
    
    for file in files:
        if any(re.match(pattern, file.lower()) for pattern in community_patterns):
            community_files.append(file)
        if any(re.match(pattern, file.lower()) for pattern in incentive_patterns):
            incentive_files.append(file)
    
    return community_files, incentive_files


def is_ip_whitelisted(ip, whitelist, distance=None, max_distance=400):
    """
    Check if an IP address is in the whitelist, with a maximum distance check.
    
    Args:
        ip: IP address to check
        whitelist: List of IP addresses or CIDR ranges
        distance: Optional distance in miles (if provided, used to enforce max_distance)
        max_distance: Maximum allowed distance in miles for whitelisting
        
    Returns:
        bool: True if IP is whitelisted and within distance limits, False otherwise
    """
    if not ip or not whitelist:
        return False
    
    # If distance is provided and exceeds max_distance, don't whitelist
    if distance is not None and pd.notna(distance):
        try:
            # Convert to float and compare with max_distance
            distance_float = float(distance)
            if distance_float > max_distance:
                logger.debug(f"IP {ip} not whitelisted due to excessive distance ({distance_float} miles > {max_distance} miles)")
                return False
        except (ValueError, TypeError) as e:
            logger.debug(f"Error converting distance '{distance}' to float: {str(e)}")
            return False
    
    # First normalize the IP address by trimming whitespace
    ip = ip.strip() if isinstance(ip, str) else ip
    
    try:
        # Convert the IP to an IPv4Address object
        ip_obj = ipaddress.ip_address(ip)
        
        # Check if IP is in any of the whitelist entries
        for entry in whitelist:
            try:
                # Normalize the whitelist entry
                entry = entry.strip() if isinstance(entry, str) else entry
                
                # Try to parse as a network (CIDR)
                if isinstance(entry, str) and '/' in entry:
                    network = ipaddress.ip_network(entry, strict=False)
                    if ip_obj in network:
                        logger.debug(f"IP {ip} is in whitelist network {entry}")
                        return True
                # Try as individual IP
                elif str(ipaddress.ip_address(entry)) == str(ip_obj):
                    logger.debug(f"IP {ip} matches whitelist IP {entry}")
                    return True
            except Exception as e:
                # Log invalid entries but continue checking other entries
                logger.debug(f"Error checking IP '{ip}' against whitelist entry '{entry}': {str(e)}")
                continue
    except Exception as e:
        # Log invalid IP format but don't raise an exception
        logger.debug(f"Error checking IP '{ip}' against whitelist: {str(e)}")
        return False
    
    return False


def calculate_distance(lat, lon, center=ROBESON_CENTER):
    """Calculate distance in miles from a point to the center coordinates"""
    if pd.isna(lat) or pd.isna(lon):
        return np.nan
    
    # Debug logging to trace issues
    logger.debug(f"Calculating distance for point ({lat}, {lon}) to center {center}")
    
    # Print original data types
    logger.debug(f"Type of lat: {type(lat)}, type of lon: {type(lon)}")
    
    # Ensure coordinates are numeric
    try:
        lat = float(lat)
        lon = float(lon)
    except (ValueError, TypeError) as e:
        logger.warning(f"Error converting coordinates to float: {e}")
        return np.nan
    
    # Verify coordinates are in reasonable ranges
    if abs(lat) > 90 or abs(lon) > 180:
        logger.warning(f"Invalid coordinates detected: ({lat}, {lon}) - out of valid range")
        return np.nan
    
    # Check for possible swapped coordinates
    if abs(lat) > 80 or abs(lon) < 10:
        logger.warning(f"Suspicious coordinates detected: ({lat}, {lon}) - may be swapped")
        # Don't automatically swap them, just flag for review
    
    # Calculate Euclidean distance and convert to approximate miles
    distance = np.sqrt(
        (lat - center[0])**2 + 
        (lon - center[1])**2
    ) * DEGREES_TO_MILES
    
    # Log the result
    logger.debug(f"Distance calculated: {distance} miles")
    
    # If distance is very large (over 1000 miles), log it
    if distance > 1000:
        logger.warning(f"Extremely large distance detected ({distance} miles) for coordinates ({lat}, {lon})")
    
    return distance


def safely_extract_value(row, keys, default=''):
    """Safely extract a value from a row using multiple possible keys"""
    for key in keys:
        if key in row and pd.notna(row[key]):
            try:
                return str(row[key])
            except:
                pass
    return default


def extract_respondent_data(incentive_row, community_row=None, ip_address=None):
    """
    Extract and format respondent data from survey rows.
    
    Args:
        incentive_row: Row from incentive survey
        community_row: Row from community survey (optional)
        ip_address: IP address string (optional)
        
    Returns:
        dict: Dictionary with extracted respondent data
    """
    # Create a basic dictionary with default values
    respondent_data = {
        'ResponseId': '',
        'IPAddress': '',
        'SubmissionDate': '',
        'RecipientEmail': '',
        'RecipientFirstName': '',
        'RecipientLastName': '',
        'Distance': 'Unknown',
        'MatchMethod': 'IPAddress',
        'SharedIP': 'No',
        'IPWhitelisted': 'No'
    }
    
    # Extract values safely
    respondent_data['ResponseId'] = safely_extract_value(incentive_row, ['ResponseId'])
    respondent_data['IPAddress'] = safely_extract_value(incentive_row, ['IPAddress'])
    respondent_data['SubmissionDate'] = safely_extract_value(incentive_row, ['RecordedDate'])
    
    # Extract email (check both standard field and survey question field)
    email = safely_extract_value(incentive_row, ['RecipientEmail', 'Q3'])
    if '@' in email and '.' in email:  # Basic email validation
        respondent_data['RecipientEmail'] = email
    
    # Extract names
    respondent_data['RecipientFirstName'] = safely_extract_value(incentive_row, ['RecipientFirstName', 'Q1'])
    respondent_data['RecipientLastName'] = safely_extract_value(incentive_row, ['RecipientLastName', 'Q2'])
    
    # Add distance information if available
    if community_row is not None and 'distance_miles' in community_row and pd.notna(community_row['distance_miles']):
        respondent_data['Distance'] = f"{float(community_row['distance_miles']):.1f} miles"
    elif 'distance_miles' in incentive_row and pd.notna(incentive_row['distance_miles']):
        respondent_data['Distance'] = f"{float(incentive_row['distance_miles']):.1f} miles"
    
    # Check if IP is shared with other respondents
    if pd.notna(ip_address):
        respondent_data['SharedIP'] = 'Yes' if pd.notna(ip_address) else 'No'
    
    return respondent_data


def read_and_prepare_dataframe(file_path, date_range=None):
    """
    Read CSV file and prepare DataFrame with common preprocessing steps
    
    Args:
        file_path: Path to CSV file
        date_range: Tuple of (start_date, end_date) datetime objects
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame
    """
    try:
        # First try to detect the file structure
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_lines = [next(f) for _ in range(5)]
        
        # Determine if we need to skip rows
        skip_rows = []
        for i, line in enumerate(first_lines):
            if any(header in line.lower() for header in ['responseid', 'response id', 'recordeddate', 'recorded date']):
                skip_rows.append(i)
        
        logger.info(f"Detected header structure, skipping rows: {skip_rows}")
        
        # Read the file with robust pandas options
        df = pd.read_csv(
            file_path,
            sep=',',
            on_bad_lines='warn',  # Only use on_bad_lines for pandas >=1.3
            engine='python',
            skiprows=skip_rows,
            encoding='utf-8'
        )
        
        logger.info(f"Successfully read {file_path} with {len(df)} rows and {len(df.columns)} columns")
        
        # Filter out header rows if they exist
        if 'ResponseId' in df.columns:
            df = df[~((df['ResponseId'] == '{\"ImportId\":\"_recordId\"}') | 
                      (df['ResponseId'] == 'Response ID'))]
        
        # Convert date columns to datetime
        if 'RecordedDate' in df.columns:
            df['RecordedDate'] = pd.to_datetime(df['RecordedDate'], errors='coerce')
            logger.info(f"Date range: {df['RecordedDate'].min()} to {df['RecordedDate'].max()}")
        
        # Convert numeric columns
        for col in ['LocationLatitude', 'LocationLongitude', 'Duration (in seconds)', 'Q_RecaptchaScore']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter by date range if provided
        if date_range and 'RecordedDate' in df.columns:
            start_date, end_date = date_range
            df_filtered = df[
                (df['RecordedDate'] >= start_date) & 
                (df['RecordedDate'] <= end_date)
            ]
            logger.info(f"Filtered rows within date range: {len(df_filtered)}")
            return df_filtered
        
        return df
        
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise ValidationError(f"Error reading file {file_path}: {e}")


def filter_completed_surveys(df, survey_type):
    """
    Filter DataFrames to only include completed surveys
    
    Args:
        df: DataFrame to filter
        survey_type: String identifying the survey type ('community' or 'incentive')
        
    Returns:
        pandas.DataFrame: Filtered DataFrame
    """
    # For community surveys, use 'Finished' column when available
    if survey_type == 'community' and 'Finished' in df.columns:
        logger.info(f"Using 'Finished' column for {survey_type} survey")
        finished_values = df['Finished'].value_counts()
        logger.info(f"{survey_type.capitalize()} 'Finished' column values: {finished_values.to_dict()}")
        
        completed = df[df['Finished'].isin([True, 1, '1', 'True'])]
    else:
        # For incentive surveys or when 'Finished' column is not available, use all rows
        logger.info(f"No 'Finished' column filter applied for {survey_type} survey - using all rows")
        completed = df
    
    logger.info(f"Completed {survey_type} survey rows: {len(completed)}")
    return completed


def ensure_ip_address_column(df, survey_type):
    """
    Ensure DataFrame has an IPAddress column, find alternative if not
    
    Args:
        df: DataFrame to check
        survey_type: String identifying the survey type
        
    Returns:
        pandas.DataFrame: DataFrame with IPAddress column
    """
    df_copy = df.copy()
    
    if 'IPAddress' not in df_copy.columns:
        logger.warning(f"No 'IPAddress' column in {survey_type} survey")
        ip_cols = [col for col in df_copy.columns if 'ip' in col.lower()]
        
        if ip_cols:
            logger.info(f"Found alternative IP column: {ip_cols[0]}")
            df_copy['IPAddress'] = df_copy[ip_cols[0]]
        else:
            error_msg = f"No IP address column found in {survey_type} survey"
            logger.error(error_msg)
            raise ValidationError(error_msg)
    
    # Count non-null IPs
    ip_count = df_copy['IPAddress'].notna().sum()
    logger.info(f"{survey_type.capitalize()} surveys with IP addresses: {ip_count} of {len(df_copy)}")
    
    return df_copy


def add_distance_calculations(df, survey_type):
    """
    Add distance calculations to the DataFrame
    
    Args:
        df: DataFrame to add distances to
        survey_type: String identifying the survey type
        
    Returns:
        pandas.DataFrame: DataFrame with distance_miles column
    """
    df_copy = df.copy()
    df_copy['distance_miles'] = np.nan
    
    if 'LocationLatitude' in df_copy.columns and 'LocationLongitude' in df_copy.columns:
        # Apply distance calculation to rows with valid location data
        valid_locations = df_copy[df_copy['LocationLatitude'].notna() & df_copy['LocationLongitude'].notna()]
        
        for idx in valid_locations.index:
            try:
                lat = df_copy.loc[idx, 'LocationLatitude']
                lon = df_copy.loc[idx, 'LocationLongitude']
                df_copy.loc[idx, 'distance_miles'] = calculate_distance(lat, lon)
            except Exception as e:
                logger.error(f"Error calculating {survey_type} distance for row {idx}: {e}")
    else:
        logger.warning(f"Missing location columns in {survey_type} survey")
    
    return df_copy


def match_surveys(community_df, incentive_df):
    """
    Match surveys across community and incentive DataFrames
    
    Args:
        community_df: Community survey DataFrame
        incentive_df: Incentive survey DataFrame
        
    Returns:
        tuple: (matched_incentive, matched_community, matching_details, statistics)
    """
    # Create a mapping of IP addresses to community survey responses
    ip_to_community = defaultdict(list)
    for idx, row in community_df.iterrows():
        if pd.notna(row['IPAddress']):
            ip_to_community[row['IPAddress']].append(idx)
    
    # Track matched rows and statistics
    matched_incentive_rows = []
    matched_community_rows = set()  # Using a set to avoid duplicates
    matching_details = []
    
    # Count matches by method
    ip_matches = 0
    ip_unique_matches = 0
    ip_multiple_matches = 0
    
    # Track which IPs are already matched to avoid double counting
    ip_matched_count = defaultdict(int)
    
    # Match by IP address
    for idx, incentive_row in incentive_df.iterrows():
        ip_address = incentive_row['IPAddress']
        
        if pd.notna(ip_address) and ip_address in ip_to_community:
            # Get all community rows with this IP
            community_matches = ip_to_community[ip_address]
            
            if community_matches:
                matched_incentive_rows.append(idx)
                
                # Count how many community surveys match this incentive
                match_count = len(community_matches)
                
                # If multiple matches, use the first one only for consistency
                community_idx = community_matches[0]
                matched_community_rows.add(community_idx)
                
                # Count match types
                ip_matches += 1
                ip_matched_count[ip_address] += 1
                
                # Track if this is a unique or multiple match
                if match_count == 1:
                    ip_unique_matches += 1
                else:
                    ip_multiple_matches += 1
                
                matching_details.append({
                    'IncentiveResponseId': str(incentive_row.get('ResponseId', '')),
                    'IncentiveIP': ip_address,
                    'CommunityResponseId': str(community_df.loc[community_idx].get('ResponseId', '')),
                    'CommunityIP': str(community_df.loc[community_idx]['IPAddress']),
                    'MatchMethod': 'IPAddress',
                    'MultipleMatches': match_count > 1
                })
    
    # Create dataframes of matched respondents
    matched_incentive = incentive_df.loc[matched_incentive_rows].copy() if matched_incentive_rows else pd.DataFrame()
    matched_community = community_df.loc[list(matched_community_rows)].copy() if matched_community_rows else pd.DataFrame()
    
    # Count unique community surveys that match to incentive surveys
    unique_community_matched = len(matched_community_rows)
    
    # Calculate statistics
    match_statistics = {
        'total_ip_matches': ip_matches,
        'unique_community_matched': unique_community_matched,
        'ip_unique_matches': ip_unique_matches,
        'ip_multiple_matches': ip_multiple_matches
    }
    
    return matched_incentive, matched_community, matching_details, match_statistics

def evaluate_eligibility(matched_incentive, matched_community, matching_details, config):
    """
    Evaluate eligibility of matched surveys
    
    Args:
        matched_incentive: DataFrame of matched incentive surveys
        matched_community: DataFrame of matched community surveys
        matching_details: List of dictionaries with match information
        config: Configuration dictionary
        
    Returns:
        tuple: (eligible_respondents, ineligible_respondents, statistics)
    """
    # Extract configuration parameters
    distance_threshold = config.get('distance_threshold', 50)
    min_completion_time = config.get('min_completion_time', 60)
    captcha_threshold = config.get('captcha_threshold', 0.5)
    heavily_shared_ip_threshold = config.get('heavily_shared_ip_threshold', 10)
    validation_failure_threshold = config.get('validation_failure_threshold', 3)
    max_whitelist_distance = config.get('max_whitelist_distance', 400)
    ip_whitelist = config.get('ip_whitelist', [])
    
    # Pre-calculate IP and email frequencies
    ip_frequencies = matched_incentive['IPAddress'].value_counts().to_dict()
    email_frequencies = {}
    
    if 'RecipientEmail' in matched_incentive.columns:
        email_series = matched_incentive['RecipientEmail'].dropna()
        email_series = email_series.apply(lambda x: str(x).lower().strip())
        email_frequencies = email_series.value_counts().to_dict()
    
    # Build matching map for easier lookup
    matching_map = {}
    for match in matching_details:
        incentive_id = match['IncentiveResponseId']
        community_id = match['CommunityResponseId']
        ip_address = match['CommunityIP']
        matching_map[incentive_id] = {
            'community_id': community_id,
            'ip_address': ip_address,
            'match_method': match['MatchMethod'],
            'multiple_matches': match.get('MultipleMatches', False)
        }
    
    # Statistics counters
    location_failures = 0
    validation_failures = 0
    captcha_failures = 0
    duplicate_comm_failures = 0
    whitelist_matches = 0
    whitelist_eligible = 0
    
    # Process each respondent to determine eligibility
    eligible_respondents = []
    ineligible_respondents = []
    
    # Process each matched incentive respondent
    for idx, incentive_row in matched_incentive.iterrows():
        try:
            # Extract response ID
            incentive_id = str(incentive_row.get('ResponseId', f"ID-{idx}"))
            
            # Find matching community info
            if incentive_id not in matching_map:
                continue
                
            match_info = matching_map[incentive_id]
            community_id = match_info['community_id']
            ip_address = match_info['ip_address']
            
            # Find the community row
            community_row = None
            for comm_idx, row in matched_community.iterrows():
                row_id = str(row.get('ResponseId', ''))
                if row_id == community_id:
                    community_row = row
                    break
                    
            if community_row is None:
                continue
            
            # Check if IPs are whitelisted with distance consideration
            community_ip_whitelisted = False
            incentive_ip_whitelisted = False
            
            # Validate community IP whitelist status
            if 'IPAddress' in community_row and pd.notna(community_row['IPAddress']):
                community_ip = str(community_row['IPAddress']).strip()
                community_distance = community_row.get('distance_miles', None)
                community_ip_whitelisted = is_ip_whitelisted(
                    community_ip, ip_whitelist, 
                    distance=community_distance, 
                    max_distance=max_whitelist_distance
                )
                if community_ip_whitelisted:
                    logger.info(f"Community IP {community_ip} is whitelisted")
                
            # Validate incentive IP whitelist status
            if 'IPAddress' in incentive_row and pd.notna(incentive_row['IPAddress']):
                incentive_ip = str(incentive_row['IPAddress']).strip()
                incentive_distance = incentive_row.get('distance_miles', None)
                incentive_ip_whitelisted = is_ip_whitelisted(
                    incentive_ip, ip_whitelist, 
                    distance=incentive_distance,
                    max_distance=max_whitelist_distance
                )
                if incentive_ip_whitelisted:
                    logger.info(f"Incentive IP {incentive_ip} is whitelisted")
                    whitelist_matches += 1
            
            # Either IP being whitelisted should make the combined location valid
            either_ip_whitelisted = community_ip_whitelisted or incentive_ip_whitelisted
            
            # Perform validation checks
            
            # 1. Location check (distance must be <= distance_threshold miles OR either IP is whitelisted)
            community_location_valid = either_ip_whitelisted or (
                'distance_miles' in community_row and 
                pd.notna(community_row['distance_miles']) and 
                float(community_row['distance_miles']) <= distance_threshold
            )
                
            incentive_location_valid = either_ip_whitelisted or (
                'distance_miles' in incentive_row and 
                pd.notna(incentive_row['distance_miles']) and 
                float(incentive_row['distance_miles']) <= distance_threshold
            )
            
            # 2. Completion time check (must be >= min_completion_time seconds)
            community_time_valid = (
                'Duration (in seconds)' in community_row and 
                pd.notna(community_row['Duration (in seconds)']) and 
                float(community_row['Duration (in seconds)']) >= min_completion_time
            )
                
            incentive_time_valid = (
                'Duration (in seconds)' in incentive_row and 
                pd.notna(incentive_row['Duration (in seconds)']) and 
                float(incentive_row['Duration (in seconds)']) >= min_completion_time
            )
            
            # 3. Captcha check (must be >= captcha_threshold)
            community_captcha_valid = not (
                'Q_RecaptchaScore' in community_row and 
                pd.notna(community_row['Q_RecaptchaScore']) and 
                float(community_row['Q_RecaptchaScore']) < captcha_threshold
            )
                
            incentive_captcha_valid = not (
                'Q_RecaptchaScore' in incentive_row and 
                pd.notna(incentive_row['Q_RecaptchaScore']) and 
                float(incentive_row['Q_RecaptchaScore']) < captcha_threshold
            )
            
            # 4. Email uniqueness check
            incentive_email_unique = True
            if 'RecipientEmail' in incentive_row and pd.notna(incentive_row['RecipientEmail']):
                email = str(incentive_row['RecipientEmail']).lower().strip()
                incentive_email_unique = email_frequencies.get(email, 0) <= 1
            
            # 5. IP uniqueness check
            ip = str(incentive_row.get('IPAddress', ''))
            ip_count = ip_frequencies.get(ip, 0)
            incentive_ip_unique = ip_count <= 1
            incentive_ip_shared = ip_count > 1
            incentive_ip_heavily_shared = ip_count > heavily_shared_ip_threshold
                
            # Store IP sharing info for the report
            ip_sharing_info = None
            if incentive_ip_heavily_shared:
                ip_sharing_info = f"IP address used in >{heavily_shared_ip_threshold} submissions (flagged for review)"
            elif incentive_ip_shared:
                ip_sharing_info = "IP address shared with other respondents (possibly from shared facility)"
                
            # Count failed checks for community survey
            community_failed_checks = 0
            community_failed_check_names = []
            
            if not community_time_valid:
                community_failed_checks += 1
                community_failed_check_names.append('valid_time')
                
            if not community_captcha_valid:
                community_failed_checks += 1
                community_failed_check_names.append('valid_captcha')
            
            # Count failed checks for incentive survey
            incentive_failed_checks = 0
            incentive_failed_check_names = []
            
            if not incentive_time_valid:
                incentive_failed_checks += 1
                incentive_failed_check_names.append('valid_time')
                
            if not incentive_captcha_valid:
                incentive_failed_checks += 1
                incentive_failed_check_names.append('valid_captcha')
                
            if not incentive_email_unique:
                incentive_failed_checks += 1
                incentive_failed_check_names.append('unique_email')
                
            if not incentive_ip_unique:
                incentive_failed_checks += 1
                incentive_failed_check_names.append('unique_ip')
                
            # Determine overall validity
            community_valid = (
                community_location_valid and 
                community_failed_checks < validation_failure_threshold and 
                community_captcha_valid
            )
            
            incentive_valid = (
                incentive_location_valid and 
                incentive_failed_checks < validation_failure_threshold and 
                incentive_captcha_valid
            )
            
            # Overall eligibility - both surveys must be valid
            is_eligible = community_valid and incentive_valid
            
            # Log eligibility decision for debugging
            logger.info(f"Eligibility decision for response {incentive_id}:")
            logger.info(f"  Either IP whitelisted: {either_ip_whitelisted}")
            logger.info(f"  Community valid: {community_valid} (location: {community_location_valid}, failed checks: {community_failed_checks}, captcha: {community_captcha_valid})")
            logger.info(f"  Incentive valid: {incentive_valid} (location: {incentive_location_valid}, failed checks: {incentive_failed_checks}, captcha: {incentive_captcha_valid})")
            logger.info(f"  Final eligibility: {is_eligible}")
            
            # Extract respondent data
            respondent_data = extract_respondent_data(incentive_row, community_row, ip_address)
            
            # Add whitelist status
            respondent_data['IPWhitelisted'] = 'Yes' if (incentive_ip_whitelisted or community_ip_whitelisted) else 'No'
            
            # Add IP sharing info if applicable
            if ip_sharing_info:
                respondent_data['IPSharingNotes'] = ip_sharing_info
            
            # Track eligibility
            if is_eligible:
                eligible_respondents.append(respondent_data)
                
                # Count whitelist eligibility
                if incentive_ip_whitelisted or community_ip_whitelisted:
                    whitelist_eligible += 1
            else:
                # Add reason for ineligibility
                reasons = []
                
                if not community_valid:
                    # Check if it's a non-whitelisted location issue
                    if not community_location_valid and not community_ip_whitelisted:
                        reasons.append(f"Community survey location outside {distance_threshold}-mile radius (IP not whitelisted)")
                        location_failures += 1
                    elif community_failed_checks >= validation_failure_threshold:
                        reasons.append(f"Community survey failed {community_failed_checks} validation checks: {', '.join(community_failed_check_names)}")
                        validation_failures += 1
                    elif not community_captcha_valid:
                        reasons.append("Community survey failed captcha check")
                        captcha_failures += 1
                    else:
                        reasons.append("Community survey failed for an unknown reason")
                
                if not incentive_valid:
                    # Check if it's a non-whitelisted location issue
                    if not incentive_location_valid and not incentive_ip_whitelisted:
                        reasons.append(f"Incentive survey location outside {distance_threshold}-mile radius (IP not whitelisted)")
                        location_failures += 1
                    elif incentive_failed_checks >= validation_failure_threshold:
                        reasons.append(f"Incentive survey failed {incentive_failed_checks} validation checks: {', '.join(incentive_failed_check_names)}")
                        validation_failures += 1
                    elif not incentive_captcha_valid:
                        reasons.append("Incentive survey failed captcha check")
                        captcha_failures += 1
                    else:
                        reasons.append("Incentive survey failed for an unknown reason")
                
                respondent_data['Reason'] = " | ".join(reasons)
                ineligible_respondents.append(respondent_data)
                
        except Exception as e:
            logger.error(f"Error processing incentive response {idx}: {str(e)}")
            continue
    
    # Compile statistics
    eligibility_statistics = {
        'eligible': len(eligible_respondents),
        'ineligible': len(ineligible_respondents),
        'location_failures': location_failures,
        'validation_failures': validation_failures,
        'captcha_failures': captcha_failures,
        'duplicate_comm_failures': duplicate_comm_failures,  # This will now be 0
        'whitelist_matches': whitelist_matches,
        'whitelist_eligible': whitelist_eligible
    }
    
    return eligible_respondents, ineligible_respondents, eligibility_statistics


def export_results(eligible_respondents, ineligible_respondents, detailed_statistics, 
                   file_paths, date_range, config):
    """
    Export the results to CSV files and Excel file with tabs for different categories,
    and generate a summary report.
    
    Args:
        eligible_respondents: List of eligible respondent data
        ineligible_respondents: List of ineligible respondent data
        detailed_statistics: Dictionary of statistics from the validation process
        file_paths: Dictionary with output file paths
        date_range: Tuple of (start_date, end_date) strings
        config: Configuration dictionary with validation criteria
    """
    try:
        import pandas as pd
        import os
        from datetime import datetime
    except ImportError:
        logger.error("Error importing required libraries. Make sure pandas and openpyxl are installed.")
        logger.error("Run: pip install pandas openpyxl")
        return
        
    logger.info("Exporting results to files...")
    
    start_date, end_date = date_range
    
    # Convert string dates to appropriate format
    # Expected format from input: YYYY-MM-DD
    # Output format: YYYYMMDD
    start_date_str = start_date.replace('-', '')
    end_date_str = end_date.replace('-', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Create Excel file path with correct dates
    excel_output_file = os.path.join(os.path.dirname(file_paths['eligible_file']), 
                                    f"eligible_respondents_{start_date_str}_to_{end_date_str}_{timestamp}.xlsx")
    
    # Update file paths with correct dates
    for key in file_paths:
        old_path = file_paths[key]
        # Replace any date patterns in the path with the correct dates
        new_path = old_path.replace("_20250215_to_20250409_", f"_{start_date_str}_to_{end_date_str}_")
        file_paths[key] = new_path
    
    # Export eligible respondents
    if eligible_respondents:
        df_eligible = pd.DataFrame(eligible_respondents)
        df_eligible.to_csv(file_paths['eligible_file'], index=False)
        logger.info(f"Exported {len(eligible_respondents)} eligible respondents to {file_paths['eligible_file']}")
        
        # Create Excel file with separate tabs
        try:
            with pd.ExcelWriter(excel_output_file, engine='openpyxl') as writer:
                # Convert Distance column to numeric for filtering
                df_eligible['distance_numeric'] = df_eligible['Distance'].str.replace(' miles', '').astype(float)
                
                # UPDATED LOGIC: "Clear Valid" now includes ALL respondents within distance threshold
                distance_threshold = config.get('distance_threshold', 50)
                
                # Create "Within Range" tab - ALL respondents within distance threshold regardless of whitelist
                within_range = df_eligible[df_eligible['distance_numeric'] <= distance_threshold]
                within_range = within_range.drop(columns=['distance_numeric'])
                within_range.to_excel(writer, sheet_name='Within_Range', index=False)
                logger.info(f"Exported {len(within_range)} respondents within {distance_threshold} miles to Excel tab")
                
                # Create "Whitelisted Only" tab - ONLY respondents outside distance threshold eligible due to whitelist
                whitelisted_only = df_eligible[(df_eligible['distance_numeric'] > distance_threshold) & 
                                             (df_eligible['IPWhitelisted'] == 'Yes')]
                whitelisted_only = whitelisted_only.drop(columns=['distance_numeric'])
                whitelisted_only.to_excel(writer, sheet_name='Whitelisted_Only', index=False)
                logger.info(f"Exported {len(whitelisted_only)} respondents eligible only due to whitelisting to Excel tab")
                
                # Create a tab for all whitelisted respondents (for reference)
                all_whitelisted = df_eligible[df_eligible['IPWhitelisted'] == 'Yes']
                all_whitelisted = all_whitelisted.drop(columns=['distance_numeric'])
                all_whitelisted.to_excel(writer, sheet_name='All_Whitelisted', index=False)
                
                # Add a summary tab
                summary_data = {
                    'Category': ['Total Eligible', 
                              f'Within {distance_threshold} miles', 
                              f'Outside {distance_threshold} miles (Whitelisted Only)',
                              'Total Whitelisted (any distance)'],
                    'Count': [len(df_eligible), len(within_range), 
                             len(whitelisted_only), len(all_whitelisted)]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Export the original data with all columns
                df_eligible.to_excel(writer, sheet_name='All_Eligible', index=False)
            
            logger.info(f"Exported eligible respondents with tabs to {excel_output_file}")
        except Exception as e:
            logger.error(f"Error creating Excel file with tabs: {e}")
            logger.error("Make sure you have openpyxl installed: pip install openpyxl")
    else:
        logger.info("No eligible respondents to export")
    
    # The rest of the export_results function remains the same
    # Export ineligible respondents
    if ineligible_respondents:
        pd.DataFrame(ineligible_respondents).to_csv(file_paths['ineligible_file'], index=False)
        logger.info(f"Exported {len(ineligible_respondents)} ineligible respondents to {file_paths['ineligible_file']}")
    else:
        logger.info("No ineligible respondents to export")
    
    # Export valid responses (eligible respondents) to a separate file
    if eligible_respondents:
        pd.DataFrame(eligible_respondents).to_csv(file_paths['valid_responses_file'], index=False)
        logger.info(f"Exported {len(eligible_respondents)} valid survey responses to {file_paths['valid_responses_file']}")
    
    # Generate summary report with improved clarity
    with open(file_paths['report_file'], 'w') as f:
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
        f.write(f"Maximum whitelist distance: {config.get('max_whitelist_distance', 400)} miles\n\n")
        
        # Report whitelist configuration
        ip_whitelist = config.get('ip_whitelist', [])
        if ip_whitelist:
            f.write(f"IP Whitelist: Enabled with {len(ip_whitelist)} entries\n")
            f.write("\n")
        else:
            f.write("IP Whitelist: None configured\n\n")
        
        f.write("Summary Statistics:\n")
        f.write("-------------------\n")
        f.write(f"Total incentive survey responses: {detailed_statistics.get('total_incentive', 0)}\n")
        f.write(f"Total community survey responses: {detailed_statistics.get('total_community', 0)}\n")
        f.write(f"Respondents who only completed incentive survey: {detailed_statistics.get('incentive_only', 0)}\n")
        
        f.write("\nMatching Statistics:\n")
        f.write("-------------------\n")
        f.write(f"Total IP matches found: {detailed_statistics.get('total_ip_matches', 0)}\n")
        f.write(f"Unique community surveys involved in matches: {detailed_statistics.get('unique_community_matched', 0)}\n")
        f.write(f"Note: Some community surveys matched with multiple incentive surveys, which explains\n")
        f.write(f"      why the number of IP matches can exceed the number of community surveys.\n\n")
        
        # Add more detailed matching statistics
        f.write(f"One-to-one matches (1 incentive to 1 community): {detailed_statistics.get('ip_unique_matches', 0)}\n")
        f.write(f"Multiple matches (incentive surveys sharing community surveys): {detailed_statistics.get('ip_multiple_matches', 0)}\n\n")
        
        f.write("Eligibility Results:\n")
        f.write("-------------------\n")
        f.write(f"Eligible respondents for incentives: {detailed_statistics.get('eligible', 0)}\n")
        f.write(f"Ineligible matched respondents: {detailed_statistics.get('ineligible', 0)}\n")
        f.write(f"Eligibility Note: Each community survey can make multiple incentive surveys eligible.\n")
        f.write(f"                  Manual review recommended to avoid duplicate incentives.\n\n")
        
        # Add whitelist eligibility statistics
        whitelist_matches = detailed_statistics.get('whitelist_matches', 0)
        whitelist_eligible = detailed_statistics.get('whitelist_eligible', 0)
        if whitelist_matches > 0:
            f.write("Whitelist Analysis:\n")
            f.write("------------------\n")
            f.write(f"Total respondents with whitelisted IPs: {whitelist_matches}\n")
            f.write(f"Eligible respondents with whitelisted IPs: {whitelist_eligible}\n")
            whitelist_pct = (whitelist_eligible / whitelist_matches * 100) if whitelist_matches > 0 else 0
            f.write(f"Whitelist eligibility rate: {whitelist_pct:.1f}%\n\n")
        
        if 'error' in detailed_statistics:
            f.write("Errors Encountered:\n")
            f.write("------------------\n")
            f.write(f"{detailed_statistics['error']}\n\n")
        
        f.write("Match Method Analysis:\n")
        f.write("---------------------\n")
        f.write("IPAddress: 100.0%\n\n")
        
        f.write("Eligibility Percentage:\n")
        f.write("---------------------\n")
        eligible_pct = (detailed_statistics.get('eligible', 0) / detailed_statistics.get('total_ip_matches', 1) * 100) if detailed_statistics.get('total_ip_matches', 0) > 0 else 0
        f.write(f"{eligible_pct:.1f}% of matched respondents are eligible for incentives\n\n")
        
        f.write("Ineligibility Reasons:\n")
        f.write("--------------------\n")
        f.write(f"Location outside {config.get('distance_threshold', 50)}-mile radius (non-whitelisted): {detailed_statistics.get('location_failures', 0)}\n")
        f.write(f"Failed {config.get('validation_failure_threshold', 3)}+ validation checks: {detailed_statistics.get('validation_failures', 0)}\n")
        f.write(f"Failed captcha check (score < {config.get('captcha_threshold', 0.5)}): {detailed_statistics.get('captcha_failures', 0)}\n")
        f.write(f"Community survey already matched: {detailed_statistics.get('duplicate_comm_failures', 0)}\n\n")
        
        if ineligible_respondents:
            f.write("Detailed Reasons for Ineligibility:\n")
            f.write("--------------------------------\n")
            
            reason_counts = {}
            for resp in ineligible_respondents:
                if 'Reason' in resp:
                    reasons = resp['Reason'].split(' | ')
                    for reason in reasons:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(ineligible_respondents) * 100)
                f.write(f"{reason}: {count} respondents ({percentage:.1f}%)\n")
        
        # Add Excel report path to output files section
        if eligible_respondents:
            file_paths['excel_report'] = excel_output_file
        
        f.write("\nOutput Files:\n")
        f.write("------------\n")
        for key, path in file_paths.items():
            if key != 'report_file':  # Don't list the report file itself
                f.write(f"{key.replace('_file', '').replace('_', ' ').title()}: {path}\n")
    
    logger.info(f"Generated summary report: {file_paths['report_file']}")

def process_surveys(community_file, incentive_file, start_date, end_date, config=None, output_dir=None):
    """
    Process survey data to identify eligible respondents for incentives.
    
    Args:
        community_file: Path to community survey CSV
        incentive_file: Path to incentive survey CSV
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
        config: Configuration dictionary with validation criteria
        output_dir: Directory for output files
    
    Returns:
        tuple: (file_paths, detailed_statistics)
    """
    if config is None:
        config = DEFAULT_CONFIG
        logger.info("Using default configuration for validation")
    if output_dir is None:
        output_dir = os.path.dirname(community_file)
    
    # Log validation criteria
    logger.info(f"Validation criteria: distance <= {config.get('distance_threshold', 50)} miles, " 
                f"completion time >= {config.get('min_completion_time', 60)} seconds")
    logger.info(f"Captcha threshold: {config.get('captcha_threshold', 0.5)}, "
                f"validation failure threshold: {config.get('validation_failure_threshold', 3)}")
    logger.info(f"Maximum distance for whitelisted IPs: {config.get('max_whitelist_distance', 400)} miles")
    
    ip_whitelist = config.get('ip_whitelist', [])
    if ip_whitelist:
        logger.info(f"Using IP whitelist with {len(ip_whitelist)} entries")
    
    # Convert date strings to datetime objects
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
    date_range = (start_date_dt, end_date_dt)
    
    logger.info(f"Analyzing survey eligibility from {start_date} to {end_date}")
    
    try:
        # Read and prepare survey data
        community_df = read_and_prepare_dataframe(community_file, date_range)
        incentive_df = read_and_prepare_dataframe(incentive_file, date_range)
        
        # Filter for completed surveys
        community_completed = filter_completed_surveys(community_df, 'community')
        incentive_completed = filter_completed_surveys(incentive_df, 'incentive')
        
        # Ensure IP address columns exist
        community_with_ip = ensure_ip_address_column(community_completed, 'community')
        incentive_with_ip = ensure_ip_address_column(incentive_completed, 'incentive')
        
        # Check for whitelisted IPs
        whitelist_count = 0
        if ip_whitelist:
            logger.info("Checking for whitelisted IPs...")
            for _, row in incentive_with_ip.iterrows():
                if pd.notna(row['IPAddress']):
                    # Check if the IP is in the whitelist (ignoring distance for initial count)
                    if is_ip_whitelisted(row['IPAddress'], ip_whitelist):
                        whitelist_count += 1
            
            logger.info(f"Found {whitelist_count} respondents with whitelisted IP addresses")
        
        # Count respondents who completed incentive but not community
        incentive_ips = set(incentive_with_ip['IPAddress'].dropna())
        community_ips = set(community_with_ip['IPAddress'].dropna())
        
        # Unique IP addresses that are in incentive but not in community
        incentive_only_ip_count = len(incentive_ips - community_ips)
        logger.info(f"ANALYSIS: {incentive_only_ip_count} unique IPs appear only in the incentive survey")
        
        # Submissions that are in incentive but not in community (based on IP)
        incentive_only_respondents = sum(~incentive_with_ip['IPAddress'].isin(community_ips))
        logger.info(f"ANALYSIS: {incentive_only_respondents} respondents completed the incentive survey but not the community survey")
        
        # Add distance calculations
        community_with_distance = add_distance_calculations(community_with_ip, 'community')
        incentive_with_distance = add_distance_calculations(incentive_with_ip, 'incentive')
        
        # Print distance statistics
        distance_threshold = config.get('distance_threshold', 50)
        max_whitelist_distance = config.get('max_whitelist_distance', 400)
        
      # Count by distance buckets
        def count_by_distance(df, column='distance_miles'):
            if column not in df.columns:
                return {}
            
            result = {
                'within_threshold': sum(df[column] <= distance_threshold),
                'within_max_whitelist': sum((df[column] > distance_threshold) & (df[column] <= max_whitelist_distance)),
                'beyond_max_whitelist': sum(df[column] > max_whitelist_distance),
                'unknown': sum(df[column].isna())
            }
            return result
        
        community_distance_stats = count_by_distance(community_with_distance)
        incentive_distance_stats = count_by_distance(incentive_with_distance)
        
        logger.info(f"Community survey distance stats: {community_distance_stats}")
        logger.info(f"Incentive survey distance stats: {incentive_distance_stats}")
        
        # Match respondents across surveys
        logger.info("Matching respondents across surveys...")
        matched_incentive, matched_community, matching_details, match_statistics = match_surveys(
            community_with_distance, incentive_with_distance
        )
        
        # If we have no matches, return early with useful error info
        if len(matched_incentive) == 0 or len(matched_community) == 0:
            logger.warning("No matched survey responses found!")
            detailed_statistics = {
                'total_incentive': len(incentive_with_distance),
                'total_community': len(community_with_distance),
                'incentive_only': incentive_only_respondents,
                'total_ip_matches': 0,
                'unique_community_matched': 0,
                'eligible': 0,
                'ineligible': 0,
                'whitelist_matches': whitelist_count
            }
            return {"error": "No matched survey responses found!"}, detailed_statistics
        
        # Evaluate eligibility
        logger.info("Determining eligibility for matched respondents...")
        eligible_respondents, ineligible_respondents, eligibility_statistics = evaluate_eligibility(
            matched_incentive, matched_community, matching_details, config
        )
        
        # Compile detailed statistics
        detailed_statistics = {
            'total_incentive': len(incentive_with_distance),
            'total_community': len(community_with_distance),
            'incentive_only': incentive_only_respondents,
            'total_ip_matches': match_statistics['total_ip_matches'],
            'unique_community_matched': match_statistics['unique_community_matched'],
            'ip_unique_matches': match_statistics['ip_unique_matches'],
            'ip_multiple_matches': match_statistics['ip_multiple_matches'],
            'eligible': eligibility_statistics['eligible'],
            'ineligible': eligibility_statistics['ineligible'],
            'location_failures': eligibility_statistics['location_failures'],
            'validation_failures': eligibility_statistics['validation_failures'],
            'captcha_failures': eligibility_statistics['captcha_failures'],
            'duplicate_comm_failures': eligibility_statistics['duplicate_comm_failures'],
            'whitelist_matches': eligibility_statistics['whitelist_matches'],
            'whitelist_eligible': eligibility_statistics['whitelist_eligible']
        }
        
        # Prepare output file paths
        start_date_str = start_date.replace('-', '')
        end_date_str = end_date.replace('-', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        base = f"{start_date_str}_to_{end_date_str}_{timestamp}"
        file_paths = {
            'eligible_file': os.path.join(output_dir, f"eligible_respondents_{base}.csv"),
            'ineligible_file': os.path.join(output_dir, f"ineligible_respondents_{base}.csv"),
            'valid_responses_file': os.path.join(output_dir, f"valid_responses_{base}.csv"),
            'report_file': os.path.join(output_dir, f"validation_report_{base}.txt")
        }
        # Export results
        export_results(eligible_respondents, ineligible_respondents, detailed_statistics, file_paths, (start_date, end_date), config)
        # Return output file paths and stats
        return file_paths, detailed_statistics
        
    except ValidationError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return {"error": str(ve)}, {}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"Unexpected error: {str(e)}"}, {}


def run_validation():
    """Main function to find files and run the validation process"""
    logger.info("Survey Validation - Starting")
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Load configuration
        config = load_config(args.config)
        
        # Use provided file paths or find files
        community_file = args.community
        incentive_file = args.incentive
        
        # If default files don't exist, try to find them
        if not os.path.exists(community_file) or not os.path.exists(incentive_file):
            logger.info("Default files not found, searching for survey files...")
            community_files, incentive_files = find_survey_files()
            
            if not os.path.exists(community_file) and community_files:
                community_file = community_files[0]
                logger.info(f"Found community survey file: {community_file}")
            
            if not os.path.exists(incentive_file) and incentive_files:
                incentive_file = incentive_files[0]
                logger.info(f"Found incentive survey file: {incentive_file}")
        
        # Check if files exist
        if not os.path.exists(community_file):
            logger.error(f"Community survey file '{community_file}' not found!")
            return
            
        if not os.path.exists(incentive_file):
            logger.error(f"Incentive survey file '{incentive_file}' not found!")
            return
        
        # Set date range
        start_date = args.start_date
        end_date = args.end_date
        
        # Create output directory if it doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            logger.info(f"Created output directory: {args.output_dir}")
        
        # Run the validation script
        logger.info("\nRunning validation with:")
        logger.info(f"Community file: {community_file}")
        logger.info(f"Incentive file: {incentive_file}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Check for whitelist
        ip_whitelist = config.get('ip_whitelist', [])
        if ip_whitelist:
            logger.info(f"Using IP whitelist with {len(ip_whitelist)} entries")
            logger.info(f"Maximum whitelist distance: {config.get('max_whitelist_distance', 400)} miles")
        else:
            logger.info("No IP whitelist configured")
        
        # Process surveys and determine eligibility
        file_paths, detailed_statistics = process_surveys(
            community_file, incentive_file, start_date, end_date, config, args.output_dir
        )
        
        logger.info("\nAnalysis complete!")
        
        # Print summary to console
        print("\nSurvey Validation Summary:")
        print("=========================")
        print(f"Total community survey responses: {detailed_statistics.get('total_community', 0)}")
        print(f"Total incentive survey responses: {detailed_statistics.get('total_incentive', 0)}")
        print(f"Total matched responses: {detailed_statistics.get('total_ip_matches', 0)}")
        print(f"Eligible respondents: {detailed_statistics.get('eligible', 0)}")
        print(f"Ineligible respondents: {detailed_statistics.get('ineligible', 0)}")
        print(f"Whitelisted IPs used: {detailed_statistics.get('whitelist_matches', 0)}")
        print(f"\nOutput files:")
        for key, path in file_paths.items():
            if key != 'report_file':  # Don't list the report file itself
                print(f"- {key.replace('_file', '').replace('_', ' ').title()}: {path}")
        
        # Check if Excel file was created
        excel_file = file_paths.get('excel_report')
        if excel_file and os.path.exists(excel_file):
            print(f"- Excel report with tabs: {excel_file}")
        
    except Exception as e:
        logger.error(f"Error in validation process: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    try:
        run_validation()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
        
