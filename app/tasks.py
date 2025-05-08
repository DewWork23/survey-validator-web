from app import celery, db
from validate_surveys import process_surveys, load_config
import pandas as pd
import os
from datetime import datetime
import json
import traceback
import logging

logger = logging.getLogger(__name__)

@celery.task(bind=True)
def process_survey_files(self, community_path, incentive_path, start_date, end_date, config_path=None):
    """
    Background task to process survey files
    """
    try:
        # Load config
        config = load_config(config_path) if config_path else None
        
        # Process surveys in smaller chunks
        chunk_size = 500  # Reduced from 1000 to 500 for better memory management
        
        # Convert date strings to datetime objects
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
        
        logger.info(f"Processing surveys from {start_date_dt} to {end_date_dt}")
        
        # Read community survey in chunks
        community_chunks = pd.read_csv(
            community_path, 
            chunksize=chunk_size, 
            sep=',', 
            on_bad_lines='warn',
            engine='python',
            encoding='utf-8',
            error_bad_lines=False,
            warn_bad_lines=True
        )
        
        # Read incentive survey in chunks
        incentive_chunks = pd.read_csv(
            incentive_path,
            chunksize=chunk_size,
            sep=',',
            on_bad_lines='warn',
            engine='python',
            encoding='utf-8',
            error_bad_lines=False,
            warn_bad_lines=True
        )
        
        # Process each chunk
        eligible_respondents = []
        ineligible_respondents = []
        detailed_statistics = {
            'total_incentive': 0,
            'total_community': 0,
            'incentive_only': 0,
            'total_ip_matches': 0,
            'unique_community_matched': 0,
            'eligible': 0,
            'ineligible': 0,
            'whitelist_matches': 0,
            'whitelist_eligible': 0
        }
        
        # Process chunks
        for i, (comm_chunk, inc_chunk) in enumerate(zip(community_chunks, incentive_chunks)):
            try:
                # Update progress
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': i * chunk_size,
                        'total': 'unknown',
                        'status': f'Processing chunk {i + 1}'
                    }
                )
                
                # Convert date columns to datetime
                if 'RecordedDate' in comm_chunk.columns:
                    comm_chunk['RecordedDate'] = pd.to_datetime(comm_chunk['RecordedDate'], errors='coerce')
                if 'RecordedDate' in inc_chunk.columns:
                    inc_chunk['RecordedDate'] = pd.to_datetime(inc_chunk['RecordedDate'], errors='coerce')
                
                # Filter chunks by date range
                comm_chunk = comm_chunk[
                    (comm_chunk['RecordedDate'] >= start_date_dt) & 
                    (comm_chunk['RecordedDate'] <= end_date_dt)
                ]
                inc_chunk = inc_chunk[
                    (inc_chunk['RecordedDate'] >= start_date_dt) & 
                    (inc_chunk['RecordedDate'] <= end_date_dt)
                ]
                
                # Skip empty chunks
                if len(comm_chunk) == 0 and len(inc_chunk) == 0:
                    continue
                
                # Process this chunk
                chunk_eligible, chunk_ineligible, chunk_stats = process_surveys(
                    comm_chunk, inc_chunk, start_date, end_date, config
                )
                
                # Accumulate results
                eligible_respondents.extend(chunk_eligible)
                ineligible_respondents.extend(chunk_ineligible)
                
                # Update statistics
                for key in detailed_statistics:
                    detailed_statistics[key] += chunk_stats.get(key, 0)
                
                # Log progress
                logger.info(f"Processed chunk {i + 1}: {len(chunk_eligible)} eligible, {len(chunk_ineligible)} ineligible")
                
                # Clear memory
                del comm_chunk
                del inc_chunk
                
            except Exception as chunk_error:
                logger.error(f"Error processing chunk {i + 1}: {str(chunk_error)}")
                logger.error(traceback.format_exc())
                continue  # Skip this chunk and continue with the next one
        
        # Generate timestamp for output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Save results
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save eligible respondents
        if eligible_respondents:
            eligible_df = pd.DataFrame(eligible_respondents)
            eligible_filename = f"eligible_respondents_{start_date}_to_{end_date}_{timestamp}.csv"
            eligible_path = os.path.join(results_dir, eligible_filename)
            eligible_df.to_csv(eligible_path, index=False)
        
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
            f.write(f"Total incentive survey responses: {detailed_statistics['total_incentive']}\n")
            f.write(f"Total community survey responses: {detailed_statistics['total_community']}\n")
            f.write(f"Respondents who only completed incentive survey: {detailed_statistics['incentive_only']}\n\n")
            
            f.write("Matching Statistics:\n")
            f.write("-------------------\n")
            f.write(f"Total IP matches found: {detailed_statistics['total_ip_matches']}\n")
            f.write(f"Unique community surveys involved in matches: {detailed_statistics['unique_community_matched']}\n\n")
            
            f.write("Eligibility Results:\n")
            f.write("-------------------\n")
            f.write(f"Eligible respondents for incentives: {detailed_statistics['eligible']}\n")
            f.write(f"Ineligible matched respondents: {detailed_statistics['ineligible']}\n")
        
        return {
            'status': 'SUCCESS',
            'report_file': report_filename,
            'eligible_file': eligible_filename if eligible_respondents else None,
            'ineligible_file': ineligible_filename if ineligible_respondents else None,
            'statistics': detailed_statistics
        }
        
    except Exception as e:
        logger.error(f"Error in process_survey_files: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'status': 'ERROR',
            'error': str(e)
        } 