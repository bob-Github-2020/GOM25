#!/usr/bin/env python3

import subprocess
import time
import logging
import os
from typing import Optional

# Configuration
THRESHOLD_MB = 5000  # Memory threshold in MB
CHECK_INTERVAL = 300  # Check every 5 minutes (300 seconds)
PROCESS_NAMES = ['GNSS_CPD_VelocityEstimation_VGG.py']  # Process names to monitor and restart

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('memory_monitor.log'),
        logging.StreamHandler()
    ]
)

def get_free_memory() -> Optional[int]:
    """Get current free memory in MB, returns None on error"""
    try:
        result = subprocess.run(
            ['free', '-m'],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.splitlines():
            if 'Mem:' in line:
                return int(line.split()[3])  # 'free' column
        return None
    except Exception as e:
        logging.error(f"Error getting memory info: {str(e)}")
        return None

def close_ai_processes() -> bool:
    """Close only the AI processes without affecting the monitor's terminal"""
    success = True
    for process_name in PROCESS_NAMES:
        try:
            # Use a more targeted approach to close only the specific processes
            subprocess.run(['pkill', '-f', process_name], check=True)
            logging.info(f"Closed processes: {process_name}")
        except subprocess.CalledProcessError:
            logging.warning(f"No {process_name} processes found")
            success = False
    return success

def restart_ai_processes():
    """Restart the AI processes using the Auto_Run script"""
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        auto_run_script = os.path.join(script_dir, 'Auto_Run_ChangePointCNN.py')
        
        # Run the script in the background
        subprocess.Popen(['python3', auto_run_script])
        logging.info("Started AI processes restart procedure")
        return True
    except Exception as e:
        logging.error(f"Failed to restart AI processes: {str(e)}")
        return False

def main():
    """Main monitoring loop"""
    logging.info("Starting simplified memory monitor")
    
    # Set a unique window title for our monitor terminal to avoid being closed
    try:
        subprocess.run(['echo', '-ne', '\033]0;MemoryMonitor\007'], check=True, shell=True)
    except:
        logging.warning("Could not set terminal title")
    
    while True:
        # Check memory status
        free_mem = get_free_memory()
        
        if free_mem is None:
            logging.error("Could not determine memory status")
        else:
            logging.info(f"Free memory: {free_mem} MB (Threshold: {THRESHOLD_MB} MB)")
            
            # Take action if below threshold
            if free_mem < THRESHOLD_MB:
                logging.warning(
                    f"Memory critical! Free: {free_mem} MB "
                    f"(Below {THRESHOLD_MB} MB)"
                )
                if close_ai_processes():
                    # Add a delay to allow processes to close properly
                    time.sleep(5)
                    restart_ai_processes()
        
        # Wait for the next check
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Memory monitor stopped by user")
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")
