#!/usr/bin/env python3

import subprocess
import time
import os

def run_command_in_terminal(terminal_type, command, working_dir):
    """Run a command in a specific terminal emulator"""
    try:
        if terminal_type == 'gnome-terminal':
            subprocess.Popen([
                'gnome-terminal', '--', 'bash', '-c',
                f'cd {working_dir} && {command}; exec bash'
            ])
        elif terminal_type == 'xterm':
            subprocess.Popen([
                'xterm', '-e', 
                f'bash -c "cd {working_dir} && {command}; bash"'
            ])
        elif terminal_type == 'konsole':
            subprocess.Popen([
                'konsole', '-e', 'bash', '-c',
                f'cd {working_dir} && {command}; exec bash'
            ])
    except Exception as e:
        print(f"Error opening {terminal_type}: {str(e)}")

def main():
    # Define the working directories and command
    base_dirs = ['./AI1', './AI2', './AI3', './AI4', './AI5']
    command_to_run = './GNSS_CPD_VelocityEstimation_VGG.py'
    
    # Find available terminal
    available_terminals = []
    for terminal in ['gnome-terminal', 'xterm', 'konsole']:
        try:
            subprocess.run(['which', terminal], check=True, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            available_terminals.append(terminal)
        except subprocess.CalledProcessError:
            continue
    
    if not available_terminals:
        print("No terminal emulator found!")
        return
    
    terminal_type = available_terminals[0]  # Use the first available terminal
    
    # Start each process with a delay between them
    for i, directory in enumerate(base_dirs):
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist!")
            continue
            
        print(f"Starting process in {directory}")
        run_command_in_terminal(terminal_type, command_to_run, directory)
        
        # Add a delay between starting processes (adjust as needed)
        if i < len(base_dirs) - 1:  # Don't wait after the last one
            time.sleep(5)

if __name__ == "__main__":
    main()
