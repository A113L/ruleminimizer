#!/usr/bin/env python3
'''
The script is used to process debugged hashcat rules from files.
Features:
- Full Hashcat Rule Engine simulation for Functional Minimization (Mode 3).
- Multiprocessing (tqdm) for fast Functional Minimization.
- Optional disk usage for initial consolidation of huge files (--use-disk).
- Statistical Cutoff (Mode 2) and Inverse Mode (Mode 4) for dual-phase attacks.
- Pareto Analysis (Cumulative Value) for suggesting cutoff limits.
- Levenshtein Distance Filtering for Semantic Redundancy Removal, optimized with NumPy (optional).
- NEW: Mode 5 - OpenCL-based rules validation and cleanup (hashcat standards).
- NEW: Mode 6 - OpenCL-accelerated Levenshtein distance filtering.
- NEW: GPU-accelerated rule counting for both disk and RAM modes.
- NEW: Option to output results to STDOUT for piping (-o / --output-stdout).
- NEW: Recursive folder search for rule files (max depth 3).
- NEW: Smart processing selection - CPU for large datasets, GPU for smaller ones.
- NEW: Memory safety with warnings at 85% RAM+swap usage.
- NEW: Proper cleanup of temporary files on Ctrl+C interrupt.
'''
import sys
import os
import re
import glob
import signal
from collections import Counter
from typing import List, Tuple, Dict, Callable, Any, Set
import argparse
import tempfile
import multiprocessing
from tqdm import tqdm
import itertools
import psutil  # For memory monitoring

# --- GLOBAL VARIABLES FOR CLEANUP ---
_temp_files_to_cleanup = []

# --- SIGNAL HANDLER FOR CLEANUP ---
def signal_handler(sig, frame):
    """Handle Ctrl+C and other termination signals to clean up temporary files."""
    print(f"\n{Colors.RED}⚠️  INTERRUPT RECEIVED - Cleaning up...{Colors.RESET}")
    
    # Clean up temporary files
    if _temp_files_to_cleanup:
        print(f"{Colors.YELLOW}Cleaning up temporary files...{Colors.RESET}")
        for temp_file in _temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"{Colors.GREEN}✓ Removed temporary file: {temp_file}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.RED}✗ Error removing {temp_file}: {e}{Colors.RESET}")
    
    print(f"{Colors.RED}Script terminated by user.{Colors.RESET}")
    sys.exit(1)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- COLOR CONSTANTS ---
class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    
    # Styles
    UNDERLINE = '\033[4m'
    REVERSE = '\033[7m'

def colorize(text: str, color: str) -> str:
    """Wrap text with color codes"""
    return f"{color}{text}{Colors.RESET}"

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE}{text:^80}{Colors.RESET}")
    print(f"{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE}{'='*80}{Colors.RESET}")

def print_section(text: str):
    """Print a section header"""
    print(f"\n{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE} {text} {Colors.RESET}")

def print_warning(text: str):
    """Print a warning message"""
    print(f"{Colors.BG_YELLOW}{Colors.BOLD}{Colors.BLUE}⚠️  WARNING:{Colors.RESET} {Colors.YELLOW}{text}{Colors.RESET}")

def print_error(text: str):
    """Print an error message"""
    print(f"{Colors.BG_RED}{Colors.BOLD}{Colors.WHITE}❌ ERROR:{Colors.RESET} {Colors.RED}{text}{Colors.RESET}")

def print_success(text: str):
    """Print a success message"""
    print(f"{Colors.BG_GREEN}{Colors.BOLD}{Colors.WHITE}✅ SUCCESS:{Colors.RESET} {Colors.GREEN}{text}{Colors.RESET}")

def print_info(text: str):
    """Print an info message"""
    print(f"{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE}ℹ️  INFO:{Colors.RESET} {Colors.BLUE}{text}{Colors.RESET}")

# --- OPENCL IMPLEMENTATION CHECK ---
PYOPENCL_AVAILABLE = False
try:
    import pyopencl as cl
    import numpy as np
    PYOPENCL_AVAILABLE = True
    print_success("PyOpenCL found. OpenCL-based validation available as Mode 5 & 6.")
    print_success("GPU-accelerated rule counting enabled.")
except ImportError:
    print_warning("PyOpenCL not found. Modes 5 & 6 (OpenCL validation/Levenshtein) will be disabled.")
    print_warning("GPU-accelerated rule counting disabled.")

# --- NUMPY IMPLEMENTATION CHECK ---
NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    print_success("NumPy found. Using optimized Levenshtein distance calculation.")
except ImportError:
    if not PYOPENCL_AVAILABLE:
        print_warning("NumPy not found. Falling back to slower pure Python Levenshtein distance calculation.")

# ==============================================================================
# MEMORY SAFETY FUNCTIONS (Colorized)
# ==============================================================================

def get_memory_usage():
    """Get current memory usage statistics."""
    try:
        virtual_mem = psutil.virtual_memory()
        swap_mem = psutil.swap_memory()
        
        return {
            'ram_used': virtual_mem.used,
            'ram_total': virtual_mem.total,
            'ram_percent': virtual_mem.percent,
            'swap_used': swap_mem.used,
            'swap_total': swap_mem.total,
            'swap_percent': swap_mem.percent,
            'total_used': virtual_mem.used + swap_mem.used,
            'total_available': virtual_mem.total + swap_mem.total,
            'total_percent': (virtual_mem.used + swap_mem.used) / (virtual_mem.total + swap_mem.total) * 100
        }
    except Exception as e:
        print_error(f"Could not monitor memory usage: {e}")
        return None

def format_bytes(bytes_size):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def check_memory_safety(threshold_percent=85):
    """
    Check if memory usage is below safety threshold.
    Returns True if safe, False if approaching limits.
    """
    mem_info = get_memory_usage()
    if not mem_info:
        return True  # Assume safe if we can't monitor
    
    total_percent = mem_info['total_percent']
    
    if total_percent >= threshold_percent:
        print_warning(f"System memory usage at {total_percent:.1f}% (threshold: {threshold_percent}%)")
        print(f"   {Colors.CYAN}RAM:{Colors.RESET} {format_bytes(mem_info['ram_used'])} / {format_bytes(mem_info['ram_total'])} ({mem_info['ram_percent']:.1f}%)")
        print(f"   {Colors.CYAN}Swap:{Colors.RESET} {format_bytes(mem_info['swap_used'])} / {format_bytes(mem_info['swap_total'])} ({mem_info['swap_percent']:.1f}%)")
        return False
    return True

def memory_safe_operation(operation_name, threshold_percent=85):
    """
    Decorator to check memory safety before running memory-intensive operations.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            print_section(f"Memory Check before {operation_name}")
            
            if not check_memory_safety(threshold_percent):
                print_error(f"{operation_name} requires significant memory.")
                print(f"   Current memory usage exceeds {threshold_percent}% threshold.")
                
                response = input(f"{Colors.YELLOW}Continue with {operation_name} anyway? (y/N): {Colors.RESET}").strip().lower()
                if response not in ['y', 'yes']:
                    print_error(f"{operation_name} cancelled due to memory constraints.")
                    return None
            
            print_success(f"Starting {operation_name}...")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def estimate_memory_usage(rules_count, avg_rule_length=50):
    """
    Estimate memory usage for rule processing operations.
    Returns estimated size in bytes.
    """
    # Rough estimation: each rule string + overhead
    estimated_bytes = rules_count * (avg_rule_length + 50)  # 50 bytes overhead per rule
    return estimated_bytes

# ==============================================================================
# A. HASHCAT RULE ENGINE SIMULATION (From the working script)
# ==============================================================================

# [Previous Rule Engine code remains the same, just add colors to print statements]
# Functions for RuleEngine
def i36(string):
    '''Shorter way of converting base 36 string to integer'''
    return int(string, 36)

# --- FUNCTS DICTIONARY ---
FUNCTS: Dict[str, Callable] = {}
FUNCTS[':'] = lambda x, i: x
FUNCTS['l'] = lambda x, i: x.lower()
FUNCTS['u'] = lambda x, i: x.upper()
FUNCTS['c'] = lambda x, i: x.capitalize()
FUNCTS['C'] = lambda x, i: x.capitalize().swapcase()
FUNCTS['t'] = lambda x, i: x.swapcase()

def T(x, i):
    number = i36(i)
    if number >= len(x): return x
    return ''.join((x[:number], x[number].swapcase(), x[number + 1:]))
FUNCTS['T'] = T

FUNCTS['r'] = lambda x, i: x[::-1]
FUNCTS['d'] = lambda x, i: x+x
FUNCTS['p'] = lambda x, i: x*(i36(i)+1)
FUNCTS['f'] = lambda x, i: x+x[::-1]
FUNCTS['{'] = lambda x, i: x[1:]+x[0] if x else x
FUNCTS['}'] = lambda x, i: x[-1]+x[:-1] if x else x
FUNCTS['$'] = lambda x, i: x+i
FUNCTS['^'] = lambda x, i: i+x
FUNCTS['['] = lambda x, i: x[1:]
FUNCTS[']'] = lambda x, i: x[:-1]

def D(x, i):
    idx = i36(i)
    if idx >= len(x): return x
    return x[:idx]+x[idx+1:]
FUNCTS['D'] = D

def x(x, i):
    # Hashcat x functions takes two arguments (start, end)
    start = i36(i[0])
    end = i36(i[1])
    if start < 0 or end < 0 or start > len(x) or end > len(x) or start > end: return "" 
    return x[start:end]
FUNCTS['x'] = x

def O(x, i):
    # Hashcat O functions takes two arguments (start, end)
    start = i36(i[0])
    end = i36(i[1])
    if start < 0 or end < 0 or start > len(x) or end > len(x): return x
    if start > end: return x
    return x[:start]+x[end+1:]
FUNCTS['O'] = O

def i(x, i):
    # Hashcat i functions takes two arguments (pos, char)
    pos = i36(i[0])
    char = i[1]
    if pos > len(x): pos = len(x)
    return x[:pos]+char+x[pos:]
FUNCTS['i'] = i

def o(x, i):
    # Hashcat o functions takes two arguments (pos, char)
    pos = i36(i[0])
    char = i[1]
    if pos >= len(x): return x
    return x[:pos]+char+x[pos+1:]
FUNCTS['o'] = o

FUNCTS["'"] = lambda x, i: x[:i36(i)]
FUNCTS['s'] = lambda x, i: x.replace(i[0], i[1])
FUNCTS['@'] = lambda x, i: x.replace(i, '')

def z(x, i):
    num = i36(i)
    if x: return x[0]*num+x
    return ''
FUNCTS['z'] = z

def Z(x, i):
    num = i36(i)
    if x: return x+x[-1]*num
    return ''
FUNCTS['Z'] = Z
FUNCTS['q'] = lambda x, i: ''.join([a*2 for a in x])

__memorized__ = ['']

def extract_memory(string, args):
    '''Insert section of stored string into current string'''
    if not __memorized__[0]: return string
    try:
        # Note: Your implementation of X uses three arguments, matching hashcat's memory extraction
        pos, length, i = map(i36, args)
        string_list = list(string)
        mem_segment = __memorized__[0][pos:pos+length]
        string_list.insert(i, mem_segment)
        return ''.join(string_list)
    except Exception:
        # Fallback to original string if arguments fail
        return string 
FUNCTS['X'] = extract_memory
FUNCTS['4'] = lambda x, i: x+__memorized__[0]
FUNCTS['6'] = lambda x, i: __memorized__[0]+x

def memorize(string, _):
    ''''Store current string in memory'''
    __memorized__[0] = string
    return string
FUNCTS['M'] = memorize


def rule_regex_gen():
    ''''Generates regex to parse rules'''
    __rules__ = [
        ':', 'l', 'u', 'c', 'C', 't', r'T\w', 'r', 'd', r'p\w', 'f', '{',
        '}', '$.', '^.', '[', ']', r'D\w', r'x\w\w', r'O\w\w', r'i\w.',
        r'o\w.', r"'\w", 's..', '@.', r'z\w', r'Z\w', 'q',
        r'X\w\w\w', '4', '6', 'M'
        ]
    # Build regex, escaping the first character but using raw regex for the arguments
    for i, func in enumerate(__rules__):
        __rules__[i] = re.escape(func[0]) + func[1:].replace(r'\w', '[a-zA-Z0-9]')
    ruleregex = '|'.join(__rules__)
    return re.compile(ruleregex)
__ruleregex__ = rule_regex_gen()


class RuleEngine(object):
    ''' Simplified Rule Engine for functional simulation '''
    def __init__(self, rules: List[str]):
        # Parse all rule strings into a list of lists of function strings
        self.rules = tuple(map(__ruleregex__.findall, rules))

    def apply(self, string: str) -> str:
        ''' 
        Apply all functions in the rule string to a single string and return the result.
        
        CRITICAL FIX: The 'return word' statement was moved outside the inner loop 
        to ensure all functions in a single rule are executed before returning.
        '''
        for rule_functions in self.rules: # self.rules contains one parsed rule (list of functions)
            word = string
            __memorized__[0] = ''
            
            for function in rule_functions: # Iterate over functions in the rule
                try:
                    word = FUNCTS[function[0]](word, function[1:])
                except Exception:
                    pass
            
            # This returns the result after ALL functions in the rule have been applied.
            return word 
        
        return string

# ==============================================================================
# B. HASHCAT RULE CLEANUP IMPLEMENTATION (Based on cleanup-rules.c)
# ==============================================================================

class HashcatRuleCleaner:
    """
    Implements hashcat's rule validation and cleanup logic.
    Based on the official cleanup-rules.c from hashcat.
    """
    
    # [Previous HashcatRuleCleaner code remains the same]
    # Rule operation constants (from hashcat)
    RULE_OP_MANGLE_NOOP             = ':'
    RULE_OP_MANGLE_LREST            = 'l'
    RULE_OP_MANGLE_UREST            = 'u'
    RULE_OP_MANGLE_LREST_UFIRST     = 'c'
    RULE_OP_MANGLE_UREST_LFIRST     = 'C'
    RULE_OP_MANGLE_TREST            = 't'
    RULE_OP_MANGLE_TOGGLE_AT        = 'T'
    RULE_OP_MANGLE_REVERSE          = 'r'
    RULE_OP_MANGLE_DUPEWORD         = 'd'
    RULE_OP_MANGLE_DUPEWORD_TIMES   = 'p'
    RULE_OP_MANGLE_REFLECT          = 'f'
    RULE_OP_MANGLE_ROTATE_LEFT      = '{'
    RULE_OP_MANGLE_ROTATE_RIGHT     = '}'
    RULE_OP_MANGLE_APPEND           = '$'
    RULE_OP_MANGLE_PREPEND          = '^'
    RULE_OP_MANGLE_DELETE_FIRST     = '['
    RULE_OP_MANGLE_DELETE_LAST      = ']'
    RULE_OP_MANGLE_DELETE_AT        = 'D'
    RULE_OP_MANGLE_EXTRACT          = 'x'
    RULE_OP_MANGLE_INSERT           = 'i'
    RULE_OP_MANGLE_OVERSTRIKE       = 'o'
    RULE_OP_MANGLE_TRUNCATE_AT      = "'"
    RULE_OP_MANGLE_REPLACE          = 's'
    RULE_OP_MANGLE_PURGECHAR        = '@'
    RULE_OP_MANGLE_TOGGLECASE_REC   = 'a'
    RULE_OP_MANGLE_DUPECHAR_FIRST   = 'z'
    RULE_OP_MANGLE_DUPECHAR_LAST    = 'Z'
    RULE_OP_MANGLE_DUPECHAR_ALL     = 'q'
    RULE_OP_MANGLE_EXTRACT_MEMORY   = 'X'
    RULE_OP_MANGLE_APPEND_MEMORY    = '4'
    RULE_OP_MANGLE_PREPEND_MEMORY   = '6'
    RULE_OP_MEMORIZE_WORD           = 'M'
    RULE_OP_REJECT_LESS             = '<'
    RULE_OP_REJECT_GREATER          = '>'
    RULE_OP_REJECT_CONTAIN          = '!'
    RULE_OP_REJECT_NOT_CONTAIN      = '/'
    RULE_OP_REJECT_EQUAL_FIRST      = '('
    RULE_OP_REJECT_EQUAL_LAST       = ')'
    RULE_OP_REJECT_EQUAL_AT         = '='
    RULE_OP_REJECT_CONTAINS         = '%'
    RULE_OP_REJECT_MEMORY           = 'Q'
    # hashcat only
    RULE_OP_MANGLE_SWITCH_FIRST     = 'k'
    RULE_OP_MANGLE_SWITCH_LAST      = 'K'
    RULE_OP_MANGLE_SWITCH_AT        = '*'
    RULE_OP_MANGLE_CHR_SHIFTL       = 'L'
    RULE_OP_MANGLE_CHR_SHIFTR       = 'R'
    RULE_OP_MANGLE_CHR_INCR         = '+'
    RULE_OP_MANGLE_CHR_DECR         = '-'
    RULE_OP_MANGLE_REPLACE_NP1      = '.'
    RULE_OP_MANGLE_REPLACE_NM1      = ','
    RULE_OP_MANGLE_DUPEBLOCK_FIRST  = 'y'
    RULE_OP_MANGLE_DUPEBLOCK_LAST   = 'Y'
    RULE_OP_MANGLE_TITLE            = 'E'

    # Maximum rules per line
    MAX_CPU_RULES = 255
    MAX_GPU_RULES = 255

    def __init__(self, mode: int = 1):
        """
        Initialize the rule cleaner.
        mode: 1 = CPU rules, 2 = GPU rules
        """
        if mode not in [1, 2]:
            raise ValueError("Mode must be 1 (CPU) or 2 (GPU)")
        self.mode = mode
        self.max_rules = self.MAX_CPU_RULES if mode == 1 else self.MAX_GPU_RULES

    @staticmethod
    def class_num(c: str) -> bool:
        """Check if character is a digit."""
        return c >= '0' and c <= '9'

    @staticmethod
    def class_upper(c: str) -> bool:
        """Check if character is uppercase letter."""
        return c >= 'A' and c <= 'Z'

    @staticmethod
    def conv_ctoi(c: str) -> int:
        """Convert character to integer (base36)."""
        if HashcatRuleCleaner.class_num(c):
            return ord(c) - ord('0')
        elif HashcatRuleCleaner.class_upper(c):
            return ord(c) - ord('A') + 10
        return -1

    def is_gpu_denied_op(self, op: str) -> bool:
        """Check if operation is denied on GPU."""
        gpu_denied_ops = {
            self.RULE_OP_MANGLE_EXTRACT_MEMORY,
            self.RULE_OP_MANGLE_APPEND_MEMORY,
            self.RULE_OP_MANGLE_PREPEND_MEMORY,
            self.RULE_OP_MEMORIZE_WORD,
            self.RULE_OP_REJECT_LESS,
            self.RULE_OP_REJECT_GREATER,
            self.RULE_OP_REJECT_CONTAIN,
            self.RULE_OP_REJECT_NOT_CONTAIN,
            self.RULE_OP_REJECT_EQUAL_FIRST,
            self.RULE_OP_REJECT_EQUAL_LAST,
            self.RULE_OP_REJECT_EQUAL_AT,
            self.RULE_OP_REJECT_CONTAINS,
            self.RULE_OP_REJECT_MEMORY
        }
        return op in gpu_denied_ops

    def validate_rule(self, rule_line: str) -> bool:
        """
        Validate a single rule line according to hashcat standards.
        Returns True if rule is valid, False otherwise.
        """
        # Remove spaces and check if empty
        clean_line = rule_line.replace(' ', '')
        if not clean_line:
            return False

        rc = 0
        cnt = 0
        pos = 0
        line_len = len(clean_line)

        while pos < line_len:
            op = clean_line[pos]
            
            # Skip spaces (though we already removed them)
            if op == ' ':
                pos += 1
                continue

            # Validate operation and parameters
            try:
                if op == self.RULE_OP_MANGLE_NOOP:
                    pass
                elif op == self.RULE_OP_MANGLE_LREST:
                    pass
                elif op == self.RULE_OP_MANGLE_UREST:
                    pass
                elif op == self.RULE_OP_MANGLE_LREST_UFIRST:
                    pass
                elif op == self.RULE_OP_MANGLE_UREST_LFIRST:
                    pass
                elif op == self.RULE_OP_MANGLE_TREST:
                    pass
                elif op == self.RULE_OP_MANGLE_TOGGLE_AT:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_REVERSE:
                    pass
                elif op == self.RULE_OP_MANGLE_DUPEWORD:
                    pass
                elif op == self.RULE_OP_MANGLE_DUPEWORD_TIMES:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_REFLECT:
                    pass
                elif op == self.RULE_OP_MANGLE_ROTATE_LEFT:
                    pass
                elif op == self.RULE_OP_MANGLE_ROTATE_RIGHT:
                    pass
                elif op == self.RULE_OP_MANGLE_APPEND:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_PREPEND:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_DELETE_FIRST:
                    pass
                elif op == self.RULE_OP_MANGLE_DELETE_LAST:
                    pass
                elif op == self.RULE_OP_MANGLE_DELETE_AT:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_EXTRACT:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_INSERT:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_OVERSTRIKE:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_TRUNCATE_AT:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_REPLACE:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_PURGECHAR:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_TOGGLECASE_REC:
                    pass
                elif op == self.RULE_OP_MANGLE_DUPECHAR_FIRST:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_DUPECHAR_LAST:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_DUPECHAR_ALL:
                    pass
                elif op == self.RULE_OP_MANGLE_DUPEBLOCK_FIRST:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_DUPEBLOCK_LAST:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_SWITCH_FIRST:
                    pass
                elif op == self.RULE_OP_MANGLE_SWITCH_LAST:
                    pass
                elif op == self.RULE_OP_MANGLE_SWITCH_AT:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_CHR_SHIFTL:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_CHR_SHIFTR:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_CHR_INCR:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_CHR_DECR:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_REPLACE_NP1:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_REPLACE_NM1:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_TITLE:
                    pass
                elif op == self.RULE_OP_MANGLE_EXTRACT_MEMORY:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_MANGLE_APPEND_MEMORY:
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_MANGLE_PREPEND_MEMORY:
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_MEMORIZE_WORD:
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_LESS:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_GREATER:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_CONTAIN:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_NOT_CONTAIN:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_EQUAL_FIRST:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_EQUAL_LAST:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_EQUAL_AT:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_CONTAINS:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_MEMORY:
                    if self.mode == 2:  # GPU mode
                        rc = -1
                else:
                    rc = -1  # Unknown operation
            except IndexError:
                rc = -1

            if rc == -1:
                break

            cnt += 1
            pos += 1

            # Check rule count limits
            if cnt > self.max_rules:
                rc = -1
                break

        return rc == 0

    def clean_rules(self, rules_data: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """
        Clean and validate rules according to hashcat standards.
        Returns only valid rules.
        """
        print_section(f"Hashcat Rule Validation ({'GPU' if self.mode == 2 else 'CPU'} Mode)")
        print(f"Validating {colorize(f'{len(rules_data):,}', Colors.CYAN)} rules for {'GPU' if self.mode == 2 else 'CPU'} compatibility...")
        
        valid_rules = []
        invalid_count = 0
        
        for rule, count in tqdm(rules_data, desc="Validating rules"):
            if self.validate_rule(rule):
                valid_rules.append((rule, count))
            else:
                invalid_count += 1
        
        print_success(f"Removed {invalid_count:,} invalid rules. {len(valid_rules):,} valid rules remaining.")
        return valid_rules

# ==============================================================================
# C. FUNCTIONAL MINIMIZATION WITH HASHCAT RULE ENGINE (Colorized)
# ==============================================================================

# Test vector for functional minimization
TEST_VECTOR = [
    "Password", "123456", "ADMIN", "1aB", "QWERTY", 
    "longword", "spec!", "!spec", "a", "b", "c", "0123", 
    "xYz!", "TEST", "tEST", "test", "0", "1", "$^", "lorem", "ipsum"
]

def worker_generate_signature(rule_data: Tuple[str, int]) -> Tuple[str, Tuple[str, int]]:
    """Worker function for multiprocessing pool."""
    rule_text, count = rule_data
    # Re-initialize RuleEngine for each rule
    engine = RuleEngine([rule_text])
    signature_parts: List[str] = []
    
    for test_word in TEST_VECTOR:
        result = engine.apply(test_word)
        signature_parts.append(result)

    signature = '|'.join(signature_parts)
    return signature, (rule_text, count)

@memory_safe_operation("Functional Minimization", 85)
def functional_minimization(data: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """
    Functional minimization using actual hashcat rule engine simulation.
    This removes rules that produce identical outputs for all test vectors.
    """
    print_section("Functional Minimization")
    print_warning("This operation is RAM intensive and may take significant time for large datasets.")
    
    if not data:
        return data
    
    # For very large datasets, warn the user
    if len(data) > 10000:
        print_warning(f"Large dataset detected ({len(data):,} rules).")
        
        # Estimate memory usage
        estimated_mem = estimate_memory_usage(len(data))
        print(f"{Colors.CYAN}[MEMORY]{Colors.RESET} Estimated memory usage: {format_bytes(estimated_mem)}")
        
        response = input(f"{Colors.YELLOW}Continue with functional minimization? (y/N): {Colors.RESET}").strip().lower()
        if response not in ['y', 'yes']:
            print_info("Skipping functional minimization.")
            return data
    
    print_info(f"Using hashcat rule engine simulation with test vector (Length: {len(TEST_VECTOR)})")
    
    signature_map: Dict[str, List[Tuple[str, int]]] = {}
    
    num_processes = multiprocessing.cpu_count()
    print(f"{Colors.CYAN}[MP]{Colors.RESET} Using {num_processes} processes for functional simulation.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(worker_generate_signature, data),
            total=len(data),
            desc="Simulating rules",
            unit=" rules"
        ))
    
    for signature, rule_data in results:
        if signature not in signature_map:
            signature_map[signature] = []
        signature_map[signature].append(rule_data)

    final_best_rules_list: List[Tuple[str, int]] = []
    
    for signature, rules_list in signature_map.items():
        # Sort by count (highest first) to pick the most common rule as the representative
        rules_list.sort(key=lambda x: x[1], reverse=True)
        best_rule_text, _ = rules_list[0]
        # Sum all counts to get the total functional value
        total_count = sum(count for _, count in rules_list)
        final_best_rules_list.append((best_rule_text, total_count))
        
    final_best_rules_list.sort(key=lambda x: x[1], reverse=True)
    
    removed_count = len(data) - len(final_best_rules_list)
    print_success(f"Removed {removed_count:,} functionally redundant rules.")
    print_success(f"Final count: {len(final_best_rules_list):,} unique functional rules.")
    
    return final_best_rules_list

# ==============================================================================
# D. FILE DISCOVERY FUNCTIONS (Colorized)
# ==============================================================================

def find_rule_files(paths: List[str], max_depth: int = 3) -> List[str]:
    """
    Recursively find rule files in directories (max depth 3).
    Supports .rule, .rules, .hr, .hashcat, .txt files and also looks for common hashcat rule file patterns.
    """
    rule_files = []
    rule_extensions = {'.rule', '.rules', '.hr', '.hashcat', '.txt'}
    
    for path in paths:
        if os.path.isfile(path):
            # Single file provided
            file_ext = os.path.splitext(path.lower())[1]
            if file_ext in rule_extensions:
                rule_files.append(path)
                print_success(f"Rule file: {path}")
            else:
                print_warning(f"Not a rule file (wrong extension): {path}")
        
        elif os.path.isdir(path):
            # Directory provided - search recursively
            print_info(f"Scanning directory: {path} (max depth: {max_depth})")
            found_in_dir = 0
            
            for depth in range(max_depth + 1):
                # Search for all rule extensions
                for ext in rule_extensions:
                    pattern = path + '/*' * depth + '*' + ext
                    depth_files = glob.glob(pattern, recursive=True)
                    
                    for file_path in depth_files:
                        if os.path.isfile(file_path) and file_path not in rule_files:
                            rule_files.append(file_path)
                            found_in_dir += 1
                            if depth == 0:
                                print_success(f"Rule file: {file_path}")
                            else:
                                print_success(f"Rule file (depth {depth}): {file_path}")
            
            if found_in_dir == 0:
                print_warning(f"No rule files found in: {path}")
            else:
                print_success(f"Found {found_in_dir} rule files in: {path}")
        
        else:
            print_error(f"Path not found: {path}")
    
    # Remove duplicates and sort
    rule_files = sorted(list(set(rule_files)))
    print_success(f"Found {len(rule_files)} unique rule files to process")
    return rule_files

# ==============================================================================
# E. SMART PROCESSING SELECTION - CPU FOR LARGE DATASETS, GPU FOR SMALLER ONES
# ==============================================================================

def get_processing_recommendation(total_rules: int) -> str:
    """
    Determine the recommended processing method based on dataset size.
    """
    if total_rules <= 10000:
        return "GPU"  # Small datasets benefit from GPU acceleration
    elif total_rules <= 100000:
        return "BOTH"  # Medium datasets can use either
    else:
        return "CPU"  # Large datasets are better on CPU to avoid GPU memory issues

def ask_processing_method(total_rules: int, use_gpu_default: bool = True) -> str:
    """
    Ask user for processing method with intelligent recommendations.
    """
    recommendation = get_processing_recommendation(total_rules)
    
    print_section("Processing Method Selection")
    print(f"Dataset size: {colorize(f'{total_rules:,}', Colors.CYAN)} rules")
    print(f"Recommendation: {colorize(recommendation, Colors.GREEN)}")
    print(f"\n{Colors.BOLD}Available processing methods:{Colors.RESET}")
    print(f" {Colors.CYAN}(1) CPU Processing{Colors.RESET} - Better for large datasets (>100K rules)")
    print(f" {Colors.CYAN}(2) GPU Processing{Colors.RESET} - Faster for small/medium datasets (<100K rules)")
    print(f" {Colors.CYAN}(3) Auto Selection{Colors.RESET} - Let the script choose the best method")
    
    if recommendation == "GPU":
        default_choice = "2"
        print(f"\n{Colors.GREEN}[SUGGESTION]{Colors.RESET} For {total_rules:,} rules, GPU is recommended for best performance")
    elif recommendation == "CPU":
        default_choice = "1" 
        print(f"\n{Colors.GREEN}[SUGGESTION]{Colors.RESET} For {total_rules:,} rules, CPU is recommended to avoid GPU memory issues")
    else:
        default_choice = "3"
        print(f"\n{Colors.GREEN}[SUGGESTION]{Colors.RESET} For {total_rules:,} rules, either method works well")
    
    while True:
        choice = input(f"{Colors.YELLOW}Choose processing method (1=CPU, 2=GPU, 3=Auto) [{default_choice}]: {Colors.RESET}").strip()
        if not choice:
            choice = default_choice
            
        if choice == '1':
            return "CPU"
        elif choice == '2':
            return "GPU"
        elif choice == '3':
            return "AUTO"
        else:
            print_error("Invalid choice. Please enter 1, 2, or 3.")

# ==============================================================================
# F. OPTIMIZED GPU-ACCELERATED RULE COUNTING (Colorized)
# ==============================================================================

class GPURuleCounter:
    def __init__(self):
        if not PYOPENCL_AVAILABLE:
            raise RuntimeError("PyOpenCL not available")
        
        # Initialize OpenCL context and queue
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        
        # Get device info for proper work group sizing
        self.device = self.ctx.devices[0]
        self.max_work_group_size = self.device.max_work_group_size
        
        print_success(f"GPU Device: {self.device.name}")
        print_info(f"Max work group size: {self.max_work_group_size}")
        
        # Build the optimized OpenCL program for rule counting
        self.program = cl.Program(self.ctx, """
        // Optimized hash function for strings (djb2 with early exit)
        unsigned long djb2_hash(__global const unsigned char* str, unsigned int max_len) {
            unsigned long hash = 5381;
            for (unsigned int i = 0; i < max_len; i++) {
                if (str[i] == 0 || str[i] == '\\n') break;
                hash = ((hash << 5) + hash) + str[i]; // hash * 33 + c
            }
            return hash;
        }

        __kernel void count_rules_gpu_simple(
            __global const unsigned char* rules_data,
            __global unsigned long* rule_hashes,
            __global unsigned int* rule_lengths,
            const unsigned int num_rules,
            const unsigned int max_rule_len)
        {
            unsigned int global_id = get_global_id(0);
            if (global_id >= num_rules) return;

            __global const unsigned char* rule_ptr = rules_data + global_id * max_rule_len;
            
            // Calculate rule length and hash in single pass
            unsigned int rule_len = 0;
            unsigned long hash = 5381;
            
            for (unsigned int i = 0; i < max_rule_len; i++) {
                unsigned char c = rule_ptr[i];
                if (c == 0 || c == '\\n') {
                    rule_len = i;
                    break;
                }
                hash = ((hash << 5) + hash) + c;
                rule_len++;
            }
            
            // Store results
            rule_lengths[global_id] = rule_len;
            rule_hashes[global_id] = (rule_len > 0) ? hash : 0;
        }

        __kernel void count_unique_rules_simple(
            __global const unsigned long* rule_hashes,
            __global const unsigned int* rule_lengths,
            __global unsigned char* unique_flags,
            __global unsigned int* occurrence_counts,
            const unsigned int num_rules)
        {
            unsigned int global_id = get_global_id(0);
            if (global_id >= num_rules) return;

            unsigned long current_hash = rule_hashes[global_id];
            unsigned int current_length = rule_lengths[global_id];
            
            if (current_hash == 0 || current_length == 0) {
                unique_flags[global_id] = 0;
                occurrence_counts[global_id] = 0;
                return;
            }

            // Check if this rule is unique by comparing with all previous rules
            unsigned char is_unique = 1;
            unsigned int count = 1;

            for (unsigned int i = 0; i < global_id; i++) {
                if (rule_hashes[i] == current_hash && rule_lengths[i] == current_length) {
                    is_unique = 0;
                    count++;
                    break;
                }
            }

            unique_flags[global_id] = is_unique;
            occurrence_counts[global_id] = count;
        }
        """).build()
        
        # Cache kernel instances to avoid repeated retrieval
        self.hash_kernel = self.program.count_rules_gpu_simple
        self.unique_kernel = self.program.count_unique_rules_simple
    
    def count_rules_gpu_ram(self, rules: List[str]) -> List[Tuple[str, int]]:
        """
        Count rule occurrences using GPU acceleration (optimized for smaller datasets).
        """
        print_section("GPU Rule Counting")
        print(f"Counting {colorize(f'{len(rules):,}', Colors.CYAN)} rules using GPU acceleration...")
        
        if not rules:
            return []
        
        # Prepare rules data
        print_info("Preparing rules data...")
        max_rule_len = max(len(rule) for rule in rules) + 1
        rules_flat = bytearray()
        
        for rule in tqdm(rules, desc="Preparing rules for GPU"):
            rule_bytes = rule.encode('latin-1', 'ignore')
            rules_flat.extend(rule_bytes)
            rules_flat.extend(b'\x00' * (max_rule_len - len(rule_bytes)))
        
        # Create OpenCL buffers
        print_info("Creating GPU buffers...")
        rules_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                             hostbuf=bytes(rules_flat))
        
        # Use optimized data types
        rule_hashes = np.zeros(len(rules), dtype=np.uint64)
        rule_lengths = np.zeros(len(rules), dtype=np.uint32)
        unique_flags = np.zeros(len(rules), dtype=np.uint8)
        occurrence_counts = np.zeros(len(rules), dtype=np.uint32)
        
        hashes_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, rule_hashes.nbytes)
        lengths_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, rule_lengths.nbytes)
        unique_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, unique_flags.nbytes)
        counts_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, occurrence_counts.nbytes)
        
        # Execute kernels with safe work group sizing
        global_size = (len(rules),)
        
        print_info("Executing hash calculation kernel...")
        try:
            # Try with a conservative local size
            local_size = (min(64, len(rules)),) if len(rules) >= 64 else None
            self.hash_kernel(self.queue, global_size, local_size,
                           rules_buf, hashes_buf, lengths_buf,
                           np.uint32(len(rules)), np.uint32(max_rule_len))
        except cl.LogicError:
            # Fall back to no local size
            self.hash_kernel(self.queue, global_size, None,
                           rules_buf, hashes_buf, lengths_buf,
                           np.uint32(len(rules)), np.uint32(max_rule_len))
        
        print_info("Executing unique counting kernel...")
        try:
            self.unique_kernel(self.queue, global_size, local_size,
                             hashes_buf, lengths_buf, unique_buf, counts_buf,
                             np.uint32(len(rules)))
        except cl.LogicError:
            self.unique_kernel(self.queue, global_size, None,
                             hashes_buf, lengths_buf, unique_buf, counts_buf,
                             np.uint32(len(rules)))
        
        # Read results efficiently
        print_info("Reading results...")
        cl.enqueue_copy(self.queue, unique_flags, unique_buf).wait()
        cl.enqueue_copy(self.queue, occurrence_counts, counts_buf).wait()
        
        # Build results
        print_info("Processing results...")
        rule_count_map = {}
        
        for i, rule in tqdm(enumerate(rules), total=len(rules), desc="Processing GPU results"):
            if unique_flags[i] == 1:
                rule_count_map[rule] = occurrence_counts[i]
        
        # Convert to sorted list
        sorted_rules = sorted(rule_count_map.items(), key=lambda x: x[1], reverse=True)
        
        print_success(f"Counting complete: {len(sorted_rules):,} unique rules found")
        return sorted_rules

# ==============================================================================
# G. OPTIMIZED CPU RULE COUNTING (FOR LARGE DATASETS) (Colorized)
# ==============================================================================

def count_rules_cpu_optimized(rules: List[str]) -> List[Tuple[str, int]]:
    """
    Count rule occurrences using optimized CPU method (better for large datasets).
    """
    print_section("CPU Rule Counting")
    print(f"Counting {colorize(f'{len(rules):,}', Colors.CYAN)} rules using optimized CPU method...")
    
    if not rules:
        return []
    
    # Use Counter for efficient counting
    print_info("Counting occurrences...")
    occurrence_counts = Counter(rules)
    
    # Convert to sorted list
    print_info("Sorting results...")
    sorted_rules = occurrence_counts.most_common()
    
    print_success(f"Counting complete: {len(sorted_rules):,} unique rules found")
    return sorted_rules

def count_rules_cpu_chunked(rules: List[str], chunk_size: int = 1000000) -> List[Tuple[str, int]]:
    """
    Count rule occurrences using chunked CPU method for very large datasets.
    """
    print_section("Chunked CPU Rule Counting")
    print(f"Counting {colorize(f'{len(rules):,}', Colors.CYAN)} rules using chunked CPU method...")
    
    if not rules:
        return []
    
    total_chunks = (len(rules) + chunk_size - 1) // chunk_size
    final_counter = Counter()
    
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(rules))
        
        chunk_rules = rules[start_idx:end_idx]
        chunk_counter = Counter(chunk_rules)
        final_counter.update(chunk_counter)
        
        print_info(f"Processed chunk {chunk_idx + 1}/{total_chunks} ({end_idx:,} rules)")
    
    # Convert to sorted list
    print_info("Sorting final results...")
    sorted_rules = final_counter.most_common()
    
    print_success(f"Counting complete: {len(sorted_rules):,} unique rules found")
    return sorted_rules

# ==============================================================================
# H. SMART PROCESSING DISPATCHER (Colorized)
# ==============================================================================

@memory_safe_operation("Rule Counting", 85)
def smart_count_rules(rules: List[str], method: str = "AUTO") -> List[Tuple[str, int]]:
    """
    Smart rule counting that chooses the best method based on dataset size and user preference.
    """
    total_rules = len(rules)
    
    if total_rules == 0:
        return []
    
    # Auto selection logic
    if method == "AUTO":
        if total_rules <= 50000 and PYOPENCL_AVAILABLE:
            method = "GPU"
        elif total_rules <= 1000000:
            method = "CPU"
        else:
            method = "CPU_CHUNKED"
    
    print(f"\n{Colors.BOLD}[PROCESSING]{Colors.RESET} Using {colorize(method, Colors.CYAN)} method for {colorize(f'{total_rules:,}', Colors.CYAN)} rules")
    
    if method == "GPU" and PYOPENCL_AVAILABLE:
        try:
            if total_rules > 500000:
                print_warning("GPU processing recommended for datasets <500K rules")
                print_warning(f"Current dataset: {total_rules:,} rules - consider using CPU")
                
                response = input(f"{Colors.YELLOW}Continue with GPU anyway? (y/N): {Colors.RESET}").strip().lower()
                if response not in ['y', 'yes']:
                    print_info("Switching to CPU method...")
                    method = "CPU"
            
            if method == "GPU":
                counter = GPURuleCounter()
                return counter.count_rules_gpu_ram(rules)
        except Exception as e:
            print_error(f"GPU counting failed: {e}. Falling back to CPU.")
            method = "CPU"
    
    if method == "CPU":
        if total_rules > 2000000:
            print_info(f"Large dataset detected ({total_rules:,} rules), using chunked CPU method")
            return count_rules_cpu_chunked(rules)
        else:
            return count_rules_cpu_optimized(rules)
    
    elif method == "CPU_CHUNKED":
        return count_rules_cpu_chunked(rules)
    
    else:
        # Fallback to CPU if GPU is not available or method is invalid
        print_warning(f"Invalid method '{method}', falling back to CPU")
        return count_rules_cpu_optimized(rules)

# ==============================================================================
# I. MAIN PROCESSING FUNCTIONS WITH SMART SELECTION (Colorized)
# ==============================================================================

def read_file_data(input_filepath: str) -> List[str]:
    """Reads all data from a single file into RAM."""
    if not os.path.exists(input_filepath):
        print_error(f"Input file '{input_filepath}' does not exist.")
        return []
    print_success(f"Reading file: {input_filepath}")
    try:
        with open(input_filepath, 'r', encoding='latin-1', errors='ignore') as f:
            data = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            return data
    except IOError as e:
        print_error(f"File read error for {input_filepath}: {e}")
        return []

def process_disk_data(input_files: List[str], processing_method: str = "AUTO") -> Tuple[List[Tuple[str, int]], int]:
    """Reads, consolidates, and counts data using disk for intermediate storage."""
    all_data_temp_file: str = ""
    total_lines = 0

    print_section("Disk-Based Processing")
    print_info("Initiating disk-based processing to conserve RAM...")
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_f:
        all_data_temp_file = temp_f.name
        _temp_files_to_cleanup.append(all_data_temp_file)  # Register for cleanup
        print_info(f"Consolidating all input data into temporary file: {all_data_temp_file}")
        
        for input_file in input_files:
            try:
                with open(input_file, 'r', encoding='latin-1', errors='ignore') as f:
                    for line in tqdm(f, desc=f"Reading {os.path.basename(input_file)}", unit="lines"):
                        stripped_line = line.strip()
                        if stripped_line and not stripped_line.startswith('#'):
                            temp_f.write(stripped_line + '\n')
                            total_lines += 1
            except IOError as e:
                print_error(f"File read error for {input_file}: {e}")
                
    if total_lines == 0:
        print_error("No valid data found across all files. Exiting disk mode.")
        cleanup_temp_files()
        return [], 0

    print_success(f"Total lines consolidated: {total_lines:,}")
    
    # Read all rules from temp file for processing
    all_rules = []
    with open(all_data_temp_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, total=total_lines, desc="Reading rules for processing"):
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith('#'):
                all_rules.append(stripped_line)
    
    # Use smart counting based on dataset size
    sorted_data = smart_count_rules(all_rules, processing_method)
    
    # Clean up temp file
    cleanup_temp_file(all_data_temp_file)

    return sorted_data, total_lines

def process_ram_data(input_files: List[str], processing_method: str = "AUTO") -> Tuple[List[Tuple[str, int]], int]:
    """Process data entirely in RAM with smart processing selection."""
    all_data: List[str] = []
    for input_file in input_files:
        file_data = read_file_data(input_file)
        if file_data:
            all_data.extend(file_data)
    
    total_lines = len(all_data)
    if not all_data:
        print_error("No valid data found. Exiting.")
        return [], 0

    # Use smart counting based on dataset size
    sorted_data = smart_count_rules(all_data, processing_method)
    
    return sorted_data, total_lines

# ==============================================================================
# J. LEVENSHTEIN FILTERING (Colorized)
# ==============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

@memory_safe_operation("Levenshtein Filtering", 85)
def levenshtein_filter(data: List[Tuple[str, int]], max_distance: int = 2) -> List[Tuple[str, int]]:
    """
    Filter rules based on Levenshtein distance to remove similar rules.
    """
    print_section("Levenshtein Filtering")
    print_warning("This operation can be slow for large datasets.")
    
    if not data:
        return data
    
    if len(data) > 5000:
        print_warning(f"Large dataset ({len(data):,} rules). This may take a while.")
        response = input(f"{Colors.YELLOW}Continue with Levenshtein filtering? (y/N): {Colors.RESET}").strip().lower()
        if response not in ['y', 'yes']:
            return data
    
    # Ask for distance threshold
    while True:
        try:
            distance_input = input(f"{Colors.YELLOW}Enter maximum Levenshtein distance (1-10) [{max_distance}]: {Colors.RESET}").strip()
            if not distance_input:
                break
            max_distance = int(distance_input)
            if 1 <= max_distance <= 10:
                break
            else:
                print_error("Please enter a value between 1 and 10.")
        except ValueError:
            print_error("Please enter a valid number.")
    
    unique_rules = []
    removed_count = 0
    
    for i, (rule, count) in tqdm(enumerate(data), total=len(data), desc="Levenshtein filtering"):
        is_similar = False
        
        # Compare with already accepted rules
        for existing_rule, _ in unique_rules:
            if levenshtein_distance(rule, existing_rule) <= max_distance:
                is_similar = True
                removed_count += 1
                break
        
        if not is_similar:
            unique_rules.append((rule, count))
    
    print_success(f"Removed {removed_count:,} similar rules.")
    print_success(f"Final count: {len(unique_rules):,} unique rules.")
    
    return unique_rules

# ==============================================================================
# K. TEMPORARY FILE CLEANUP FUNCTIONS
# ==============================================================================

def cleanup_temp_file(temp_file: str):
    """Clean up a single temporary file and remove it from the cleanup list."""
    try:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print_info(f"Cleaned up temporary file: {temp_file}")
        if temp_file in _temp_files_to_cleanup:
            _temp_files_to_cleanup.remove(temp_file)
    except OSError as e:
        print_error(f"Error deleting temporary file {temp_file}: {e}")

def cleanup_temp_files():
    """Clean up all registered temporary files."""
    if not _temp_files_to_cleanup:
        return
    
    print_info("Cleaning up temporary files...")
    for temp_file in _temp_files_to_cleanup[:]:  # Use slice copy to avoid modification during iteration
        cleanup_temp_file(temp_file)

# ==============================================================================
# L. INTERACTIVE PROCESSING LOGIC (Colorized)
# ==============================================================================

def analyze_cumulative_value(sorted_data: List[Tuple[str, int]], total_lines: int):
    """Performs Pareto analysis and prints suggestions for MAX_COUNT filtering."""
    if not sorted_data:
        print_error("No data to analyze.")
        return
        
    total_value = sum(count for _, count in sorted_data)
    cumulative_count = 0
    milestones: List[Tuple[int, int]] = []
    target_percentages = [50, 80, 90, 95] 
    next_target = 0
    
    for i, (_, count) in enumerate(sorted_data):
        cumulative_count += count
        current_percentage = (cumulative_count / total_value) * 100
        if next_target < len(target_percentages) and current_percentage >= target_percentages[next_target]:
            milestones.append((target_percentages[next_target], i + 1))
            next_target += 1
        if next_target >= len(target_percentages): 
            break
            
    print_header("CUMULATIVE VALUE ANALYSIS (PARETO) - SUGGESTED CUTOFF LIMITS")
    print(f"Total value (line occurrences) after consolidation: {colorize(f'{total_value:,}', Colors.CYAN)}")
    print(f"Total number of unique rules: {colorize(f'{len(sorted_data):,}', Colors.CYAN)}")

    for target, rules_needed in milestones:
        rules_percentage = (rules_needed / len(sorted_data)) * 100
        color = Colors.GREEN if target <= 80 else Colors.YELLOW if target <= 90 else Colors.RED
        print(f"{color}[{target}% OF VALUE]:{Colors.RESET} Reached with {colorize(f'{rules_needed:,}', Colors.CYAN)} rules. ({rules_percentage:.2f}% of unique rules)")
    
    print(f"{Colors.BOLD}{'-'*60}{Colors.RESET}")
    if milestones:
        last_milestone_rules = milestones[-1][1]
        print(f"{Colors.GREEN}[SUGGESTION]{Colors.RESET} Consider using a limit of: {colorize(f'{last_milestone_rules:,}', Colors.CYAN)} or {colorize(f'{int(last_milestone_rules * 1.1):,}', Colors.CYAN)} for safety.")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")

def hashcat_rule_cleanup(data: List[Tuple[str, int]], mode: int = 1) -> List[Tuple[str, int]]:
    """Clean rules using hashcat's validation standards."""
    print_section(f"Hashcat Rule Cleanup ({'GPU' if mode == 2 else 'CPU'} Mode)")
    cleaner = HashcatRuleCleaner(mode)
    cleaned_data = cleaner.clean_rules(data)
    return cleaned_data

def filter_by_min_occurrence(data: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Filter rules by minimum occurrence count."""
    if not data:
        return data
    max_count = data[0][1]
    suggested = max(1, sum(count for _, count in data) // 1000)
    
    while True:
        try:
            threshold = int(input(f"{Colors.YELLOW}Enter MINIMUM occurrence count (1-{max_count:,}, suggested: {suggested:,}): {Colors.RESET}"))
            if 1 <= threshold <= max_count:
                filtered = [(rule, count) for rule, count in data if count >= threshold]
                print_success(f"Kept {len(filtered):,} rules (min count: {threshold:,})")
                return filtered
            else:
                print_error(f"Please enter a value between 1 and {max_count:,}")
        except ValueError:
            print_error("Please enter a valid number.")

def filter_by_max_rules(data: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Filter rules by maximum number to keep."""
    if not data:
        return data
    max_possible = len(data)
    
    while True:
        try:
            limit = int(input(f"{Colors.YELLOW}Enter MAXIMUM number of rules to keep (1-{max_possible:,}): {Colors.RESET}"))
            if 1 <= limit <= max_possible:
                filtered = data[:limit]
                print_success(f"Kept top {len(filtered):,} rules")
                return filtered
            else:
                print_error(f"Please enter a value between 1 and {max_possible:,}")
        except ValueError:
            print_error("Please enter a valid number.")

def inverse_mode_filter(data: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Inverse mode - keep rules below a certain rank."""
    if not data:
        return data
    max_possible = len(data)
    
    while True:
        try:
            cutoff = int(input(f"{Colors.YELLOW}Enter cutoff rank (rules BELOW this rank will be kept, 1-{max_possible:,}): {Colors.RESET}"))
            if 1 <= cutoff <= max_possible:
                filtered = data[cutoff:]
                print_success(f"Kept {len(filtered):,} rules below rank {cutoff:,}")
                return filtered
            else:
                print_error(f"Please enter a value between 1 and {max_possible:,}")
        except ValueError:
            print_error("Please enter a valid number.")

def save_rules_to_file(data: List[Tuple[str, int]], first_input_file: str):
    """Save current rules to file."""
    if not data:
        print_error("No rules to save!")
        return
        
    first_basename = os.path.basename(os.path.splitext(first_input_file)[0])
    output_file = f"{first_basename}_processed_{len(data)}rules.rule"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for rule, count in data:
                f.write(f"{rule}\n")
        print_success(f"{len(data):,} rules saved to: {output_file}")
    except IOError as e:
        print_error(f"Failed to save file: {e}")

def interactive_processing_loop(sorted_data: List[Tuple[str, int]], total_lines: int, args: argparse.Namespace):
    """Main interactive processing loop after initial counting."""
    
    current_data = sorted_data
    unique_count = len(current_data)
    
    try:
        while True:
            print_header("RULE PROCESSOR - INTERACTIVE MENU")
            print(f"Current dataset: {colorize(f'{unique_count:,}', Colors.CYAN)} unique rules")
            print(f"{Colors.BOLD}{'-'*60}{Colors.RESET}")
            print(f"{Colors.BOLD}FILTERING OPTIONS:{Colors.RESET}")
            print(f" {Colors.GREEN}(1){Colors.RESET} Filter by {Colors.CYAN}MINIMUM OCCURRENCE{Colors.RESET}")
            print(f" {Colors.GREEN}(2){Colors.RESET} Filter by {Colors.CYAN}MAXIMUM NUMBER OF RULES{Colors.RESET} (Statistical Cutoff - TOP N)")
            print(f" {Colors.GREEN}(3){Colors.RESET} Filter by {Colors.CYAN}FUNCTIONAL REDUNDANCY{Colors.RESET} (Logic Minimization) [RAM INTENSIVE]")
            print(f" {Colors.GREEN}(4){Colors.RESET} {Colors.YELLOW}**INVERSE MODE**{Colors.RESET} - Save rules *BELOW* the MAX_COUNT limit")
            if PYOPENCL_AVAILABLE and not args.no_gpu:
                print(f" {Colors.GREEN}(5){Colors.RESET} {Colors.MAGENTA}**HASHCAT CLEANUP**{Colors.RESET} - Validate and clean rules (CPU/GPU compatible)")
                print(f" {Colors.GREEN}(6){Colors.RESET} {Colors.MAGENTA}**LEVENSHTEIN FILTER**{Colors.RESET} - Remove similar rules (GPU-accelerated)")
            print(f" {Colors.BLUE}(p){Colors.RESET} Show {Colors.CYAN}PARETO analysis{Colors.RESET}")
            print(f" {Colors.BLUE}(s){Colors.RESET} {Colors.GREEN}SAVE{Colors.RESET} current rules to file")
            print(f" {Colors.BLUE}(r){Colors.RESET} {Colors.YELLOW}RESET{Colors.RESET} to original dataset")
            print(f" {Colors.BLUE}(q){Colors.RESET} {Colors.RED}QUIT{Colors.RESET} program")
            print(f"{Colors.BOLD}{'-'*60}{Colors.RESET}")
            
            choice = input(f"{Colors.YELLOW}Enter your choice: {Colors.RESET}").strip().lower()
            
            if choice == 'q':
                print_header("THANK YOU FOR USING THE RULE PROCESSOR!")
                break
                
            elif choice == 'p':
                analyze_cumulative_value(current_data, total_lines)
                continue
                
            elif choice == 's':
                save_rules_to_file(current_data, args.input_files[0])
                continue
                
            elif choice == 'r':
                current_data = sorted_data
                unique_count = len(current_data)
                print_success(f"Restored original dataset: {unique_count:,} rules")
                continue
                
            elif choice == '1':
                current_data = filter_by_min_occurrence(current_data)
                unique_count = len(current_data)
                
            elif choice == '2':
                current_data = filter_by_max_rules(current_data)
                unique_count = len(current_data)
                
            elif choice == '3':
                current_data = functional_minimization(current_data)
                unique_count = len(current_data)
                
            elif choice == '4':
                current_data = inverse_mode_filter(current_data)
                unique_count = len(current_data)
                
            elif choice == '5' and PYOPENCL_AVAILABLE and not args.no_gpu:
                # Ask for CPU or GPU compatibility
                print(f"\n{Colors.MAGENTA}[HASHCAT CLEANUP]{Colors.RESET} Choose compatibility mode:")
                print(f" {Colors.CYAN}(1){Colors.RESET} CPU compatibility (all rules allowed)")
                print(f" {Colors.CYAN}(2){Colors.RESET} GPU compatibility (memory/reject rules disabled)")
                mode_choice = input(f"{Colors.YELLOW}Enter mode (1 or 2): {Colors.RESET}").strip()
                mode = 1 if mode_choice == '1' else 2
                current_data = hashcat_rule_cleanup(current_data, mode)
                unique_count = len(current_data)
                
            elif choice == '6' and PYOPENCL_AVAILABLE and not args.no_gpu:
                current_data = levenshtein_filter(current_data, args.levenshtein_max_dist)
                unique_count = len(current_data)
                
            else:
                print_error("Invalid choice. Please try again.")
                continue
            
            # Show updated stats after each operation
            if choice in ['1', '2', '3', '4', '5', '6']:
                print_success(f"Dataset updated: {unique_count:,} unique rules")
                analyze_cumulative_value(current_data, total_lines)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interactive menu interrupted by user.{Colors.RESET}")
        print(f"{Colors.CYAN}Returning to main program...{Colors.RESET}")

def process_multiple_files(args: argparse.Namespace):
    
    # First, find all rule files recursively
    input_files = find_rule_files(args.input_files, max_depth=3)
    
    if not input_files:
        print_error("No rule files found to process!")
        return
    
    print_header("HASHCAT RULE PROCESSOR - MULTI-FILE PROCESSING")
    print(f"Found {colorize(f'{len(input_files)}', Colors.CYAN)} rule files")
    print(f"Processing Mode: {colorize('DISK' if args.use_disk else 'RAM', Colors.CYAN)}")
    print(f"GPU Available: {colorize('YES' if PYOPENCL_AVAILABLE and not args.no_gpu else 'NO', Colors.GREEN if PYOPENCL_AVAILABLE and not args.no_gpu else Colors.RED)}")
    print(f"Levenshtein Filter Max Dist: {colorize(f'{args.levenshtein_max_dist}', Colors.CYAN)}")
    
    # Check initial memory state
    print_section("Initial Memory Status")
    mem_info = get_memory_usage()
    if mem_info:
        print(f"   {Colors.CYAN}RAM:{Colors.RESET} {format_bytes(mem_info['ram_used'])} / {format_bytes(mem_info['ram_total'])} ({mem_info['ram_percent']:.1f}%)")
        print(f"   {Colors.CYAN}Swap:{Colors.RESET} {format_bytes(mem_info['swap_used'])} / {format_bytes(mem_info['swap_total'])} ({mem_info['swap_percent']:.1f}%)")
        print(f"   {Colors.CYAN}Total:{Colors.RESET} {format_bytes(mem_info['total_used'])} / {format_bytes(mem_info['total_available'])} ({mem_info['total_percent']:.1f}%)")
    
    # Determine processing method
    processing_method = "AUTO"
    if args.no_gpu:
        processing_method = "CPU"
        print_info("GPU disabled by user, using CPU processing")
    else:
        # For very large datasets, ask user for preference
        if PYOPENCL_AVAILABLE:
            # Estimate total rules by reading first file
            try:
                with open(input_files[0], 'r', encoding='latin-1', errors='ignore') as f:
                    sample_lines = sum(1 for _ in f)
                estimated_total = sample_lines * len(input_files)
                
                if estimated_total > 100000:  # Only ask for large datasets
                    processing_method = ask_processing_method(estimated_total)
                else:
                    print_info(f"Small dataset estimated ({estimated_total:,} rules), using auto-selection")
            except:
                print_info("Could not estimate dataset size, using auto-selection")
    
    try:
        # 1. Reading, Combining, Counting, and Sorting Data 
        if args.use_disk:
            sorted_data_textual, total_lines = process_disk_data(input_files, processing_method)
        else:
            sorted_data_textual, total_lines = process_ram_data(input_files, processing_method)
        
        if total_lines == 0 or not sorted_data_textual:
            print_error("No valid data to process. Exiting.")
            return

        # 2. Post-processing Stats and Analysis (Textual)
        unique_count_textual = len(sorted_data_textual)
        
        if total_lines > 0:
            unique_percentage = (unique_count_textual / total_lines) * 100
            redundant_lines = total_lines - unique_count_textual
            redundant_percentage = (redundant_lines / total_lines) * 100
        else:
            unique_percentage = 0.0
            redundant_lines = 0
            redundant_percentage = 0.0

        print_section("Initial Processing Results")
        print(f"The consolidated dataset contains {colorize(f'{unique_count_textual:,}', Colors.CYAN)} TEXTUALLY unique entries.")
        print(f"{Colors.GREEN}[STAT]{Colors.RESET} Unique entries are {colorize(f'{unique_percentage:.2f}%', Colors.CYAN)} of the total lines read.")
        print(f"{Colors.YELLOW}[STAT]{Colors.RESET} Redundant lines (duplicates) removed: {colorize(f'{redundant_lines:,}', Colors.CYAN)} ({redundant_percentage:.2f}%)")
        
        # Show Pareto analysis immediately after counting
        analyze_cumulative_value(sorted_data_textual, total_lines)
        
        # Continue with interactive menu for further processing
        interactive_processing_loop(sorted_data_textual, total_lines, args)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.RED}Processing interrupted by user.{Colors.RESET}")
    except Exception as e:
        print_error(f"Unexpected error during processing: {e}")
    finally:
        # Always clean up temporary files
        cleanup_temp_files()

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input_files', nargs='+', 
                       help='Paths to the debug hashcat rule files or directories to process recursively.')
    parser.add_argument('-d', '--use-disk', action='store_true', 
                       help='Use disk (temp files) for initial consolidation to save RAM.')
    parser.add_argument('-ld', '--levenshtein-max-dist', type=int, default=0, 
                       help='Maximum Levenshtein distance for similarity filtering (0 = disabled).')
    parser.add_argument('-o', '--output-stdout', action='store_true', 
                       help='Output the result to STDOUT for piping.')
    parser.add_argument('--no-gpu', action='store_true', 
                       help='Disable GPU acceleration for rule counting.')
    
    args = parser.parse_args()
    
    # Check if psutil is available for memory monitoring
    try:
        import psutil
    except ImportError:
        print_warning("psutil not installed. Memory monitoring disabled.")
        print("Install with: pip install psutil")
        # Create dummy functions
        def get_memory_usage(): return None
        def check_memory_safety(threshold=85): return True
        def memory_safe_operation(op_name, threshold=85):
            def decorator(func):
                return func
            return decorator
    
    try:
        process_multiple_files(args)
    except KeyboardInterrupt:
        print(f"\n{Colors.RED}Script terminated by user.{Colors.RESET}")
    finally:
        # Final cleanup
        cleanup_temp_files()

if __name__ == "__main__":
    main()
