#!/usr/bin/env python3
"""
Debug Tracer Module for Insulin-AI App
Provides runtime debugging, function tracing, and stack monitoring capabilities
"""

import sys
import os
import signal
import traceback
import threading
import time
from typing import Optional, Callable


class DebugTracer:
    """Runtime debugging and tracing utility"""
    
    def __init__(self):
        self.function_tracing_enabled = False
        self.signal_tracing_enabled = False
        self.periodic_dump_thread = None
        self.periodic_dump_active = False
        self.original_trace = None
        
    def enable_signal_tracing(self):
        """Enable signal-based debugging (send USR1 to dump stack)"""
        def signal_handler(signum, frame):
            print(f"\n🔍 DEBUG SIGNAL RECEIVED (PID: {os.getpid()})")
            print("=" * 60)
            traceback.print_stack(frame)
            print("=" * 60)
            
        signal.signal(signal.SIGUSR1, signal_handler)
        self.signal_tracing_enabled = True
        print(f"🚀 Signal debugging enabled for PID {os.getpid()}")
        print(f"   Send signal with: kill -USR1 {os.getpid()}")
        
    def enable_function_tracing(self):
        """Enable function-level tracing"""
        if self.function_tracing_enabled:
            print("⚠️ Function tracing already enabled")
            return
            
        def trace_calls(frame, event, arg):
            if event == 'call':
                filename = frame.f_code.co_filename
                func_name = frame.f_code.co_name
                lineno = frame.f_lineno
                
                # Filter to only show our app functions (not system/library calls)
                if any(name in filename for name in ['insulin', 'psmiles', 'chatbot', 'literature']):
                    print(f"🔍 CALL: {os.path.basename(filename)}:{lineno} -> {func_name}()")
                    
            return trace_calls
        
        self.original_trace = sys.gettrace()
        sys.settrace(trace_calls)
        self.function_tracing_enabled = True
        print("🔄 Function tracing enabled")
        
    def disable_function_tracing(self):
        """Disable function-level tracing"""
        if not self.function_tracing_enabled:
            print("⚠️ Function tracing not currently enabled")
            return
            
        sys.settrace(self.original_trace)
        self.function_tracing_enabled = False
        print("🛑 Function tracing disabled")
        
    def periodic_stack_dump(self, interval: int = 30):
        """Start periodic stack dumps"""
        if self.periodic_dump_active:
            print("⚠️ Periodic dumps already active")
            return
            
        def dump_loop():
            self.periodic_dump_active = True
            print(f"⏰ Starting periodic stack dumps every {interval} seconds")
            
            while self.periodic_dump_active:
                time.sleep(interval)
                if self.periodic_dump_active:  # Check again after sleep
                    print(f"\n⏰ PERIODIC STACK DUMP ({time.strftime('%H:%M:%S')})")
                    print("=" * 50)
                    traceback.print_stack()
                    print("=" * 50)
                    
        self.periodic_dump_thread = threading.Thread(target=dump_loop, daemon=True)
        self.periodic_dump_thread.start()
        
    def stop_periodic_dumps(self):
        """Stop periodic stack dumps"""
        if self.periodic_dump_active:
            self.periodic_dump_active = False
            print("🛑 Periodic stack dumps stopped")
        else:
            print("⚠️ No periodic dumps currently active")
            
    def get_current_stack(self) -> str:
        """Get current stack trace as string"""
        import io
        f = io.StringIO()
        traceback.print_stack(file=f)
        return f.getvalue()
        
    def get_process_info(self) -> dict:
        """Get current process information"""
        import psutil
        try:
            process = psutil.Process(os.getpid())
            return {
                'pid': os.getpid(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'threads': process.num_threads(),
                'working_dir': os.getcwd()
            }
        except ImportError:
            return {
                'pid': os.getpid(),
                'working_dir': os.getcwd()
            }


def enable_runtime_debugging():
    """Enable comprehensive runtime debugging"""
    print("🔍 Enabling runtime debugging...")
    tracer.enable_signal_tracing()
    print("✅ Runtime debugging enabled")


# Global tracer instance
tracer = DebugTracer()

# Cleanup on exit
def _cleanup():
    tracer.stop_periodic_dumps()
    tracer.disable_function_tracing()

import atexit
atexit.register(_cleanup) 