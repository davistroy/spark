#!/usr/bin/env python3
"""Lightweight Prometheus exporter for NVIDIA GPU metrics via nvidia-smi."""
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler

QUERY_FIELDS = (
    "temperature.gpu,utilization.gpu,utilization.memory,"
    "power.draw,power.limit,memory.total,memory.used,memory.free,"
    "clocks.gr,clocks.mem,fan.speed,pstate"
)

def safe_float(v, default=0):
    try:
        return float(v.replace("[Not Supported]", "0").replace("N/A", "0"))
    except (ValueError, AttributeError):
        return default

def get_gpu_metrics():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + QUERY_FIELDS,
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return "# nvidia-smi failed\n"

        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 12:
            return "# unexpected nvidia-smi output\n"

        temp, gpu_util, mem_util, power_draw, power_limit = parts[0:5]
        mem_total, mem_used, mem_free = parts[5:8]
        clk_gr, clk_mem, fan, pstate = parts[8:12]
        pstate_num = int(pstate.replace("P", "")) if pstate.startswith("P") else 0

        metrics = []
        gauge = lambda name, help_text, value: metrics.extend([
            "# HELP " + name + " " + help_text,
            "# TYPE " + name + " gauge",
            name + '{gpu="0"} ' + str(value),
        ])

        gauge("gpu_temperature_celsius", "GPU temperature in Celsius.", safe_float(temp))
        gauge("gpu_utilization_percent", "GPU utilization percentage.", safe_float(gpu_util))
        gauge("gpu_memory_utilization_percent", "GPU memory utilization percentage.", safe_float(mem_util))
        gauge("gpu_power_draw_watts", "GPU power draw in watts.", safe_float(power_draw))
        gauge("gpu_power_limit_watts", "GPU power limit in watts.", safe_float(power_limit))
        gauge("gpu_memory_total_bytes", "GPU total memory in bytes.", safe_float(mem_total) * 1048576)
        gauge("gpu_memory_used_bytes", "GPU used memory in bytes.", safe_float(mem_used) * 1048576)
        gauge("gpu_memory_free_bytes", "GPU free memory in bytes.", safe_float(mem_free) * 1048576)
        gauge("gpu_clock_graphics_mhz", "GPU graphics clock in MHz.", safe_float(clk_gr))
        gauge("gpu_clock_memory_mhz", "GPU memory clock in MHz.", safe_float(clk_mem))
        gauge("gpu_pstate", "GPU performance state (0=max, 12=min).", pstate_num)

        return "\n".join(metrics) + "\n"
    except Exception as e:
        return "# ERROR: " + str(e) + "\n"

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            body = get_gpu_metrics()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(body.encode())
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 9400), MetricsHandler)
    print("GPU exporter listening on :9400")
    server.serve_forever()
